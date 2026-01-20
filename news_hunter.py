import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime
import concurrent.futures

# ================= 1. 系统配置 =================
st.set_page_config(page_title="A股操盘手 V40", layout="wide", page_icon="☁️")

# 初始化状态
if 'watchlist' not in st.session_state: st.session_state.watchlist = {}
if 'scan_results' not in st.session_state: st.session_state.scan_results = None
if 'diagnosis_result' not in st.session_state: st.session_state.diagnosis_result = None
if 'last_update_str' not in st.session_state: st.session_state.last_update_str = "未刷新"

# 数据迁移兼容
try:
    for code, val in st.session_state.watchlist.items():
        if isinstance(val, str): 
            st.session_state.watchlist[code] = {'name': val, 'cost': 0.0, 'add_time': datetime.now().strftime('%m-%d')}
except: pass

# ================= 2. 🔥 核心：云端数据中心 (Global Cache) =================

@st.cache_data(ttl=3600*4) # 基础信息缓存 4小时 (板块名称等不常变)
def fetch_basic_info():
    try:
        # 板块数据
        df_sector = ak.stock_board_industry_name_em()
        sector_map = dict(zip(df_sector['板块名称'], df_sector['涨跌幅']))
        return sector_map
    except: return {}

@st.cache_data(ttl=60) # 🔥 全市场行情缓存 60秒 (核心加速引擎)
def fetch_market_spot_data():
    """
    这是 V40 的核心。所有模块都共用这一份数据。
    在 Streamlit Cloud 上，这份数据会被缓存到服务器内存中。
    60秒内的所有操作都直接读内存，速度极快。
    """
    try:
        df = ak.stock_zh_a_spot_em()
        # 预处理：转字符串，方便后续匹配
        df['代码'] = df['代码'].astype(str)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600) # 大盘情绪缓存 10分钟
def fetch_market_sentiment_cached():
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        if df.empty: return "未知", 1.0
        last = df.iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        if last['close'] > ma20: return "📈 大盘多头 (安全)", 1.0
        else: return "🌧️ 大盘空头 (轻仓)", 0.8
    except: return "未知环境", 1.0

# ================= 3. 基础算法库 =================

def calculate_atr(df, period=14):
    high_low = df['最高'] - df['最低']
    high_close = np.abs(df['最高'] - df['收盘'].shift())
    low_close = np.abs(df['最低'] - df['收盘'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_kdj(df, n=9, m1=3, m2=3):
    low_list = df['最低'].rolling(window=n).min()
    high_list = df['最高'].rolling(window=n).max()
    rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def get_individual_fund_flow(code):
    try:
        df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
        if df.empty: return 0.0
        df = df.tail(1)
        net_flow = df.iloc[0]['主力净流入-净额']
        return float(net_flow) / 100000000.0 
    except: return 0.0

def get_stock_industry(code):
    try:
        df = ak.stock_individual_info_em(symbol=code)
        val = df[df['item'] == '行业']['value'].values
        return val[0] if len(val) > 0 else "其他"
    except: return "其他"

# ================= 4. 业务逻辑 (对接数据中心) =================

def analyze_stock_core(code, name, spot_row, market_factor=1.0, sector_map=None, strict_mode=True):
    try:
        current_price = spot_row['最新价']
        current_pct = spot_row['涨跌幅']
        pe, turnover = spot_row['市盈率-动态'], spot_row['换手率']
        
        # 1. 基础数据 (K线) - 这个必须实时拉取，无法全局缓存
        df_day = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        if df_day.empty or len(df_day) < 60: return None
        
        # 2. 快速初筛
        close = df_day['收盘'].iloc[-1]
        ma20 = df_day['收盘'].rolling(20).mean().iloc[-1]
        ma60 = df_day['收盘'].rolling(60).mean().iloc[-1]
        vol_5 = df_day['成交量'].tail(5).mean()
        vol_20 = df_day['成交量'].tail(20).mean()
        
        if strict_mode:
            if close < ma20: return None
            if vol_5 < 1.0 * vol_20: return None
        
        # 3. 深入数据
        industry = get_stock_industry(code)
        sector_pct = 0.0
        if sector_map and industry in sector_map:
            sector_pct = sector_map[industry]
            
        individual_flow = get_individual_fund_flow(code)
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 0
        atr_val = calculate_atr(df_day).iloc[-1]
        
        stop_loss_pct = (max(0, close - 2 * atr_val) - close) / close * 100
        bias_60 = (close - ma60) / ma60 * 100
        is_high_risk = bias_60 > 30
        alpha = current_pct - sector_pct
        has_limit_up = (df_day.tail(20)['涨跌幅'] > 9.5).any()
        
        if strict_mode:
            if is_high_risk: return None 
            if sector_pct < -2.0: return None
        
        df_60m = ak.stock_zh_a_hist_min_em(symbol=code, period='60', adjust='qfq')
        
        score = 60.0
        reasons = []
        is_broken = False
        
        if 0 < pe < 60: score += 10
        else: score -= 5
        if turnover > 5.0: score += 10
        if close > ma20: score += 10
        else: score -= 20; is_broken = True
        if vol_ratio > 1.5: score += 10; reasons.append("✅放量")
        
        if alpha > 0: score += 15; reasons.insert(0, "👑强Alpha")
        else: score -= 10
            
        if has_limit_up: score += 15; reasons.insert(0, "🧬妖股")
        
        if individual_flow > 0.3: score += 20; reasons.insert(0, f"💸主力+{individual_flow:.1f}亿")
        elif individual_flow < -0.3: score -= 20; reasons.append(f"🩸主力-{abs(individual_flow):.1f}亿")
            
        advice_60m = "⚖️ 震荡"
        df_60m_data = None
        has_gold_cross = False
        
        if not df_60m.empty:
            df_60m['K'], df_60m['D'], _ = calculate_kdj(df_60m)
            last_60, prev_60 = df_60m.iloc[-1], df_60m.iloc[-2]
            if prev_60['K'] < prev_60['D'] and last_60['K'] > last_60['D']:
                score += 20; reasons.insert(0, "⚡60分金叉"); advice_60m="💎 起爆"; has_gold_cross = True
            elif last_60['K'] < last_60['D']:
                score -= 10; reasons.append("⏳60分死叉"); advice_60m="✋ 回调"
            df_60m_data = df_60m
            
        # 抢筹信号
        day0, day1 = df_day.iloc[-1], df_day.iloc[-2]
        ma20_vol_s = df_day['成交量'].rolling(20).mean()
        force_signal = None
        if day1['成交量'] > 2*ma20_vol_s.iloc[-2] and day1['涨跌幅']>4 and day0['收盘']>day1['开盘']: force_signal="🔥昨抢筹"
        elif day0['成交量'] > 2*ma20_vol_s.iloc[-1] and day0['涨跌幅']>4: force_signal="🔥今抢筹"
        if force_signal: score += 20; reasons.insert(0, force_signal); advice_60m = "🔥 点火"
            
        if is_high_risk: score -= 15; reasons.append("⚠️高位")
        if is_broken: score = min(score, 40); advice_60m="🛑 离场"
        
        score = max(0.0, min(100.00, score * market_factor))
        priority = score + (100 if has_gold_cross and not is_broken else 0) + (50 if alpha > 0 else 0) + (30 if individual_flow > 0.5 else 0)
        
        recent_day = df_day.tail(30).copy()
        recent_day['日期'] = pd.to_datetime(recent_day['日期']).dt.strftime('%Y-%m-%d')
        
        return {
            "代码": code, "名称": name, "行业": industry, 
            "板块涨幅": sector_pct, "个股资金": individual_flow,
            "现价": current_price, "ATR止损": round(stop_loss_pct, 2),
            "综合评分": round(score, 2), "排序权重": round(priority, 2),
            "评分理由": " ".join(reasons), "微操建议": advice_60m,
            "60分数据": df_60m_data, "日线数据": recent_day, "主力信号": force_signal
        }
    except: return None

def analyze_stock_task(args):
    return analyze_stock_core(args[0], args[1], args[2], args[3], args[4], strict_mode=True)

def diagnose_single_stock(code, market_factor, sector_map):
    try:
        # 🔥 直接使用云端缓存，无需再次联网
        spot = fetch_market_spot_data()
        if spot.empty: return None, "行情数据获取失败"
        
        row = spot[spot['代码'] == code]
        if row.empty: return None, "代码不存在"
        
        res = analyze_stock_core(code, row.iloc[0]['名称'], row.iloc[0], market_factor, sector_map, strict_mode=False)
        return res, None
    except Exception as e: return None, str(e)

# ================= 5. 绘图与界面 =================

def draw_mini_chart(df):
    if df is None: return go.Figure()
    mini_data = df.tail(20)
    fig = go.Figure(go.Candlestick(
        x=mini_data['时间'], open=mini_data['开盘'], high=mini_data['最高'], low=mini_data['最低'], close=mini_data['收盘'],
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=80, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return fig

def draw_detail_chart(df, name):
    if df is None: return go.Figure()
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['日期'], open=df['开盘'], high=df['最高'], low=df['最低'], close=df['收盘'], name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'))
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA5'], line=dict(color='orange', width=1), name='MA5'))
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA20'], line=dict(color='blue', width=1), name='MA20'))
    fig.update_layout(title=f"{name} 日线趋势", height=350, xaxis_rangeslider_visible=False, yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'))
    return fig

# --- 侧边栏 ---
with st.sidebar:
    st.header("💸 操盘手 V40 (云端版)")
    
    # 🔥 极速刷新逻辑：直接调用缓存函数
    if st.button("🔄 刷新全市场行情", type="primary"):
        with st.spinner("同步云端数据中心..."):
            # 1. 强制清除旧缓存（确保数据是最新的）
            fetch_market_spot_data.clear()
            # 2. 重新加载并缓存
            df = fetch_market_spot_data()
            st.session_state.last_update_str = datetime.now().strftime('%H:%M:%S')
        st.success(f"已同步 {len(df)} 只股票行情")
        time.sleep(0.5)
        st.rerun()
    
    st.caption(f"数据时间: {st.session_state.last_update_str}")
    st.info("💡 提示：点击上方按钮后，全站所有功能（扫描、自选、诊股）将在 60秒内 **0延迟** 共享这份数据。")

    if st.session_state.watchlist:
        st.markdown("---")
        # 🔥 从缓存中提取自选股数据，速度极快
        df_cache = fetch_market_spot_data()
        
        for code, info in st.session_state.watchlist.items():
            name = info['name']
            cost = info.get('cost', 0)
            
            # 尝试从缓存获取最新价
            curr = cost
            gain = 0.0
            if not df_cache.empty:
                row = df_cache[df_cache['代码'] == str(code)]
                if not row.empty:
                    curr = float(row.iloc[0]['最新价'])
                    
            gain = (curr - cost) / cost * 100 if cost > 0 and curr > 0 else 0
            color = "red" if gain > 0 else ("green" if gain < 0 else "gray")
            
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.markdown(f"**{name}**\n<span style='font-size:12px;color:gray'>{code}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='color:{color};font-weight:bold'>{gain:+.2f}%</span>", unsafe_allow_html=True)
            if c3.button("✕", key=f"del_{code}"):
                del st.session_state.watchlist[code]
                st.rerun()
            st.markdown("---")
    
    page = st.radio("功能模式:", ["⚡ 极速实战扫描", "📊 个股深度诊疗", "📂 资产看板"])

# --- 主页面 ---
if page == "⚡ 极速实战扫描":
    col_env1, col_env2 = st.columns(2)
    with col_env1:
        market_status, market_factor = fetch_market_sentiment_cached()
        st.success(f"🌞 {market_status}") if market_factor >= 1.0 else st.warning(f"🌩️ {market_status}")
    
    with col_env2:
        sector_map = fetch_basic_info() # 读取缓存的板块数据
        st.caption("板块数据已就绪")

    col1, col2 = st.columns([4, 1])
    with col1: st.info("筛选：主板 + 主力 + **资金穿透** | 排序：Alpha + 妖股基因")
    
    if col2.button("🚀 立即扫描", type="primary"):
        with st.spinner("🚀 正在调用云端算力..."):
            try:
                # 🔥 直接从缓存读取，不再下载
                df_spot = fetch_market_spot_data()
                if df_spot.empty:
                    st.error("行情数据未加载，请先点击侧边栏的【刷新全市场行情】！")
                else:
                    mask = (~df_spot['名称'].str.contains("ST") & ~df_spot['代码'].str.startswith(("688", "8", "4", "9")) & (df_spot['换手率'] > 3.0) & (df_spot['市盈率-动态'] < 80))
                    candidates = df_spot[mask].sort_values(by='换手率', ascending=False).head(60) # Cloud资源有限，限制前60以防超时
                    
                    tasks = [(r['代码'], r['名称'], r, market_factor, sector_map) for _, r in candidates.iterrows()]
                    results = []
                    
                    # Streamlit Cloud 建议线程数不要过高，16-20 是安全区
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                        futures = {executor.submit(analyze_stock_task, t): t for t in tasks}
                        for f in concurrent.futures.as_completed(futures):
                            res = f.result()
                            if res: results.append(res)
                    
                    if results:
                        st.session_state.scan_results = pd.DataFrame(results).sort_values(by='排序权重', ascending=False)
                        st.success(f"扫描完成：命中 {len(results)} 只标的")
                    else:
                        st.warning("无符合条件标的")
            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
        df_res = st.session_state.scan_results
        for idx, row in df_res.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([1.8, 3, 3, 1.5])
                c1.markdown(f"**{row['代码']} {row['名称']}**")
                
                sec_pct = row['板块涨幅']
                flow = row['个股资金'] 
                sec_bg = "#fed7d7" if sec_pct > 1.0 else ("#f0fff4" if sec_pct < -1.0 else "#edf2f7")
                flow_color = "red" if flow > 0 else "green"
                
                c1.markdown(f"<span style='background:{sec_bg};padding:2px;font-size:12px'>🏭 {row['行业']} {sec_pct:+.2f}%</span> <span style='color:{flow_color};font-weight:bold'>{flow:+.2f}亿</span>", unsafe_allow_html=True)
                c1.caption(f"评分: **{row['综合评分']:.0f}**")
                
                c2.markdown(f"<span style='font-size:13px;color:#555'>{row['评分理由']}</span>", unsafe_allow_html=True)
                c2.markdown(f"💡 <span style='color:red'>{row['微操建议']}</span>", unsafe_allow_html=True)
                
                if row['60分数据'] is not None:
                    c3.plotly_chart(draw_mini_chart(row['60分数据']), use_container_width=True, key=f"mini_{row['代码']}")
                else: c3.caption("无数据")

                if row['代码'] in st.session_state.watchlist:
                    c4.button("已加", disabled=True, key=f"s_{row['代码']}")
                else:
                    if c4.button("➕", key=f"s_add_{row['代码']}"):
                        st.session_state.watchlist[row['代码']] = {
                            'name': row['名称'], 'cost': row['现价'], 'add_time': datetime.now().strftime('%m-%d')
                        }
                        st.rerun()
                st.markdown("---")

elif page == "📊 个股深度诊疗":
    st.title("🏥 个股诊疗")
    market_status, market_factor = fetch_market_sentiment_cached()
    sector_map = fetch_basic_info()

    c1, c2 = st.columns([3, 1])
    code_in = c1.text_input("输入代码", placeholder="6位代码")
    if c2.button("诊断") and len(code_in)==6:
        with st.spinner("分析中..."):
            res, err = diagnose_single_stock(code_in, market_factor, sector_map)
            if res: st.session_state.diagnosis_result = res
            else: st.error(err)
            
    if st.session_state.diagnosis_result:
        res = st.session_state.diagnosis_result
        k1, k2, k3 = st.columns(3)
        k1.metric("综合评分", f"{res['综合评分']:.0f}")
        k2.metric("建议", res['微操建议'])
        k3.metric("资金", f"{res['个股资金']:+.2f}亿")
        
        st.info(res['评分理由'])
        st.plotly_chart(draw_detail_chart(res['日线数据'], res['名称']), use_container_width=True)
        
        if res['代码'] not in st.session_state.watchlist:
            if st.button(f"➕ 加入自选 ({res['名称']})"):
                st.session_state.watchlist[res['代码']] = {'name': res['名称'], 'cost': res['现价'], 'add_time': datetime.now().strftime('%m-%d')}
                st.rerun()

elif page == "📂 资产看板":
    st.title("📂 资产看板")
    st.caption(f"数据快照: {st.session_state.last_update_str} (点侧边栏刷新更新)")
    
    if not st.session_state.watchlist:
        st.info("暂无自选股")
    else:
        df_cache = fetch_market_spot_data() # 读取缓存
        data = []
        for code, info in st.session_state.watchlist.items():
            curr = info.get('cost', 0)
            daily_pct = 0.0
            
            if not df_cache.empty:
                row = df_cache[df_cache['代码'] == str(code)]
                if not row.empty:
                    curr = float(row.iloc[0]['最新价'])
                    daily_pct = float(row.iloc[0]['涨跌幅'])
            
            cost = info.get('cost', 0)
            total_gain = (curr - cost) / cost * 100 if cost > 0 else 0
            
            data.append({
                "代码": code, "名称": info['name'],
                "现价": curr, "涨跌%": daily_pct, "盈亏%": total_gain
            })
        
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
