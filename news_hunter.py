import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime
import concurrent.futures

# ================= 1. 系统配置 =================
st.set_page_config(page_title="🚀 A股操盘手 V38 (手动版)", layout="wide", page_icon="💰")

# 初始化状态
if 'watchlist' not in st.session_state: st.session_state.watchlist = {}
if 'scan_results' not in st.session_state: st.session_state.scan_results = None
if 'diagnosis_result' not in st.session_state: st.session_state.diagnosis_result = None
if 'sector_map' not in st.session_state: st.session_state.sector_map = {} 
if 'latest_prices' not in st.session_state: st.session_state.latest_prices = {}
if 'last_update_str' not in st.session_state: st.session_state.last_update_str = "未刷新"

# 数据迁移兼容
try:
    for code, val in st.session_state.watchlist.items():
        if isinstance(val, str): 
            st.session_state.watchlist[code] = {'name': val, 'cost': 0.0, 'add_time': datetime.now().strftime('%m-%d')}
except: pass

# ================= 2. 基础算法库 =================

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

def fetch_market_sentiment():
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        if df.empty: return "未知", 1.0
        last = df.iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        if last['close'] > ma20: return "📈 大盘多头 (安全)", 1.0
        else: return "🌧️ 大盘空头 (轻仓)", 0.8
    except: return "未知环境", 1.0

def fetch_sector_map():
    try:
        df = ak.stock_board_industry_name_em()
        return dict(zip(df['板块名称'], df['涨跌幅']))
    except: return {}

def get_individual_fund_flow(code):
    try:
        df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
        if df.empty: return 0.0
        df = df.sort_values(by='日期', ascending=False)
        latest = df.iloc[0]
        net_flow = latest['主力净流入-净额']
        return float(net_flow) / 100000000.0 
    except: return 0.0

def get_stock_industry(code):
    try:
        df = ak.stock_individual_info_em(symbol=code)
        val = df[df['item'] == '行业']['value'].values
        return val[0] if len(val) > 0 else "其他"
    except: return "其他"

# ================= 3. 核心分析逻辑 =================

def analyze_stock_core(code, name, spot_row, market_factor=1.0, sector_map=None, strict_mode=True):
    try:
        current_price = spot_row['最新价']
        current_pct = spot_row['涨跌幅']
        pe, turnover = spot_row['市盈率-动态'], spot_row['换手率']
        
        industry = get_stock_industry(code)
        sector_pct = 0.0
        if sector_map and industry in sector_map:
            sector_pct = sector_map[industry]
            
        individual_flow = get_individual_fund_flow(code)
        
        df_day = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        if df_day.empty or len(df_day) < 60: return None
        
        close = df_day['收盘'].iloc[-1]
        ma20 = df_day['收盘'].rolling(20).mean().iloc[-1]
        ma60 = df_day['收盘'].rolling(60).mean().iloc[-1]
        vol_5 = df_day['成交量'].tail(5).mean()
        vol_20 = df_day['成交量'].tail(20).mean()
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 0
        
        atr_series = calculate_atr(df_day)
        atr_val = atr_series.iloc[-1]
        stop_loss_price = max(0, close - 2 * atr_val)
        stop_loss_pct = (stop_loss_price - close) / close * 100
        
        bias_60 = (close - ma60) / ma60 * 100
        is_high_risk = bias_60 > 30
        alpha = current_pct - sector_pct
        
        recent_20 = df_day.tail(20)
        has_limit_up = (recent_20['涨跌幅'] > 9.5).any()
        
        if strict_mode:
            if close < ma20: return None
            if vol_5 < 1.5 * vol_20: return None
            if is_high_risk: return None 
            if sector_pct < -1.5: return None
        
        df_60m = ak.stock_zh_a_hist_min_em(symbol=code, period='60', adjust='qfq')
        
        score = 60.0
        reasons = []
        is_broken = False
        
        if 0 < pe < 60: score += 10
        else: score -= 5
        if turnover > 3.0: score += 10
        if close > ma20: score += 10
        else: score -= 20; is_broken = True
        if vol_ratio > 1.5: score += 10; reasons.append("✅放量")
        
        if alpha > 0: score += 15; reasons.insert(0, "👑强于板块")
        else: score -= 10
            
        if has_limit_up: score += 15; reasons.insert(0, "🧬妖股基因")
        
        if individual_flow > 0.5:
            score += 20; reasons.insert(0, f"💸主力买{individual_flow:.1f}亿")
        elif individual_flow > 0.1: score += 5
        elif individual_flow < -0.5:
            score -= 20; reasons.append(f"🩸主力逃{abs(individual_flow):.1f}亿")
            
        advice_60m = "⚖️ 分时震荡"
        df_60m_data = None
        has_gold_cross = False
        
        if not df_60m.empty:
            df_60m['K'], df_60m['D'], _ = calculate_kdj(df_60m)
            last_60, prev_60 = df_60m.iloc[-1], df_60m.iloc[-2]
            if prev_60['K'] < prev_60['D'] and last_60['K'] > last_60['D']:
                score += 20; reasons.insert(0, "⚡60分金叉"); advice_60m="💎 完美起爆"; has_gold_cross = True
            elif last_60['K'] < last_60['D']:
                score -= 10; reasons.append("⏳60分死叉"); advice_60m="✋ 暂缓(回调)"
            df_60m_data = df_60m
            
        day0, day1 = df_day.iloc[-1], df_day.iloc[-2]
        ma20_vol_s = df_day['成交量'].rolling(20).mean()
        force_signal = None
        if day1['成交量'] > 2*ma20_vol_s.iloc[-2] and day1['涨跌幅']>4 and day0['收盘']>day1['开盘']: force_signal="🔥主力昨抢筹"
        elif day0['成交量'] > 2*ma20_vol_s.iloc[-1] and day0['涨跌幅']>4: force_signal="🔥主力今抢筹"
        
        if force_signal: 
            score += 20; reasons.insert(0, force_signal); advice_60m = "🔥 主力点火"
            
        if is_high_risk: score -= 15; reasons.append("⚠️高位风险")
            
        tie_breaker = (turnover * 0.01) + (vol_ratio * 0.01)
        score += tie_breaker
        score = score * market_factor
        if is_broken: score = min(score, 40); advice_60m="🛑 离场"
        
        score = min(100.00, score)
        score = max(0.0, score)
        
        priority = score
        if has_gold_cross and not is_broken: priority += 100
        if alpha > 0: priority += 50
        if individual_flow > 0.5: priority += 30 
        
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
        spot = ak.stock_zh_a_spot_em()
        row = spot[spot['代码'] == code]
        if row.empty: return None, "代码不存在"
        res = analyze_stock_core(code, row.iloc[0]['名称'], row.iloc[0], market_factor, sector_map, strict_mode=False)
        return res, None
    except Exception as e: return None, str(e)

# ================= 4. 绘图与数据更新 =================

def draw_mini_chart(df):
    if df is None: return go.Figure()
    mini_data = df.tail(20)
    fig = go.Figure(go.Candlestick(
        x=mini_data['时间'], open=mini_data['开盘'], high=mini_data['最高'], low=mini_data['最低'], close=mini_data['收盘'],
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ))
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0), height=80, 
        xaxis_rangeslider_visible=False, xaxis=dict(visible=False), yaxis=dict(visible=False), 
        showlegend=False, plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def draw_detail_chart(df, name):
    if df is None: return go.Figure()
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['日期'], open=df['开盘'], high=df['最高'], low=df['最低'], close=df['收盘'], name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'))
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA5'], line=dict(color='orange', width=1), name='MA5'))
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA20'], line=dict(color='blue', width=1), name='MA20'))
    fig.update_layout(title=f"{name} 日线趋势", height=350, xaxis_rangeslider_visible=False, plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'))
    return fig

def get_watchlist_updates():
    if not st.session_state.watchlist: return {}
    
    updates = {}
    try:
        df = ak.stock_zh_a_spot_em()
        df['代码'] = df['代码'].astype(str)
        
        for code in st.session_state.watchlist.keys():
            code_str = str(code)
            row = df[df['代码'] == code_str]
            
            if not row.empty:
                price = float(row.iloc[0]['最新价'])
                pct = float(row.iloc[0]['涨跌幅'])
                updates[code_str] = {'price': price, 'pct': pct}
        
        st.session_state.last_update_str = datetime.now().strftime('%H:%M:%S')
        return updates
    except Exception:
        return {}

# ================= 5. 页面布局 =================

with st.sidebar:
    st.header("💸 A股操盘手 V38")
    st.caption("🔒 模式：手动刷新")
    
    if st.button("🔄 手动刷新行情", type="primary"):
        with st.spinner("正在连接交易所..."):
            st.session_state.latest_prices = get_watchlist_updates()
        st.success("刷新成功")
        time.sleep(0.5)
        st.rerun()
    
    st.info(f"🕒 数据锁定于: {st.session_state.last_update_str}")
    
    if not st.session_state.latest_prices and st.session_state.watchlist:
        st.session_state.latest_prices = get_watchlist_updates()
        st.rerun()
        
    latest_prices = st.session_state.latest_prices

    if st.session_state.watchlist:
        st.markdown("---")
        for code, info in st.session_state.watchlist.items():
            name = info['name']
            price_data = latest_prices.get(code, {'price': info.get('cost', 0), 'pct': 0.0})
            curr = price_data['price']
            
            cost = info.get('cost', 0)
            gain = (curr - cost) / cost * 100 if cost > 0 and curr > 0 else 0
            color = "red" if gain > 0 else ("green" if gain < 0 else "gray")
            
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.markdown(f"**{name}**\n<span style='font-size:12px;color:gray'>{code}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='color:{color};font-weight:bold'>{gain:+.2f}%</span>", unsafe_allow_html=True)
            if c3.button("✕", key=f"del_{code}"):
                del st.session_state.watchlist[code]
                st.rerun()
            st.markdown("---")
    else: st.caption("暂无自选股")
    
    page = st.radio("功能模式:", ["⚡ 极速实战扫描", "📊 个股深度诊疗", "📂 资产看板"])

if page == "⚡ 极速实战扫描":
    st.title("⚡ 资金穿透·狙击手 V38")
    
    col_env1, col_env2 = st.columns(2)
    with col_env1:
        market_status, market_factor = fetch_market_sentiment()
        if market_factor < 1.0: st.warning(f"🌩️ {market_status}")
        else: st.success(f"🌞 {market_status}")
    
    with col_env2:
        if st.button("🔄 刷新板块数据"):
            with st.spinner("更新中..."):
                st.session_state.sector_map = fetch_sector_map()
        
        if 'sector_map' not in st.session_state: st.session_state.sector_map = fetch_sector_map()
        sector_map = st.session_state.sector_map

    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1: st.info("筛选：主板 + 主力 + **资金穿透** | 排序：Alpha + 妖股基因")
    
    if col2.button("🚀 立即扫描", type="primary"):
        with st.spinner("🚀 全市场资金扫描中..."):
            try:
                df_spot = ak.stock_zh_a_spot_em()
                mask = (~df_spot['名称'].str.contains("ST") & ~df_spot['代码'].str.startswith(("688", "8", "4", "9")) & (df_spot['换手率'] > 3.0) & (df_spot['市盈率-动态'] < 80))
                candidates = df_spot[mask].sort_values(by='换手率', ascending=False).head(80)
                
                tasks = [(r['代码'], r['名称'], r, market_factor, sector_map) for _, r in candidates.iterrows()]
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(analyze_stock_task, t): t for t in tasks}
                    for f in concurrent.futures.as_completed(futures):
                        res = f.result()
                        if res: results.append(res)
                
                if results:
                    st.session_state.scan_results = pd.DataFrame(results).sort_values(by='排序权重', ascending=False)
                    st.success(f"命中 {len(results)} 只标的。")
                else:
                    st.session_state.scan_results = pd.DataFrame()
                    st.warning("无符合条件标的。")
            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
        df_res = st.session_state.scan_results
        st.subheader(f"🔥 狙击目标 (Top {len(df_res)})")
        
        for idx, row in df_res.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([1.8, 3, 3, 1.5])
                c1.markdown(f"**{row['代码']} {row['名称']}**")
                
                sec_pct = row['板块涨幅']
                flow = row['个股资金'] 
                sec_color = "#f56565" if sec_pct > 1.0 else ("#48bb78" if sec_pct < -1.0 else "#4a5568")
                sec_bg = "#fed7d7" if sec_pct > 1.0 else ("#f0fff4" if sec_pct < -1.0 else "#edf2f7")
                
                flow_str = f"💰+{flow:.2f}亿" if flow > 0 else f"💸{flow:.2f}亿"
                flow_color = "red" if flow > 0 else "green"
                
                c1.markdown(f"<span style='background:{sec_bg};color:{sec_color};padding:2px 6px;border-radius:4px;font-size:12px;font-weight:bold'>🏭 {row['行业']} {sec_pct:+.2f}%</span> <span style='font-size:12px;color:{flow_color};font-weight:bold'>{flow_str}</span>", unsafe_allow_html=True)
                c1.caption(f"评分: **{row['综合评分']:.2f}**")
                
                tags = row['评分理由'].split(" ")
                tag_html = ""
                for t in tags:
                    if "👑" in t: color, bg = "#d69e2e", "#fefcbf"
                    elif "🧬" in t: color, bg = "#805ad5", "#e9d8fd"
                    elif "主力" in t or "买" in t: color, bg = "#c53030", "#fff5f5"
                    elif "金叉" in t: color, bg = "#2f855a", "#f0fff4"
                    else: color, bg = "#4a5568", "#edf2f7"
                    tag_html += f"<span style='color:{color};background:{bg};padding:2px 6px;border-radius:4px;font-size:12px;margin-right:4px;display:inline-block;margin-bottom:4px'>{t}</span>"
                c2.markdown(tag_html, unsafe_allow_html=True)
                
                adv_color = "red" if "起爆" in row['微操建议'] or "点火" in row['微操建议'] else "gray"
                c2.markdown(f"<span style='color:{adv_color};font-size:14px'>💡 {row['微操建议']}</span> | <span style='font-size:12px;color:gray'>ATR止损: {row['ATR止损']}%</span>", unsafe_allow_html=True)
                
                if row['60分数据'] is not None:
                    c3.plotly_chart(draw_mini_chart(row['60分数据']), use_container_width=True, key=f"mini_{row['代码']}")
                else: c3.caption("无数据")

                if row['代码'] in st.session_state.watchlist:
                    c4.button("✅ 已在自选", disabled=True, key=f"s_added_{row['代码']}")
                else:
                    if c4.button("➕ 加入", key=f"s_add_{row['代码']}"):
                        st.session_state.watchlist[row['代码']] = {
                            'name': row['名称'], 'cost': row['现价'], 'add_time': datetime.now().strftime('%m-%d')
                        }
                        st.rerun()
                st.markdown("---")

elif page == "📊 个股深度诊疗":
    st.title("🏥 个股深度诊疗 V38")
    market_status, market_factor = fetch_market_sentiment()
    if 'sector_map' not in st.session_state: st.session_state.sector_map = fetch_sector_map()
    sector_map = st.session_state.sector_map

    c1, c2 = st.columns([3, 1])
    code_in = c1.text_input("输入代码", placeholder="6位代码")
    if c2.button("诊断") and len(code_in)==6:
        with st.spinner("深度分析中..."):
            res, err = diagnose_single_stock(code_in, market_factor, sector_map)
            if res: st.session_state.diagnosis_result = res
            else: st.error(err)
            
    if st.session_state.diagnosis_result:
        res = st.session_state.diagnosis_result
        k1, k2, k3, k4 = st.columns([1.2, 3, 1.2, 1.2])
        
        s = res['综合评分']
        s_color = "inverse" if s > 80 else "normal"
        k1.metric("综合评分", f"{s:.2f}", delta_color=s_color)
        
        adv = res['微操建议']
        adv_col = "inverse" if "起爆" in adv or "点火" in adv else "off"
        k2.metric("短期建议", adv, delta_color=adv_col)
        
        sec_pct = res['板块涨幅']
        flow = res['个股资金']
        k3.metric("资金净流", f"{flow:+.2f}亿", delta=f"板块 {sec_pct:+.2f}%")
        k4.metric("ATR止损", f"{res['ATR止损']}%")
        
        st.info(f"评分理由: {res['评分理由']}")
        st.plotly_chart(draw_detail_chart(res['日线数据'], res['名称']), use_container_width=True)
        
        if res['代码'] in st.session_state.watchlist:
            st.button("✅ 已在自选", disabled=True)
        else:
            if st.button(f"➕ 加入自选 ({res['名称']})"):
                st.session_state.watchlist[res['代码']] = {'name': res['名称'], 'cost': res['现价'], 'add_time': datetime.now().strftime('%m-%d')}
                st.rerun()

elif page == "📂 资产看板":
    st.title("📂 资产看板")
    st.caption(f"数据快照时间: {st.session_state.last_update_str} (请手动点击刷新获取最新数据)")
    
    if st.button("🔄 手动刷新列表", type="primary"):
        with st.spinner("正在拉取最新行情..."):
            st.session_state.latest_prices = get_watchlist_updates()
        st.rerun()

    if not st.session_state.watchlist:
        st.info("暂无自选股，请去扫描或诊股页面添加。")
    else:
        data = []
        for code, info in st.session_state.watchlist.items():
            price_data = latest_prices.get(code, {'price': info.get('cost', 0), 'pct': 0.0})
            curr = price_data['price']
            daily_pct = price_data['pct'] 
            
            cost = info.get('cost', 0)
            total_gain = (curr - cost) / cost * 100 if cost > 0 else 0
            
            data.append({
                "代码": code,
                "名称": info['name'],
                "成本": cost,
                "现价": curr,
                "当日涨跌%": daily_pct, 
                "累计盈亏%": total_gain
            })
        
        st.dataframe(
            pd.DataFrame(data), 
            column_config={
                "成本": st.column_config.NumberColumn(format="¥%.2f"),
                "现价": st.column_config.NumberColumn(format="¥%.2f"),
                "当日涨跌%": st.column_config.NumberColumn(format="%.2f%%"),
                "累计盈亏%": st.column_config.NumberColumn(format="%.2f%%")
            }, 
            hide_index=True, 
            use_container_width=True
        )
