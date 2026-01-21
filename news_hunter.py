import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import time
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import concurrent.futures
from github import Github, GithubException

# ================= 1. 系统配置 =================
st.set_page_config(page_title="A股操盘手 V55 (云端增强版)", layout="wide", page_icon="⚡")

# --- 核心：GitHub 云端持久化层 ---
USER_DATA_FILE = "sentinel_userdata.json"
MARKET_DATA_FILE = "market_snapshot.json"

def get_github_repo():
    """获取 GitHub 仓库对象"""
    try:
        # 优先尝试从 secrets 获取，如果没有则返回 None
        if "GITHUB_TOKEN" not in st.secrets:
            return None
        token = st.secrets["GITHUB_TOKEN"]
        repo_name = st.secrets["REPO_NAME"]
        g = Github(token)
        return g.get_repo(repo_name)
    except Exception as e:
        # print(f"GitHub 连接失败: {e}") # 调试用
        return None

def load_userdata():
    """加载用户自选和持仓数据"""
    if 'user_data_loaded' in st.session_state:
        return {"watchlist": st.session_state.watchlist, "portfolio": st.session_state.strategy_portfolio}
    
    repo = get_github_repo()
    if not repo: 
        return {"watchlist": {}, "portfolio": {}}
        
    try:
        contents = repo.get_contents(USER_DATA_FILE)
        data = json.loads(contents.decoded_content.decode("utf-8"))
        st.session_state.user_data_loaded = True
        return data
    except Exception:
        # 文件不存在或解析失败，返回空结构
        return {"watchlist": {}, "portfolio": {}}

def save_userdata():
    """保存用户数据到 GitHub"""
    repo = get_github_repo()
    if not repo: return
    
    data = {
        "watchlist": st.session_state.watchlist,
        "portfolio": st.session_state.strategy_portfolio,
        "last_save": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    json_str = json.dumps(data, ensure_ascii=False, indent=4)
    
    try:
        try:
            contents = repo.get_contents(USER_DATA_FILE)
            repo.update_file(path=USER_DATA_FILE, message="[Auto] User Data", content=json_str, sha=contents.sha)
        except Exception:
            repo.create_file(path=USER_DATA_FILE, message="[Init] User Data", content=json_str)
    except Exception as e:
        st.error(f"保存用户数据失败: {e}")

def save_market_snapshot(df):
    """保存行情快照到 GitHub"""
    repo = get_github_repo()
    if not repo: 
        st.error("未配置 GitHub Token，无法备份行情！")
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    utc_now = datetime.utcnow()
    bj_now = utc_now + timedelta(hours=8)
    time_str = bj_now.strftime('%Y-%m-%d %H:%M:%S')
    
    snapshot_data = {
        "timestamp": time_str,
        "count": len(df),
        "data": df.to_dict(orient='records')
    }
    # 使用紧凑格式节省空间
    json_str = json.dumps(snapshot_data, ensure_ascii=False, separators=(',', ':'))
    
    try:
        try:
            contents = repo.get_contents(MARKET_DATA_FILE)
            repo.update_file(path=MARKET_DATA_FILE, message=f"[Snapshot] {time_str}", content=json_str, sha=contents.sha)
        except Exception:
            repo.create_file(path=MARKET_DATA_FILE, message=f"[Init] {time_str}", content=json_str)
        
        st.toast(f"✅ 云端备份成功！时间戳: {time_str}")
        return time_str
    except Exception as e:
        st.error(f"云备份失败: {e}")
        return time_str

def load_market_snapshot():
    """从 GitHub 加载行情快照"""
    repo = get_github_repo()
    if not repo: 
        return pd.DataFrame(), "未连接GitHub"
        
    try:
        contents = repo.get_contents(MARKET_DATA_FILE)
        data_packet = json.loads(contents.decoded_content.decode("utf-8"))
        df = pd.DataFrame(data_packet['data'])
        if not df.empty and '代码' in df.columns:
            df['代码'] = df['代码'].astype(str)
        return df, data_packet.get('timestamp', '未知时间')
    except Exception:
        return pd.DataFrame(), "无云端存档"

# ================= 2. 数据接口 (增强版) =================

@st.cache_data(ttl=3600*4) 
def fetch_basic_info():
    try:
        df_sector = ak.stock_board_industry_name_em()
        sector_map = dict(zip(df_sector['板块名称'], df_sector['涨跌幅']))
        return df_sector, sector_map
    except:
        return pd.DataFrame(), {}

def download_market_spot_data():
    """
    下载实时行情，包含重试机制，应对云端网络波动
    """
    max_retries = 3
    for i in range(max_retries):
        try:
            # 尝试下载
            df = ak.stock_zh_a_spot_em()
            
            if df is not None and not df.empty:
                # 数据清洗
                if '代码' in df.columns:
                    df['代码'] = df['代码'].astype(str)
                # 简单过滤，去掉无关数据
                return df
        except Exception as e:
            time.sleep(1) # 失败等待1秒
            continue # 重试
            
    return pd.DataFrame() # 此时返回空，由上层处理

@st.cache_data(ttl=600) 
def fetch_market_sentiment_cached():
    try:
        # 指数数据通常较小，容易获取
        df = ak.stock_zh_index_daily(symbol="sh000001")
        if df.empty: return "未知", 1.0
        last = df.iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        if last['close'] > ma20: return "📈 大盘多头 (安全)", 1.0
        else: return "🌧️ 大盘空头 (轻仓)", 0.8
    except:
        return "未知环境", 1.0

# ================= 3. 核心算法 (保持不变) =================
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

def calculate_kelly(score, win_loss_ratio=2.0):
    if score < 60: p = 0.4
    else: p = 0.5 + (score - 60) * 0.00625
    p = min(0.8, p); b = win_loss_ratio; q = 1 - p; f = (b * p - q) / b; f_safe = f * 0.5
    if f_safe <= 0: return 0.0
    return round(f_safe * 100, 1)

def get_individual_fund_flow(code):
    try:
        market = "sh" if code.startswith("6") else "sz"
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df.empty: return 0.0
        return float(df.tail(1).iloc[0]['主力净流入-净额']) / 100000000.0 
    except: return 0.0

def get_stock_industry(code):
    try:
        df = ak.stock_individual_info_em(symbol=code)
        val = df[df['item'] == '行业']['value'].values
        return val[0] if len(val) > 0 else "其他"
    except: return "其他"

def analyze_stock_core(code, name, spot_row, market_factor=1.0, sector_map=None, strict_mode=True):
    try:
        # 基础数据
        current_price = spot_row['最新价']
        current_pct = spot_row['涨跌幅']
        pe = spot_row['市盈率-动态']
        turnover = spot_row['换手率']
        
        # ⚠️ 网络请求耗时点
        df_day = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        if df_day.empty or len(df_day) < 60: return None
        
        close = df_day['收盘'].iloc[-1]
        ma20 = df_day['收盘'].rolling(20).mean().iloc[-1]
        ma60 = df_day['收盘'].rolling(60).mean().iloc[-1]
        vol_5 = df_day['成交量'].tail(5).mean()
        vol_20 = df_day['成交量'].tail(20).mean()
        
        if strict_mode:
            if close < ma20: return None 
            if vol_5 < 1.0 * vol_20: return None
        
        industry = get_stock_industry(code)
        sector_pct = sector_map.get(industry, 0.0) if sector_map else 0.0
        
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
        score = 60.0; reasons = []; is_broken = False
        
        # 评分逻辑
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
        
        advice_60m = "⚖️ 震荡"; df_60m_data = None; has_gold_cross = False
        if not df_60m.empty:
            df_60m['K'], df_60m['D'], _ = calculate_kdj(df_60m)
            last_60, prev_60 = df_60m.iloc[-1], df_60m.iloc[-2]
            if prev_60['K'] < prev_60['D'] and last_60['K'] > last_60['D']:
                score += 20; reasons.insert(0, "⚡60分金叉"); advice_60m="💎 起爆"; has_gold_cross = True
            elif last_60['K'] < last_60['D']: score -= 10; reasons.append("⏳60分死叉"); advice_60m="✋ 回调"
            df_60m_data = df_60m
            
        day0, day1 = df_day.iloc[-1], df_day.iloc[-2]
        ma20_vol_s = df_day['成交量'].rolling(20).mean()
        force_signal = None
        if day1['成交量'] > 2*ma20_vol_s.iloc[-2] and day1['涨跌幅']>4 and day0['收盘']>day1['开盘']: force_signal="🔥昨抢筹"
        elif day0['成交量'] > 2*ma20_vol_s.iloc[-1] and day0['涨跌幅']>4: force_signal="🔥今抢筹"
        
        if force_signal: score += 25; reasons.insert(0, force_signal); advice_60m = "🔥 点火"
        if is_high_risk: score -= 15; reasons.append("⚠️高位")
        if is_broken: score = min(score, 40); advice_60m="🛑 离场"
        
        final_score = max(0.0, min(100.00, score * market_factor))
        kelly_pct = calculate_kelly(final_score, win_loss_ratio=2.0)
        priority = final_score + (100 if has_gold_cross and not is_broken else 0) + (50 if alpha > 0 else 0) + (30 if individual_flow > 0.5 else 0)
        
        recent_day = df_day.tail(30).copy()
        recent_day['日期'] = pd.to_datetime(recent_day['日期']).dt.strftime('%Y-%m-%d')
        
        return {
            "代码": code, "名称": name, "行业": industry, "板块涨幅": sector_pct, "个股资金": individual_flow,
            "现价": current_price, "ATR止损": round(stop_loss_pct, 2), "综合评分": round(final_score, 2), "排序权重": round(priority, 2),
            "评分理由": " ".join(reasons), "微操建议": advice_60m, "60分数据": df_60m_data, "日线数据": recent_day, "主力信号": force_signal,
            "换手率": turnover, "涨跌幅": current_pct, "凯利仓位": kelly_pct
        }
    except Exception:
        return None

def analyze_stock_task(args): 
    return analyze_stock_core(args[0], args[1], args[2], args[3], args[4], strict_mode=True)

def diagnose_single_stock(code, market_factor, sector_map):
    try:
        if 'market_snapshot' in st.session_state and not st.session_state.market_snapshot.empty: 
            spot = st.session_state.market_snapshot
        else: 
            # 如果快照为空，尝试临时下载单个数据
            spot = ak.stock_zh_a_spot_em()
            if spot.empty: return None, "无法获取市场数据"
        
        if '代码' in spot.columns: spot['代码'] = spot['代码'].astype(str)
        row = spot[spot['代码'] == code]
        
        if row.empty: return None, "代码不存在或未在列表中"
        
        res = analyze_stock_core(code, row.iloc[0]['名称'], row.iloc[0], market_factor, sector_map, strict_mode=False)
        return res, None
    except Exception as e: return None, str(e)

# ================= 4. 绘图与 UI =================
def draw_mini_chart_compact(df):
    if df is None: return go.Figure()
    mini_data = df.tail(20)
    fig = go.Figure(go.Candlestick(x=mini_data['时间'], open=mini_data['开盘'], high=mini_data['最高'], low=mini_data['最低'], close=mini_data['收盘'], increasing_line_color='#ef5350', decreasing_line_color='#26a69a'))
    fig.update_layout(margin=dict(l=0,r=0,t=2,b=2), height=45, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def draw_detail_chart(df, name):
    if df is None: return go.Figure()
    df['MA5'] = df['收盘'].rolling(5).mean(); df['MA20'] = df['收盘'].rolling(20).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['日期'], open=df['开盘'], high=df['最高'], low=df['最低'], close=df['收盘'], name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
    colors = ['#ef5350' if r['收盘'] >= r['开盘'] else '#26a69a' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df['日期'], y=df['成交量'], marker_color=colors, name='成交量'), row=2, col=1)
    fig.update_layout(title=f"{name} 量价趋势", height=400, xaxis_rangeslider_visible=False, yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'), margin=dict(l=10, r=10, t=40, b=10))
    return fig

def render_sector_pills(df_sec):
    if df_sec.empty: return
    df_sec = df_sec.sort_values(by='涨跌幅', ascending=False)
    top5 = df_sec.head(6); bot5 = df_sec.tail(6).sort_values(by='涨跌幅', ascending=True)
    st.markdown("""<style>.sector-container { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; } .sector-badge { padding: 4px 10px; border-radius: 15px; font-size: 13px; font-weight: 600; white-space: nowrap; } .badge-up { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; } .badge-down { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }</style>""", unsafe_allow_html=True)
    html_up = '<div class="sector-container"><span style="align-self:center;font-weight:bold;color:#d32f2f">🚀 领涨:</span>' + ''.join([f'<span class="sector-badge badge-up">{r["板块名称"]} {r["涨跌幅"]:.2f}%</span>' for _, r in top5.iterrows()]) + '</div>'
    st.markdown(html_up, unsafe_allow_html=True)
    html_down = '<div class="sector-container"><span style="align-self:center;font-weight:bold;color:#388e3c">💚 领跌:</span>' + ''.join([f'<span class="sector-badge badge-down">{r["板块名称"]} {r["涨跌幅"]:.2f}%</span>' for _, r in bot5.iterrows()]) + '</div>'
    st.markdown(html_down, unsafe_allow_html=True)

def render_stock_list(df_subset, page_state_key):
    if df_subset.empty: st.info("暂无符合该分类的标的"); return
    items_per_page = 10; total_items = len(df_subset); total_pages = max(1, (total_items - 1) // items_per_page + 1)
    current_page = st.session_state[page_state_key]
    if current_page >= total_pages: current_page = total_pages - 1
    if current_page < 0: current_page = 0
    st.session_state[page_state_key] = current_page
    start_idx = current_page * items_per_page; end_idx = min(start_idx + items_per_page, total_items)
    page_data = df_subset.iloc[start_idx:end_idx]
    st.caption(f"第 {current_page+1}/{total_pages} 页 | 共 {total_items} 只")
    for idx, row in page_data.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 2.5, 2, 1])
            with c1: st.markdown(f"**{row['名称']}**"); st.caption(f"{row['代码']}"); sec_color = "red" if row['板块涨幅'] > 0 else "green"; st.markdown(f"<span style='font-size:12px;color:gray'>{row['行业']} <span style='color:{sec_color}'>{row['板块涨幅']:+.1f}%</span></span>", unsafe_allow_html=True)
            with c2: pct_color = "red" if row['涨跌幅'] > 0 else "green"; st.markdown(f"<span style='font-size:16px;font-weight:bold;color:{pct_color}'>{row['涨跌幅']:+.2f}%</span>", unsafe_allow_html=True); flow_color = "#c53030" if row['个股资金'] > 0 else "#2f855a"; st.markdown(f"<span style='font-size:12px;color:{flow_color};font-weight:bold'>主力 {row['个股资金']:+.2f}亿</span>", unsafe_allow_html=True)
            with c3:
                kelly_val = row['凯利仓位']; kelly_color = "#9c27b0" if kelly_val > 20 else ("#1976d2" if kelly_val > 10 else "#607d8b")
                st.markdown(f"<span style='background:#f3e5f5;color:{kelly_color};padding:2px 5px;border-radius:4px;font-weight:bold;font-size:12px'>🎲 凯利: {kelly_val}%</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:13px'>建议: <span style='color:red;font-weight:bold'>{row['微操建议']}</span></span>", unsafe_allow_html=True)
            with c4:
                if row['60分数据'] is not None: st.plotly_chart(draw_mini_chart_compact(row['60分数据']), use_container_width=True, key=f"mini_{row['代码']}_{page_state_key}")
            with c5:
                if row['代码'] not in st.session_state.watchlist:
                    if st.button("➕", key=f"add_{row['代码']}_{page_state_key}"):
                        st.session_state.watchlist[row['代码']] = {'name': row['名称'], 'cost': row['现价'], 'buy_time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'highest': row['现价']}; save_userdata(); st.rerun()
                else: st.button("✔", disabled=True, key=f"done_{row['代码']}_{page_state_key}")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1: 
        if st.button("⬅️", key=f"prev_{page_state_key}", disabled=(current_page == 0)): st.session_state[page_state_key] -= 1; st.rerun()
    with c3: 
        if st.button("➡️", key=f"next_{page_state_key}", disabled=(end_idx >= total_items)): st.session_state[page_state_key] += 1; st.rerun()

# ================= 5. 初始化与主流程 =================

with st.spinner("☁️ 正在同步账户数据..."):
    user_data = load_userdata()

if 'watchlist' not in st.session_state: st.session_state.watchlist = user_data.get("watchlist", {})
if 'strategy_portfolio' not in st.session_state: st.session_state.strategy_portfolio = user_data.get("portfolio", {})

# 增强的初始化逻辑
if 'market_snapshot' not in st.session_state or st.session_state.market_snapshot.empty:
    st.session_state.market_snapshot = pd.DataFrame()
    st.session_state.last_update_str = "等待加载..."
    st.session_state.data_source = "未知"
    
    # 1. 优先尝试云端恢复 (最快且不被封)
    df_snap, snap_time = load_market_snapshot()
    
    if not df_snap.empty:
        st.session_state.market_snapshot = df_snap
        st.session_state.last_update_str = snap_time
        st.session_state.data_source = "☁️ 云端存档"
        st.toast(f"已恢复 {snap_time} 的行情数据")
    else:
        # 2. 如果云端没有，才尝试实时下载 (容易被墙)
        with st.spinner("🌐 云端无存档，正在尝试连接交易所..."):
            df_live = download_market_spot_data()
            if not df_live.empty:
                st.session_state.market_snapshot = df_live
                st.session_state.last_update_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.data_source = "🌐 实时抓取"
            else:
                st.session_state.data_source = "⚠️ 连接失败"
                st.error("无法获取数据。请在本地运行并点击刷新以推送到云端。")

if 'scan_results' not in st.session_state: st.session_state.scan_results = None
if 'diagnosis_result' not in st.session_state: st.session_state.diagnosis_result = None
if 'page_idx_attack' not in st.session_state: st.session_state.page_idx_attack = 0
if 'page_idx_ambush' not in st.session_state: st.session_state.page_idx_ambush = 0

# --- 侧边栏 ---
with st.sidebar:
    st.header("💸 操盘手 V55")
    
    st.info("💡 提示: 如果云端抓不到数据，请在本地电脑运行此App，点击下方按钮，数据会自动同步到云端。")
    
    if st.button("🔄 刷新全市场 (并备份)", type="primary"):
        with st.spinner("📥 1. 下载全市场数据 (带重试)..."):
            df = download_market_spot_data()
        
        if not df.empty:
            with st.spinner("☁️ 2. 上传至云端数据库..."):
                saved_time = save_market_snapshot(df)
                st.session_state.market_snapshot = df
                st.session_state.last_update_str = saved_time
                st.session_state.data_source = "🔴 实时 (已备份)"
            st.success(f"已更新 {len(df)} 只标的"); time.sleep(0.5); st.rerun()
        else:
            st.error("刷新失败：无法连接到数据源。")

    source_color = "red" if "实时" in st.session_state.get('data_source', '') else "blue"
    st.markdown(f"**数据源:** <span style='color:{source_color}'>{st.session_state.get('data_source', '未加载')}</span>", unsafe_allow_html=True)
    st.caption(f"数据时间: {st.session_state.last_update_str}")
    
    if st.session_state.watchlist:
        st.markdown("### 👀 重点关注")
        df_cache = st.session_state.market_snapshot
        for code, info in st.session_state.watchlist.items():
            name = info['name']; cost = info.get('cost', 0); curr, pct = cost, 0.0
            if not df_cache.empty:
                row = df_cache[df_cache['代码'] == str(code)]
                if not row.empty: curr = float(row.iloc[0]['最新价']); pct = float(row.iloc[0]['涨跌幅'])
            signal_icon = "🔥" if pct > 5.0 else ("🚀" if pct > 3.0 else ("💚" if pct < -3.0 else ""))
            with st.container():
                c1, c2, c3 = st.columns([3.5, 2, 1])
                c1.markdown(f"**{name}** {signal_icon}", unsafe_allow_html=True)
                color = "red" if pct > 0 else "green"; c2.markdown(f"<span style='color:{color};font-weight:bold'>{pct:+.2f}%</span>", unsafe_allow_html=True)
                if c3.button("✕", key=f"del_{code}"): del st.session_state.watchlist[code]; save_userdata(); st.rerun()
            st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)

    page = st.radio("模式选择:", ["⚡ 战术扫描", "🤖 策略组合", "📊 深度诊疗", "📂 资产看板"])

# --- 主页面内容 ---
if page == "⚡ 战术扫描":
    col_env1, col_env2 = st.columns([1, 3])
    with col_env1:
        market_status, market_factor = fetch_market_sentiment_cached()
        bg_color = "#e8f5e9" if market_factor >= 1.0 else "#ffebee"; text_color = "#2e7d32" if market_factor >= 1.0 else "#c62828"
        st.markdown(f"""<div style="background:{bg_color};padding:10px;border-radius:8px;text-align:center;color:{text_color};font-weight:bold;margin-bottom:10px">{market_status}</div>""", unsafe_allow_html=True)
    with col_env2:
        df_sec, sector_map = fetch_basic_info(); render_sector_pills(df_sec)

    st.markdown("---")
    
    c_scan1, c_scan2, c_scan3 = st.columns([2, 2, 1])
    with c_scan1: st.info("策略：资金穿透 + 妖股基因 + **凯利风控**")
    with c_scan2: scan_depth = st.slider("🔍 扫描深度 (只看前多少名)", 20, 100, 30, help="数字越小，速度越快！")
    with c_scan3: 
        if st.button("🚀 扫描", type="primary"):
            st.session_state.page_idx_attack = 0; st.session_state.page_idx_ambush = 0
            with st.spinner(f"🚀 正在极速分析 Top {scan_depth} 龙头股..."):
                try:
                    if st.session_state.market_snapshot.empty: st.error("无基础数据，请先刷新全市场")
                    else:
                        df_spot = st.session_state.market_snapshot
                        mask = (~df_spot['名称'].str.contains("ST") & ~df_spot['代码'].str.startswith(("688", "8", "4", "9")))
                        mask = mask & (df_spot['涨跌幅'] > 0)
                        candidates = df_spot[mask].sort_values(by='换手率', ascending=False).head(scan_depth)
                        
                        tasks = [(r['代码'], r['名称'], r, market_factor, sector_map) for _, r in candidates.iterrows()]
                        results = []
                        # 降低线程数以减少被封风险
                        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                            futures = {executor.submit(analyze_stock_task, t): t for t in tasks}
                            for f in concurrent.futures.as_completed(futures):
                                res = f.result(); 
                                if res: results.append(res)
                        if results: st.session_state.scan_results = pd.DataFrame(results).sort_values(by='排序权重', ascending=False); st.success(f"⚡ 完成！命中 {len(results)} 标的")
                        else: st.warning("无标的")
                except Exception as e: st.error(f"Error: {e}")

    if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
        df_res = st.session_state.scan_results
        mask_attack = df_res['微操建议'].str.contains("起爆|点火|金叉")
        df_attack = df_res[mask_attack]; df_ambush = df_res[~mask_attack]
        tab1, tab2 = st.tabs([f"🔥 核心进攻 ({len(df_attack)})", f"🕵️ 潜伏埋伏 ({len(df_ambush)})"])
        with tab1: render_stock_list(df_attack, "page_idx_attack")
        with tab2: render_stock_list(df_ambush, "page_idx_ambush")

elif page == "🤖 策略组合":
    st.title("🤖 策略组合 (实盘模拟)")
    st.caption("数据已开启硬盘级永久保存。")
    c1, c2 = st.columns([3, 1])
    with c1: st.info("AI 自动精选 Top 3 龙头股，并持续跟踪。")
    if c2.button("⚡ AI一键建仓", type="primary"):
        if st.session_state.scan_results is None or st.session_state.scan_results.empty: st.error("请先扫描！")
        else:
            top3 = st.session_state.scan_results.head(3)
            st.session_state.strategy_portfolio = {}
            for _, row in top3.iterrows():
                st.session_state.strategy_portfolio[row['代码']] = {'name': row['名称'], 'cost': row['现价'], 'buy_time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'highest': row['现价'], 'kelly': row['凯利仓位']}
            save_userdata(); st.success("✅ 建仓完成并存档！"); st.rerun()
    portfolio = st.session_state.strategy_portfolio
    if not portfolio: st.warning("暂无持仓")
    else:
        df_cache = st.session_state.market_snapshot
        for code, data in portfolio.items():
            curr = data['cost']
            if not df_cache.empty:
                row = df_cache[df_cache['代码'] == str(code)]
                if not row.empty: curr = float(row.iloc[0]['最新价'])
            if curr > data.get('highest', 0): portfolio[code]['highest'] = curr; save_userdata() 
            pnl = (curr - data['cost']) / data['cost'] * 100
            high = data.get('highest', curr); dd = (curr - high) / high * 100 if high > 0 else 0
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                c1.markdown(f"**{data['name']}** ({code})"); c1.caption(f"📅 {data['buy_time']}")
                color = "red" if pnl > 0 else "green"; c2.markdown(f"收益: <span style='color:{color};font-size:18px;font-weight:bold'>{pnl:+.2f}%</span>", unsafe_allow_html=True)
                c3.markdown(f"回撤: {dd:.2f}% | 凯利: {data.get('kelly', 0)}%")
                if c4.button("平仓", key=f"sell_ai_{code}"): del st.session_state.strategy_portfolio[code]; save_userdata(); st.rerun()

elif page == "📊 深度诊疗":
    st.title("🏥 个股诊疗")
    market_status, market_factor = fetch_market_sentiment_cached(); _, sector_map = fetch_basic_info()
    c1, c2 = st.columns([3, 1]); code_in = c1.text_input("输入代码", placeholder="6位代码")
    if c2.button("诊断") and len(code_in)==6:
        with st.spinner("分析中..."):
            res, err = diagnose_single_stock(code_in, market_factor, sector_map)
            if res: st.session_state.diagnosis_result = res
            else: st.error(err)
    if st.session_state.diagnosis_result:
        res = st.session_state.diagnosis_result
        k1, k2, k3 = st.columns(3); k1.metric("综合评分", f"{res['综合评分']:.0f}"); k2.metric("建议仓位", f"{res['凯利仓位']}%"); k3.metric("资金", f"{res['个股资金']:+.2f}亿")
        st.info(res['评分理由']); st.plotly_chart(draw_detail_chart(res['日线数据'], res['名称']), use_container_width=True)
        if res['代码'] not in st.session_state.watchlist:
            if st.button(f"➕ 加入自选 ({res['名称']})", use_container_width=True):
                st.session_state.watchlist[res['代码']] = {'name': res['名称'], 'cost': res['现价'], 'buy_time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'highest': res['现价']}; save_userdata(); st.rerun()

elif page == "📂 资产看板":
    st.title("📂 实盘账户分析")
    all_holdings = []
    for code, info in st.session_state.watchlist.items(): info['type'] = '手动'; info['code'] = code; all_holdings.append(info)
    for code, info in st.session_state.strategy_portfolio.items(): info['type'] = 'AI'; info['code'] = code; all_holdings.append(info)
    if not all_holdings: st.info("暂无持仓记录")
    else:
        df_cache = st.session_state.market_snapshot
        for item in all_holdings:
            code = item['code']; curr = item.get('cost', 0)
            if not df_cache.empty:
                row = df_cache[df_cache['代码'] == str(code)]
                if not row.empty: curr = float(row.iloc[0]['最新价'])
            highest = item.get('highest', item['cost'])
            if curr > highest:
                highest = curr
                if item['type'] == '手动': st.session_state.watchlist[code]['highest'] = highest
                else: st.session_state.strategy_portfolio[code]['highest'] = highest
                save_userdata()
            pnl = (curr - item['cost']) / item['cost'] * 100; dd = (curr - highest) / highest * 100 if highest > 0 else 0
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.5, 1.5, 2, 1])
                tag_bg = "#e3f2fd" if item['type'] == 'AI' else "#fff3e0"; tag_color = "#1565c0" if item['type'] == 'AI' else "#e65100"
                c1.markdown(f"**{item['name']}** <span style='background:{tag_bg};color:{tag_color};padding:2px 6px;border-radius:4px;font-size:12px'>{item['type']}</span>", unsafe_allow_html=True)
                c1.caption(f"建仓: {item.get('buy_time', '--')}")
                pnl_color = "red" if pnl > 0 else "green"; c2.markdown(f"<span style='color:{pnl_color};font-size:18px;font-weight:bold'>{pnl:+.2f}%</span>", unsafe_allow_html=True)
                c2.caption(f"成本: {item['cost']} -> 现价: {curr}"); c3.metric("最大回撤", f"{dd:.2f}%")
                if c4.button("平仓/删", key=f"del_all_{code}_{item['type']}"):
                    if item['type'] == '手动': del st.session_state.watchlist[code]
                    else: del st.session_state.strategy_portfolio[code]
                    save_userdata(); st.rerun()
