import akshare as ak
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import platform

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 智能路径配置 (Win/Linux 兼容)
# ==========================================
def get_output_dir():
    """跨平台获取桌面路径并创建文件夹"""
    system = platform.system()
    if system == "Windows":
        home = os.environ.get('USERPROFILE', '.')
    else: # Linux, Darwin (Mac)
        home = os.environ.get('HOME', '.')
        
    desktop = os.path.join(home, 'Desktop')
    
    # 兜底逻辑
    if not os.path.exists(desktop):
        desktop = os.getcwd()
    
    folder_name = "股票分析_TSTSS"
    output_path = os.path.join(desktop, folder_name)
    
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            print(f"Created folder: {output_path}")
        except Exception as e:
            print(f"Warning: Could not create folder. Saving to Desktop instead. Error: {e}")
            output_path = desktop
            
    return output_path

OUTPUT_DIR = get_output_dir()

# ==========================================
# 2. 配置参数
# ==========================================
POOL_SIZE = 150            # 扫描池大小
TOP_N_STOCKS = 10          # 推荐 Top 10
ANALYSIS_DAYS = 120        # 分析近 120 天
CAPITAL = 100000           # 模拟本金
RISK_RATIO = 0.02          # 单笔风险 2%

plt.style.use('dark_background')

# ==========================================
# 3. 核心数据获取
# ==========================================
def fetch_market_environment():
    """获取沪深300指数，判断市场环境"""
    try:
        df = ak.index_zh_a_hist(symbol="000300", period="daily", start_date="20240101", end_date=datetime.now().strftime('%Y%m%d'))
        if df.empty: return "Neutral"
        ma20 = df['收盘'].rolling(20).mean().iloc[-1]
        current = df['收盘'].iloc[-1]
        return "Bullish" if current > ma20 else "Bearish"
    except:
        return "Neutral"

def fetch_data(code):
    """获取历史数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=250)).strftime('%Y%m%d')
    
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty: return None
        df.rename(columns={'开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '日期':'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except: return None

# ==========================================
# 4. 算法引擎：三重滤网 + MFI修复 + 杀猪盘检测 (修复版)
# ==========================================
def calculate_mfi_safe(df, period=14):
    """安全的 MFI 计算"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.copy()
    negative_flow = money_flow.copy()
    
    flow_change = typical_price.diff()
    for i in range(1, len(df)):
        if flow_change[i] > 0:
            negative_flow[i] = 0
        else:
            positive_flow[i] = 0
            
    positive_mf = positive_flow.rolling(window=period, min_periods=period).sum()
    negative_mf = negative_flow.rolling(window=period, min_periods=period).sum()
    
    # 修复分母为零
    ratio = np.where(negative_mf == 0, 100, positive_mf / negative_mf)
    mfi = 100 - (100 / (1 + ratio))
    return mfi

def calculate_super_indicators(df):
    """计算全套指标，包含MACD斜率和 OBV MA"""
    df = df.copy()
    
    # 均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_dif'] = exp12 - exp26
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2
    
    # --- 新增：MACD 斜率 ---
    df['macd_slope'] = df['macd_hist'].diff()
    
    # BOLL
    df['boll_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['boll_upper'] = df['boll_mid'] + (std * 2)
    df['boll_lower'] = df['boll_mid'] - (std * 2)
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # KDJ
    low_list = df['low'].rolling(9, min_periods=9).min()
    high_list = df['high'].rolling(9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    # >>>> 修复点：添加 OBV 均线 <<<<
    df['obv_ma'] = df['obv'].rolling(20).mean()
    
    # MFI
    df['mfi'] = calculate_mfi_safe(df)
    
    return df

def apply_triple_screen(df):
    """三重滤网逻辑"""
    if len(df) < 60: return 'Neutral', 'Data Insufficient'
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Screen 1: MACD Slope
    slope = curr['macd_slope']
    if pd.isna(slope): return 'Neutral', 'MACD Slope NA'
    is_trend_up = slope > 0
    
    if not is_trend_up:
        return 'Neutral', 'Screen 1: Trend Down'
    
    # Screen 2: RSI Dip
    rsi_val = curr['rsi']
    if pd.isna(rsi_val): return 'Neutral', 'RSI NA'
    
    # 趋势向上，RSI在30-55之间回调
    is_pullback = (30 < rsi_val < 55)
    
    if not is_pullback:
        return 'Neutral', 'Screen 2: No Pullback'
    
    # Screen 3: Entry
    is_stop_fall = curr['close'] > prev['close']
    is_strong = curr['close'] > curr['ma5']
    
    if is_stop_fall and is_strong:
        return 'Bullish Setup', 'Triple Screen: Trend Up + RSI Dip + Breakout'
    elif is_stop_fall:
        return 'Bullish Setup', 'Triple Screen: Trend Up + RSI Dip (Waiting Breakout)'
    else:
        return 'Neutral', 'Screen 3: Waiting for Price Action'

def detect_advanced_scam(df):
    """杀猪盘检测逻辑"""
    if len(df) < 60: return False, "Safe"
    curr = df.iloc[-1]
    ma60 = curr['ma60']
    
    if pd.isna(ma60): return False, "Safe"
    
    is_high_price = curr['close'] > (ma60 * 1.2)
    if not is_high_price: return False, "Safe"
    
    recent_vol = df['volume'].iloc[-3:]
    vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
    
    if pd.isna(vol_ma5): return False, "Safe"
    is_sustained_huge_vol = (recent_vol > vol_ma5 * 2).all()
    
    body_low = min(curr['open'], curr['close'])
    upper_shadow = curr['high'] - body_low
    total_range = curr['high'] - curr['low']
    
    if total_range == 0: is_long_shadow = False
    else: is_long_shadow = (upper_shadow / total_range) > 0.5
    
    if is_sustained_huge_vol and is_long_shadow:
        return True, "Risk: High Pos + Huge Vol + Shadow"
        
    return False, "Safe"

def score_stock(df_hist, real_time_price, market_env):
    """综合评分逻辑 (修复版)"""
    if df_hist is None or len(df_hist) < 90: return None 
    
    df = calculate_super_indicators(df_hist)
    curr = df.iloc[-1]
    
    # 1. 杀猪盘检测
    is_scam, scam_reason = detect_advanced_scam(df)
    if is_scam:
        return {'score': 0, 'reason': scam_reason, 'setup': 'Scam'}
        
    # 2. MFI 极度超买
    mfi_val = curr['mfi'] if not pd.isna(curr['mfi']) else 50
    if mfi_val > 90:
        return {'score': 0, 'reason': 'MFI > 90 (Extreme Risk)', 'setup': 'Risk'}
        
    # 3. 三重滤网检测
    tsts_status, tsts_msg = apply_triple_screen(df)
    
    if tsts_status != 'Bullish Setup':
        return {'score': 0, 'reason': tsts_msg, 'setup': tsts_status}
        
    # --- 评分计算 ---
    score = 0
    reasons = [f"TSTSS: {tsts_msg}"]
    
    score += 60 # 通过三重滤网基础分
    
    # 趋势强度
    if not pd.isna(curr['ma20']) and not pd.isna(curr['ma60']):
        if curr['close'] > curr['ma20'] > curr['ma60']:
            score += 20
            reasons.append("Strong MA Alignment")
        
    # 动能
    if curr['macd_hist'] > 0 and curr['macd_dif'] > curr['macd_dea']:
        score += 10
        reasons.append("MACD Momentum")
        
    # 资金 (增加了 obv_ma 的 NaN 检查)
    if not pd.isna(curr['obv_ma']):
        if curr['obv'] > curr['obv_ma']:
            score += 10
            reasons.append("OBV Inflow")
    else:
        # 数据不足时不扣分，视为中性
        pass
    
    # --- 动态 ATR 止损 ---
    current_atr = curr['atr']
    if pd.isna(current_atr): current_atr = 1.0 # 防御
    
    atr_avg_30 = df['atr'].rolling(30).mean().iloc[-1]
    if pd.isna(atr_avg_30): atr_avg_30 = current_atr
    
    vol_ratio = current_atr / atr_avg_30 if atr_avg_30 > 0 else 1.0
    
    if vol_ratio < 0.8: atr_mult = 2.0
    elif vol_ratio > 1.5: atr_mult = 1.0
    else: atr_mult = 1.5
        
    stop_loss = curr['close'] - (current_atr * atr_mult)
    take_profit = curr['close'] + (current_atr * 3.0)
    
    price_floor = df['low'].iloc[-20:].min() * 0.95
    if stop_loss < price_floor: stop_loss = price_floor
        
    risk_per_share = curr['close'] - stop_loss
    if risk_per_share <= 0: risk_per_share = curr['close'] * 0.05 # 强制止损距离
        
    shares = int((CAPITAL * RISK_RATIO) / risk_per_share)
    shares = (shares // 100) * 100
    if shares < 100: shares = 100
    
    return {
        'score': score,
        'reasons': reasons,
        'setup': tsts_status,
        'df_analysis': df,
        'current_data': curr,
        'shares': shares,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }

# ==========================================
# 5. 可视化模块
# ==========================================
def generate_stock_chart(code, name, analysis_data):
    df = analysis_data['df_analysis'].tail(ANALYSIS_DAYS) 
    curr = analysis_data['current_data']
    score = analysis_data['score']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1.5, 1]})
    
    # Plot 1: Price
    ax1.plot(df['date'], df['close'], label='Close Price', color='white', linewidth=2)
    ax1.plot(df['date'], df['ma20'], label='MA20', color='yellow', linestyle='--')
    ax1.plot(df['date'], df['ma60'], label='MA60', color='cyan', linestyle=':')
    
    current_date = df['date'].iloc[-1]
    ax1.scatter(current_date, curr['close'], color='lime', s=100, zorder=5, label='Entry Zone')
    ax1.text(current_date, curr['close'], f"  {curr['close']:.2f}", color='lime', fontsize=12, fontweight='bold')
    
    ax1.set_title(f"{code} {name} - Triple Screen Analysis (Score: {score})", fontsize=14, color='white')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MACD
    ax2.bar(df['date'], df['macd_hist'], color=['g' if x > 0 else 'r' for x in df['macd_hist']], label='MACD Hist', alpha=0.5)
    ax2.plot(df['date'], df['macd_dif'], label='DIF', color='cyan')
    ax2.plot(df['date'], df['macd_dea'], label='DEA', color='orange')
    ax2.axhline(0, color='white', linewidth=0.5)
    ax2.set_title("Screen 1: Weekly Trend (MACD Slope)", fontsize=10, color='yellow')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RSI
    ax3.plot(df['date'], df['rsi'], label='RSI (14)', color='magenta', linewidth=2)
    ax3.axhline(30, color='green', linestyle='--', alpha=0.5) 
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5)   
    ax3.axhline(50, color='white', linestyle='-', alpha=0.3) 
    ax3.fill_between(df['date'], 30, 70, color='gray', alpha=0.1) 
    
    ax3.set_title("Screen 2: Oscillator (RSI Pullback)", fontsize=10, color='yellow')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.get_xticklabels(), visible=True)
    
    filename = f"{OUTPUT_DIR}/{code}_{name}_TSTSS.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename

# ==========================================
# 6. 主程序
# ==========================================
def main():
    start_time = time.time()
    print("\n" + "="*70)
    print("  A股AI量化分析系统 v6.1 (Triple Screen - Fixed)")
    print("="*70)
    
    print(f"OS Detected: {platform.system()}")
    print(f"Output Dir: {OUTPUT_DIR}")
    
    market_env = fetch_market_environment()
    print(f"Market Status: {market_env}")
    
    print("Scanning Market Liquidity...")
    try:
        spot_df = ak.stock_zh_a_spot_em()
        spot_df = spot_df[~spot_df['名称'].str.contains('ST|退|停')]
        spot_df = spot_df[(spot_df['最新价'] > 3) & (spot_df['最新价'] < 200)] 
        spot_df = spot_df.sort_values(by='成交额', ascending=False).head(POOL_SIZE)
    except Exception as e:
        print("Data fetch failed:", e)
        return
        
    final_results = []
    
    print(f"Deep Analyzing with Triple Screen Logic...")
    for idx, row in tqdm(spot_df.iterrows(), total=len(spot_df)):
        code = row['代码']
        name = row['名称']
        price = row['最新价']
        
        try:
            hist_df = fetch_data(code)
            res = score_stock(hist_df, price, market_env)
            
            if res and res['score'] >= 60:
                final_results.append({
                    '代码': code,
                    '名称': name,
                    '现价': price,
                    'AI评分': res['score'],
                    '三重滤网状态': res['setup'],
                    '核心逻辑': "，".join(res['reasons']),
                    'RSI(14)': round(res['current_data']['rsi'], 2),
                    'MACD斜率': round(res['current_data']['macd_slope'], 2),
                    'OBV资金': "流入" if not pd.isna(res['current_data']['obv_ma']) and res['current_data']['obv'] > res['current_data']['obv_ma'] else "流出",
                    '买入建议(手)': int(res['shares']/100),
                    '止损价': round(res['stop_loss'], 2),
                    '目标价': round(res['take_profit'], 2),
                    '_raw_data': res
                })
        except Exception as inner_e:
            # 忽略单只股票的错误，保证程序继续运行
            # print(f"Error processing {code}: {inner_e}")
            pass
        
        time.sleep(0.2)
        
    if not final_results:
        print("\nNo stocks passed the Triple Screen criteria.")
        return

    final_results.sort(key=lambda x: x['AI评分'], reverse=True)
    top_stocks = final_results[:TOP_N_STOCKS]
    
    print("\nGenerating Triple Screen Charts...")
    for stock in tqdm(top_stocks, desc="Charting"):
        try:
            chart_file = generate_stock_chart(stock['代码'], stock['名称'], stock['_raw_data'])
            stock['走势图文件'] = chart_file
        except:
            pass
        del stock['_raw_data'] 

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    excel_name = f"{OUTPUT_DIR}/TSTSS_Report_{timestamp}.xlsx"
    
    df_export = pd.DataFrame(top_stocks)
    
    with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='TSTSS_Analysis')
        worksheet = writer.sheets['TSTSS_Analysis']
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            worksheet.column_dimensions[column].width = min(max_length + 2, 50)

    print("\n" + "="*70)
    print(f"  Analysis Complete! Triple Screen System Active")
    print("="*70)
    
    display_cols = ['代码', '名称', '现价', 'AI评分', '三重滤网状态', 'RSI(14)', '买入建议(手)', '止损价']
    print(df_export[display_cols].to_string(index=False))
    
    print(f"\nExcel: {excel_name}")
    print(f"Folder: {OUTPUT_DIR}")
    print(f"Total Time: {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    main()
