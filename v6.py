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
from mplfinance.original_flavor import candlestick_ohlc # 专门画K线

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 系统路径与配置模块
# ==========================================
class Config:
    """系统配置常量"""
    POOL_SIZE = 200            # 深度扫描池
    TOP_N_STOCKS = 10           # 推荐 Top 10 (仅生成周K线，保证速度)
    ANALYSIS_DAYS = 180        # 深度回测周期 (6个月)
    CAPITAL = 100000           # 模拟本金
    RISK_RATIO = 0.02          # 单笔风险 2%
    STYLE = 'dark_background'  # 绘图风格
    
    # 算法阈值
    MAX_BIAS = 8.0             # 最大乖离率阈值
    MIN_LIQUIDITY = 50000000   # 最小成交额 (5千万)
    MANIPULATION_VOL_RATIO = 3.5 # 检测操纵的量比倍数

def get_output_path():
    """跨平台路径管理器"""
    system = platform.system()
    home = os.environ.get('USERPROFILE', '.') if system == 'Windows' else os.environ.get('HOME', '.')
    desktop = os.path.join(home, 'Desktop')
    if not os.path.exists(desktop): desktop = os.getcwd()
    
    folder = "股票分析_深度终极版"
    path = os.path.join(desktop, folder)
    try:
        if not os.path.exists(path): os.makedirs(path)
    except:
        path = desktop
    return path

OUTPUT_PATH = get_output_path()

# ==========================================
# 2. 数据获取与宏观分析模块
# ==========================================
class DataLoader:
    """数据加载器：负责获取大盘、个股日线、个股周线"""
    
    @staticmethod
    def get_index_trend():
        """
        宏观分析：获取上证指数趋势
        返回: (Status, Slope)
        """
        try:
            df = ak.index_zh_a_hist(symbol="000001", period="daily", start_date="20230101", end_date=datetime.now().strftime('%Y%m%d'))
            if df.empty: return "Neutral", 0
            
            ma20 = df['收盘'].rolling(20).mean()
            ma60 = df['收盘'].rolling(60).mean()
            
            # 计算斜率：最新MA20 - 10天前MA20
            slope = ma20.iloc[-1] - ma20.iloc[-10]
            
            if df['收盘'].iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
                return "Bullish", slope
            elif df['收盘'].iloc[-1] < ma20.iloc[-1]:
                return "Bearish", slope
            else:
                return "Neutral", slope
        except Exception as e:
            print(f"Index Error: {e}")
            return "Neutral", 0

    @staticmethod
    def get_stock_daily(code):
        """获取日K线数据（深度分析用）"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty: return None
            df.rename(columns={'开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '日期':'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except: return None

    @staticmethod
    def get_stock_weekly(code):
        """获取周K线数据（中长期趋势分析用）"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="weekly", start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty: return None
            df.rename(columns={'开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '日期':'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except: return None

# ==========================================
# 3. 深度算法引擎
# ==========================================
class AlgoEngine:
    """算法核心：包含指标计算、操纵检测、市场情绪计算"""
    
    @staticmethod
    def calculate_indicators(df):
        """计算全套技术指标"""
        df = df.copy()
        
        # 1. 均线系统
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 2. MACD & 趋势斜率
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = exp12 - exp26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2
        df['macd_slope'] = df['macd_hist'].diff()
        
        # 3. 市场情绪
        # 定义：股价在20日相对位置，0-100
        low_20 = df['low'].rolling(20).min()
        high_20 = df['high'].rolling(20).max()
        df['sentiment'] = (df['close'] - low_20) / (high_20 - low_20) * 100
        
        # 4. 乖离率 BIAS
        df['bias_12'] = (df['close'] - df['ma20']) / df['ma20'] * 100
        
        # 5. 真实波幅 ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        
        # 6. OBV (资金流向)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        
        return df

    @staticmethod
    def detect_manipulation(df, latest_row_spot):
        """
        机构操纵/异常行为检测
        逻辑：
        1. 巨量滞涨：成交量爆量，但价格几乎不动（主力出货）
        2. 无量空涨：价格大涨，但成交量极小（诱多/控盘庄股）
        3. 极端震荡：ATR异常大（妖股/庄股）
        """
        risk_score = 0
        warnings = []
        
        if df is None or len(df) < 20: return 0, ["Data Insuff"]
        
        curr = df.iloc[-1]
        vol_avg_20 = df['volume'].rolling(20).mean().iloc[-1]
        
        # 异常1：巨量滞涨 (成交量 > 3.5倍平均，但涨跌幅 < 1%)
        vol_ratio = curr['volume'] / vol_avg_20
        price_change_pct = (curr['close'] - curr['open']) / curr['open'] * 100
        
        if vol_ratio > Config.MANIPULATION_VOL_RATIO and abs(price_change_pct) < 1.5:
            risk_score += 50 # 极高风险
            warnings.append(f"CRITICAL: High Vol ({vol_ratio:.1f}x) No Move")
            
        # 异常2：无量空涨 (涨 > 5% 但成交量 < 0.8倍平均)
        if price_change_pct > 5.0 and vol_ratio < 0.8:
            risk_score += 30
            warnings.append("WARNING: High Price Low Vol (Fake Move?)")
            
        # 异常3：极端乖离
        bias = curr['bias_12'] if 'bias_12' in curr else 0
        if abs(bias) > Config.MAX_BIAS:
            risk_score += 20
            warnings.append(f"WARNING: Extreme Bias {bias:.2f}%")
            
        return risk_score, warnings

    @staticmethod
    def analyze_market_cap(row):
        """
        市值与估值分析
        返回：市值类型 (Small/Mid/Large), 估值等级
        """
        try:
            total_mv = float(row['总市值']) # 单位：万元
            pe = float(row['市盈率-动态'])
            
            # 市值分类 (单位转换: 万 -> 亿)
            mv_yi = total_mv / 10000
            if mv_yi < 100: cap_type = "Small Cap"
            elif mv_yi < 500: cap_type = "Mid Cap"
            else: cap_type = "Large Cap"
            
            # 估值逻辑 (A股特色，小盘股PE高正常，大盘股PE高风险)
            valuation = "Neutral"
            if mv_yi < 50 and pe < 30: valuation = "Undervalued"
            elif mv_yi > 500 and pe > 60: valuation = "Overvalued"
            
            return cap_type, valuation, mv_yi
        except:
            return "Unknown", "Unknown", 0

    @staticmethod
    def comprehensive_score(df, spot_row, market_status, index_slope):
        """
        综合评分模型 (100分制 + 宏观减分)
        """
        if df is None or len(df) < 60: return None
        
        df = AlgoEngine.calculate_indicators(df)
        curr = df.iloc[-1]
        
        # --- 风险前置检查 (一票否决) ---
        manip_score, manip_warns = AlgoEngine.detect_manipulation(df, spot_row)
        if manip_score >= 50: # 一票否决
            return {'score': 0, 'reason': "Manipulation Detected: " + "".join(manip_warns), 'risk': 'High'}
            
        # --- 1. 宏观环境修正 ---
        base_score = 100
        
        # 如果大盘下跌趋势，所有股票评分打折
        if market_status == "Bearish":
            base_score *= 0.9
        elif market_status == "Bullish":
            base_score *= 1.05
            
        score = 0
        reasons = []
        
        # --- 2. 技术趋势得分 (40分) ---
        # 多头排列
        if curr['close'] > curr['ma20'] > curr['ma60']:
            score += 40
            reasons.append("Strong Uptrend")
        elif curr['close'] > curr['ma20']:
            score += 20
        else:
            score -= 30 # 趋势向下重扣
            
        # --- 3. 动能得分 (20分) ---
        # MACD 斜率向上
        if not pd.isna(curr['macd_slope']) and curr['macd_slope'] > 0:
            score += 20
            reasons.append("MACD Accel")
            
        # --- 4. 市场情绪得分 (20分) ---
        sentiment = curr['sentiment']
        # 40-70 为健康区间，过低超卖，过高超买
        if 40 < sentiment < 70:
            score += 20
            reasons.append("Healthy Sentiment")
        elif sentiment > 85:
            score -= 10 # 贪婪阶段，风险大
            
        # --- 5. 资金流向 (20分) ---
        if curr['obv'] > curr['obv_ma']:
            score += 20
            reasons.append("Smart Money Inflow")
        else:
            score -= 10
            
        # --- 6. 市值/估值加分 (扣分项) ---
        cap_type, valuation, mv = AlgoEngine.analyze_market_cap(spot_row)
        if valuation == "Undervalued":
            score += 10
            reasons.append("Valuation Low")
        elif valuation == "Overvalued":
            score -= 10 # 高估值扣分
            
        # --- 7. 惩罚项 ---
        score -= manip_score # 减去操纵风险分
        
        if score < 0: score = 0
        
        # --- 动态止损计算 (基于ATR) ---
        atr = curr['atr']
        stop_loss = curr['close'] - (atr * 2.0)
        take_profit = curr['close'] + (atr * 3.0)
        
        # 计算仓位
        risk_price = curr['close'] - stop_loss
        shares = int((Config.CAPITAL * Config.RISK_RATIO) / risk_price)
        shares = (shares // 100) * 100
        if shares < 100: shares = 100
        
        return {
            'score': round(score, 1),
            'reasons': reasons,
            'warnings': manip_warns,
            'risk_score': manip_score,
            'cap_type': cap_type,
            'valuation': valuation,
            'market_value': mv,
            'sentiment': round(sentiment, 1),
            'df_analysis': df,
            'current_data': curr,
            'shares': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

# ==========================================
# 4. 可视化模块 (包含周K线)
# ==========================================
class Visualizer:
    """可视化引擎：生成日K图和周K图"""
    
    @staticmethod
    def plot_daily_chart(code, name, data_dict, folder):
        """生成详细的日K分析图"""
        df = data_dict['df_analysis'].tail(60)
        curr = data_dict['current_data']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1.5, 1]})
        
        # 价格
        ax1.plot(df['date'], df['close'], color='white', label='Price', linewidth=2)
        ax1.plot(df['date'], df['ma20'], label='MA20', color='yellow', linestyle='--')
        ax1.plot(df['date'], df['ma60'], label='MA60', color='cyan', linestyle=':')
        ax1.fill_between(df['date'], df['ma20'], df['ma60'], color='blue', alpha=0.1)
        
        # 标记当前点
        current_date = df['date'].iloc[-1]
        ax1.scatter(current_date, curr['close'], color='lime', s=100, zorder=5)
        ax1.text(current_date, curr['close'], f"  {curr['close']:.2f}", color='lime')
        
        # 操纵标记
        if data_dict['risk_score'] > 0:
            ax1.text(df['date'].iloc[0], df['high'].max(), "WARNING: Abnormal Flow Detected", color='red', fontsize=10)
            
        ax1.set_title(f"{code} {name} Daily Analysis (Score: {data_dict['score']})", color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MACD
        colors = ['g' if x > 0 else 'r' for x in df['macd_hist']]
        ax2.bar(df['date'], df['macd_hist'], color=colors, alpha=0.5)
        ax2.plot(df['date'], df['macd_dif'], color='cyan')
        ax2.set_title("MACD", color='yellow')
        ax2.grid(True, alpha=0.3)
        
        # Sentiment (市场情绪)
        ax3.plot(df['date'], df['sentiment'], color='magenta', label='Sentiment')
        ax3.axhline(80, color='red', linestyle='--', label='Greed')
        ax3.axhline(20, color='green', linestyle='--', label='Fear')
        ax3.fill_between(df['date'], 20, 80, color='gray', alpha=0.1)
        ax3.set_ylim(0, 100)
        ax3.set_title("Market Sentiment (Position in 20-day range)", color='yellow')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 格式化
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), visible=True)
            
        filename = f"{folder}/{code}_{name}_Daily.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename

    @staticmethod
    def plot_weekly_chart(code, name, folder):
        """
        生成独立的周K线图 (近2-3个月)
        用于分析中长期趋势
        """
        df = DataLoader.get_stock_weekly(code)
        if df is None or len(df) < 10: return None
        
        # 只取最近12周 (约3个月)
        df = df.tail(12)
        df['date_num'] = mdates.date2num(df['date'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制K线
        candlestick_ohlc(ax, zip(df['date_num'], df['open'], df['high'], df['low'], df['close']), 
                          width=2, colorup='g', colordown='r', alpha=0.8)
        
        # 添加趋势线 (简单线性回归)
        x = np.arange(len(df))
        y = df['close'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(df['date_num'], p(x), "y--", linewidth=2, label="Trend Line")
        
        # 格式化
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.set_title(f"{code} {name} - Weekly Trend (Medium Term)", color='white', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 背景色
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        
        filename = f"{folder}/{code}_{name}_Weekly.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename

# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    start_time = time.time()
    print("\n" + "="*70)
    print("  A股AI深度量化分析系统 v8.0 (Deep Algo / Weekly Chart / Macro)")
    print("="*70)
    
    print(f"Output Path: {OUTPUT_PATH}")
    
    # 1. 宏观环境扫描
    print(">>> Analyzing Macro Environment (Index Trend)...")
    market_status, index_slope = DataLoader.get_index_trend()
    print(f"Market Status: {market_status} | Index Slope: {index_slope:.2f}")
    
    # 2. 获取股票池
    print(">>> Scanning Market Liquidity (Target: High Quality)...")
    try:
        spot_df = ak.stock_zh_a_spot_em()
        # 深度清洗：剔除ST、停牌、低流动性
        spot_df = spot_df[~spot_df['名称'].str.contains('ST|退|停')]
        spot_df = spot_df[spot_df['成交额'] > Config.MIN_LIQUIDITY]
        spot_df = spot_df[(spot_df['最新价'] > 3) & (spot_df['最新价'] < 200)] 
        
        # 按成交额排序
        spot_df = spot_df.sort_values(by='成交额', ascending=False).head(Config.POOL_SIZE)
    except Exception as e:
        print(f"Data Fetch Failed: {e}")
        return
        
    final_results = []
    
    # 3. 深度循环分析
    print(f">>> Deep Analyzing {len(spot_df)} stocks with Full Algorithm Stack...")
    for idx, row in tqdm(spot_df.iterrows(), total=len(spot_df)):
        code = row['代码']
        name = row['名称']
        price = row['最新价']
        
        try:
            hist_df = DataLoader.get_stock_daily(code)
            res = AlgoEngine.comprehensive_score(hist_df, row, market_status, index_slope)
            
            # 筛选逻辑：分数 > 65 且 无高风险
            if res and res['score'] > 65 and res['risk_score'] < 50:
                final_results.append({
                    '代码': code,
                    '名称': name,
                    '现价': price,
                    '综合评分': res['score'],
                    '市值类型': res['cap_type'],
                    '市值(亿)': round(res['market_value'], 2),
                    '估值状态': res['valuation'],
                    '市场情绪': res['sentiment'],
                    '操纵风险': f"{res['risk_score']} ({'Low' if res['risk_score']==0 else 'Warning'})",
                    '核心逻辑': "，".join(res['reasons']),
                    '风险提示': "，".join(res['warnings']) if res['warnings'] else "None",
                    '买入建议(手)': int(res['shares']/100),
                    '止损价': round(res['stop_loss'], 2),
                    '目标价': round(res['take_profit'], 2),
                    '_raw': res
                })
        except Exception as e:
            # 内部静默处理，防止程序中断
            pass
        
        time.sleep(0.1) # 略微提速
        
    # 4. 排序与筛选
    if not final_results:
        print("\nNo stocks passed the Deep Analysis criteria.")
        return
        
    final_results.sort(key=lambda x: x['综合评分'], reverse=True)
    top_stocks = final_results[:Config.TOP_N_STOCKS]
    
    # 5. 生成图表 (日K + 周K)
    print("\n>>> Generating Visualization Charts (Daily + Weekly)...")
    for stock in tqdm(top_stocks, desc="Charting"):
        try:
            # 日K
            daily_file = Visualizer.plot_daily_chart(stock['代码'], stock['名称'], stock['_raw'], OUTPUT_PATH)
            stock['日K图'] = daily_file
            
            # 周K (仅对Top 10生成，节省时间)
            weekly_file = Visualizer.plot_weekly_chart(stock['代码'], stock['名称'], OUTPUT_PATH)
            stock['周K图'] = weekly_file if weekly_file else "N/A"
            
        except Exception as e:
            pass
        del stock['_raw'] # 释放内存

    # 6. 导出 Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    excel_name = f"{OUTPUT_PATH}/Deep_Analysis_Report_{timestamp}.xlsx"
    
    cols = ['代码', '名称', '现价', '综合评分', '市值类型', '市值(亿)', '估值状态', 
            '市场情绪', '操纵风险', '核心逻辑', '买入建议(手)', '止损价', '目标价', '周K图']
    df_export = pd.DataFrame(top_stocks)[cols]
    
    with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Deep_Analysis')
        ws = writer.sheets['Deep_Analysis']
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_len: max_len = len(str(cell.value))
                except: pass
            ws.column_dimensions[col_letter].width = min(max_len + 2, 50)

    # 7. 输出总结
    print("\n" + "="*70)
    print(f"  Analysis Complete. Reports & Charts Saved to: {OUTPUT_PATH}")
    print("="*70)
    print(df_export.to_string(index=False))
    print(f"\nExcel: {excel_name}")
    print(f"Time: {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    main()
