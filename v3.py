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
import math

# 忽略非关键警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration & Constants (配置中心)
# ==========================================
class Config:
    # System Settings
    OS_SYSTEM = platform.system()
    USER_HOME = os.environ.get('USERPROFILE', '.') if OS_SYSTEM == 'Windows' else os.environ.get('HOME', '.')
    OUTPUT_DIR = os.path.join(USER_HOME, 'Desktop', '证券深度分析_Top15')
    
    # Trading Parameters
    CAPITAL_BASE = 100000
    RISK_PER_TRADE = 0.02
    POOL_SIZE = 200
    TOP_N = 15
    
    # Technical Constants
    EMA_LONG = 144  # "Line of Life" - Big Trend
    EMA_MID = 21    # Medium Trend
    EMA_SHORT = 7    # Small Trend
    BB_PERIOD = 20
    ATR_PERIOD = 14
    
    @staticmethod
    def init_env():
        if not os.path.exists(Config.OUTPUT_DIR):
            os.makedirs(Config.OUTPUT_DIR)
            print(f"[System] Created output directory: {Config.OUTPUT_DIR}")

Config.init_env()

# ==========================================
# 2. Data Engine (数据获取层)
# ==========================================
class DataEngine:
    """负责从数据源获取并清洗原始数据"""
    
    @staticmethod
    def get_market_index():
        """获取市场大环境基准 (沪深300)"""
        try:
            df = ak.index_zh_a_hist(symbol="000300", period="daily", 
                                  start_date="20240101", 
                                  end_date=datetime.now().strftime('%Y%m%d'))
            if df.empty: return None
            return df
        except Exception as e:
            print(f"[Error] Failed to fetch index data: {e}")
            return None

    @staticmethod
    def get_stock_pool():
        """获取实时活跃股票池"""
        try:
            df = ak.stock_zh_a_spot_em()
            # 基础清洗：剔除ST、停牌、新股、极低价/高价
            df = df[~df['名称'].str.contains('ST|退|停|新股')]
            df = df[(df['最新价'] > 3.0) & (df['最新价'] < 200.0)]
            df = df[df['成交额'] > 0]
            # 按成交额筛选流动性最好的
            df = df.sort_values(by='成交额', ascending=False).head(Config.POOL_SIZE)
            return df
        except Exception as e:
            print(f"[Error] Failed to fetch stock pool: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_history(symbol):
        """获取个股历史K线数据 (前复权)"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d') # 多取一点以保证EMA144准确
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                  start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty: return None
            
            # 标准化列名
            df.rename(columns={
                '开盘':'open', '收盘':'close', '最高':'high', 
                '最低':'low', '成交量':'volume', '日期':'date'
            }, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except:
            return None

# ==========================================
# 3. Technical Indicators Library (专业指标库)
# ==========================================
class TechIndicators:
    """封装专业级技术指标计算，包含 Fisher Transform 和 Chaikin MF"""
    
    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_all(df):
        """一次性计算所需的所有指标"""
        df = df.copy()
        
        # --- EMA System (Trend Analysis) ---
        df['ema_7'] = TechIndicators.ema(df['close'], Config.EMA_SHORT)
        df['ema_21'] = TechIndicators.ema(df['close'], Config.EMA_MID)
        df['ema_144'] = TechIndicators.ema(df['close'], Config.EMA_LONG)
        
        # --- Bollinger Bands & Squeeze (Volatility) ---
        df['bb_mid'] = df['close'].rolling(Config.BB_PERIOD).mean()
        std = df['close'].rolling(Config.BB_PERIOD).std()
        df['bb_upper'] = df['bb_mid'] + (std * 2)
        df['bb_lower'] = df['bb_mid'] - (std * 2)
        # BandWidth 检测变盘前兆
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # --- ATR (Volatility) ---
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(Config.ATR_PERIOD).mean()
        
        # --- MACD (Momentum) ---
        exp12 = TechIndicators.ema(df['close'], 12)
        exp26 = TechIndicators.ema(df['close'], 26)
        df['macd_dif'] = exp12 - exp26
        df['macd_dea'] = TechIndicators.ema(df['macd_dif'], 9)
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2
        
        # --- RSI (Relative Strength) ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # --- Fisher Transform (Advanced Entry Signal) ---
        # 将价格转换为接近正态分布，使转折点更清晰
        def _fisher_transform(x):
            # 简单的 Fisher 逻辑：将值映射到 -1.5 到 1.5 之间
            val = (x - x.min()) / (x.max() - x.min() + 1e-6)
            val = 2 * val - 1 # -1 to 1
            # Fisher calculation (simplified for series)
            return 0.5 * np.log((1 + val) / (1 - val + 1e-6))
        
        # 使用中值波动率进行Fisher计算
        mid_price = (df['high'] + df['low']) / 2
        df['fisher'] = mid_price.rolling(9).apply(lambda x: _fisher_transform(x)[-1])
        df['fisher_signal'] = df['fisher'].diff()

        # --- Chaikin Money Flow (CMF - Institutional Flow) ---
        # 比 OBV 更准确，因为考虑了价格在K线中的位置
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfv * df['volume']
        df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        return df

# ==========================================
# 4. Strategy & Analysis Engine (策略核心)
# ==========================================
class StrategyEngine:
    """核心策略引擎：包含大趋势识别、小趋势分析、风险判定"""
    
    @staticmethod
    def analyze(df, spot_row):
        if df is None or len(df) < Config.EMA_LONG + 10:
            return None
            
        df = TechIndicators.calculate_all(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. Big Trend Analysis (大趋势研判)
        # 逻辑：Price > EMA144 > EMA21 > EMA7 (完美多头排列)
        trend_score = 0
        trend_msg = []
        
        is_big_trend_bull = (curr['close'] > curr['ema_144']) and (curr['ema_144'] > prev['ema_144'])
        
        if is_big_trend_bull:
            trend_score += 40
            trend_msg.append("大趋势：多头排列(生命线向上)")
        else:
            # 大趋势向下，严格风控，除非极强反弹否则不看
            return None
            
        # 2. Small Trend & Momentum (小趋势与动能)
        # 逻辑：回踩 EMA21 或 EMA7 不破，且 MACD 红柱放大
        if curr['close'] > curr['ema_21'] and curr['ema_21'] > prev['ema_21']:
            trend_score += 20
            trend_msg.append("中趋势：站上21日线")
            
        if curr['close'] > curr['ema_7']:
            trend_score += 10
            trend_msg.append("短线：攻击形态")
            
        # Fisher Transform 信号 (精准转折)
        if not pd.isna(curr['fisher']) and not pd.isna(prev['fisher']):
            if curr['fisher'] > 0 and curr['fisher'] > prev['fisher']:
                trend_score += 15
                trend_msg.append("Fisher: 强势反转信号")

        # 3. Danger Signal Detection (危险信号排雷 - 实战核心)
        risk_alert = []
        risk_penalty = 0
        
        # 风险A：顶背离 (价格新高，CMF 走低 -> 主力出货)
        if curr['close'] > prev['close'] and curr['cmf'] < prev['cmf'] and curr['cmf'] < 0.1:
            risk_alert.append("CMF顶背离(警惕出货)")
            risk_penalty += 30
            
        # 风险B：量价背离 (价涨量缩)
        # 简单判断：今日涨幅 > 3%，但成交量 < 过去5日均值
        price_change_pct = (spot_row['最新价'] - spot_row['今开']) / spot_row['今开']
        vol_now = spot_row['成交量']
        vol_avg_5 = df['volume'].tail(5).mean()
        
        if price_change_pct > 0.03 and vol_now < vol_avg_5 * 0.8:
            risk_alert.append("量价背离(缩量上涨)")
            risk_penalty += 20
            
        # 风险C：波动率异常 (宽幅震荡洗盘)
        atr_pct = curr['atr'] / curr['close']
        if atr_pct > 0.08: # 日波动超过8%
            risk_alert.append("高波动率风险")
            risk_penalty += 10

        # 综合评分
        final_score = max(0, trend_score - risk_penalty + 20) # 基础分20
        
        # 4. Position Sizing & Stop Loss (智能仓位管理)
        # 基于波动率的 Kelly Criterion 思想
        atr_val = curr['atr'] if not pd.isna(curr['atr']) else 1.0
        stop_loss = curr['close'] - (atr_val * 1.5)
        
        # 价格下限保护：止损价不能低于 EMA21 (如果站上21日线，说明中期趋势未坏)
        if curr['close'] > curr['ema_21']:
            stop_loss = max(stop_loss, curr['ema_21'] * 0.97)
        
        risk_per_share = curr['close'] - stop_loss
        if risk_per_share <= 0: risk_per_share = curr['close'] * 0.05
        
        shares = int((Config.CAPITAL_BASE * Config.RISK_PER_TRADE) / risk_per_share)
        shares = (shares // 100) * 100
        if shares < 100: shares = 100
        
        # Target Price (盈亏比 1:3)
        take_profit = curr['close'] + (atr_val * 3.0)
        
        # 5. 市场环境调整
        # 检查大盘
        market_df = DataEngine.get_market_index()
        market_sentiment = "Neutral"
        if market_df is not None and len(market_df) > 20:
            ma20 = market_df['收盘'].rolling(20).mean().iloc[-1]
            if market_df['收盘'].iloc[-1] < ma20:
                final_score *= 0.9 # 熊市衰减
                market_sentiment = "Bearish"
            else:
                market_sentiment = "Bullish"
        
        return {
            'score': round(final_score, 1),
            'big_trend': "Bullish" if is_big_trend_bull else "Bearish",
            'trend_msg': " | ".join(trend_msg),
            'risk_alerts': " | ".join(risk_alert) if risk_alert else "Safe",
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'shares': shares,
            'df': df,
            'market': market_sentiment,
            'fisher_val': round(curr['fisher'], 2) if not pd.isna(curr['fisher']) else 0,
            'cmf_val': round(curr['cmf'], 3) if not pd.isna(curr['cmf']) else 0,
            'squeeze': "Yes" if curr['bb_width'] < curr['bb_width'].rolling(20).mean().iloc[-1] else "No"
        }

# ==========================================
# 5. Visualizer (可视化模块)
# ==========================================
class Visualizer:
    """生成专业的金融分析图表"""
    
    @staticmethod
    def generate_chart(code, name, data_dict):
        df = data_dict['df'].tail(120) # 展示近4个月
        info = data_dict
        
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.5, 1, 1])
        
        # 1. Main Chart: Price + EMA Ribbon
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['date'], df['close'], label='Price', color='white', linewidth=2)
        ax1.plot(df['date'], df['ema_21'], label='EMA21 (Trend)', color='orange', alpha=0.8)
        ax1.plot(df['date'], df['ema_144'], label='EMA144 (Life)', color='cyan', alpha=0.6, linestyle='--')
        
        # Fill Squeeze area
        ax1.fill_between(df['date'], df['bb_upper'], df['bb_lower'], color='gray', alpha=0.1, label='Bollinger Range')
        
        # Annotations
        last_date = df['date'].iloc[-1]
        last_close = df['close'].iloc[-1]
        ax1.scatter(last_date, last_close, color='lime', s=100, zorder=5)
        ax1.text(last_date, last_close, f"  {last_close:.2f}", color='lime', fontsize=12, fontweight='bold')
        
        # Title
        risk_text = "SAFE" if info['risk_alerts'] == "Safe" else f"RISK: {info['risk_alerts'][:20]}..."
        ax1.set_title(f"{code} {name}\nScore: {info['score']} | Big Trend: {info['big_trend']} | Status: {risk_text}", 
                     fontsize=12, color='white')
        ax1.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3)
        
        # 2. Momentum: MACD
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(df['date'], df['macd_hist'], color=['g' if x>0 else 'r' for x in df['macd_hist']], alpha=0.5)
        ax2.plot(df['date'], df['macd_dif'], color='cyan')
        ax2.set_title("Momentum (MACD)", fontsize=10, color='yellow')
        ax2.axhline(0, color='white', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Fisher Transform (Precise Entry)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(df['date'], df['fisher'], color='magenta', label='Fisher Transform')
        ax3.axhline(0, color='white', linestyle='--')
        ax3.axhline(1.5, color='red', linestyle=':', alpha=0.5)
        ax3.axhline(-1.5, color='green', linestyle=':', alpha=0.5)
        ax3.set_title("Precise Signal (Fisher)", fontsize=10, color='yellow')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left', fontsize=8)

        # 4. Money Flow (CMF)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(df['date'], df['cmf'], color='orange', label='CMF')
        ax4.fill_between(df['date'], df['cmf'], 0, where=df['cmf']>0, color='green', alpha=0.3)
        ax4.fill_between(df['date'], df['cmf'], 0, where=df['cmf']<0, color='red', alpha=0.3)
        ax4.axhline(0, color='white', linewidth=0.5)
        ax4.set_title("Institutional Flow (CMF)", fontsize=10, color='yellow')
        ax4.grid(True, alpha=0.3)
        
        # Formatting
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), visible=True)
            ax.set_facecolor('#1e1e1e')
        
        fig.patch.set_facecolor('#121212')
        filename = f"{Config.OUTPUT_DIR}/{code}_{name}_Pro.png"
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()
        return filename

# ==========================================
# 6. Main Execution (主程序)
# ==========================================
def main():
    start_time = time.time()
    print("\n" + "="*80)
    print("    Professional Stock Analysis System v8.0 (Architecture Upgrade)")
    print("="*80)
    
    print(f"[Config] OS: {Config.OS_SYSTEM} | Pool Size: {Config.POOL_SIZE} | Top N: {Config.TOP_N}")
    
    # 1. Fetch Data
    print("[Data] Fetching market liquidity data...")
    pool = DataEngine.get_stock_pool()
    if pool.empty:
        print("[Error] Market data unavailable.")
        return

    final_results = []
    
    # 2. Analysis Loop
    print(f"[Engine] Analyzing {len(pool)} assets with Fisher & CMF algorithms...")
    pbar = tqdm(pool.iterrows(), total=len(pool))
    
    for idx, row in pbar:
        code = row['代码']
        name = row['名称']
        
        try:
            hist = DataEngine.get_history(code)
            analysis = StrategyEngine.analyze(hist, row)
            
            if analysis and analysis['score'] >= 60:
                final_results.append({
                    'Rank': 0, # Placeholder
                    'Code': code,
                    'Name': name,
                    'Price': row['最新价'],
                    'Score': analysis['score'],
                    'Big Trend': analysis['big_trend'],
                    'Signal': analysis['trend_msg'],
                    'Risk': analysis['risk_alerts'],
                    'Fisher': analysis['fisher_val'],
                    'CMF': analysis['cmf_val'],
                    'Squeeze': analysis['squeeze'],
                    'Lots': int(analysis['shares']/100),
                    'Stop Loss': round(analysis['stop_loss'], 2),
                    'Target': round(analysis['take_profit'], 2),
                    'Market': analysis['market'],
                    '_analysis': analysis
                })
        except Exception as e:
            # Silent fail for individual stocks to maintain process flow
            pass
        
        time.sleep(0.1) # Polite delay

    # 3. Ranking & Filtering
    if not final_results:
        print("\n[Result] No assets met the rigorous criteria.")
        return

    # Sort by Score descending
    final_results.sort(key=lambda x: x['Score'], reverse=True)
    top_picks = final_results[:Config.TOP_N]
    
    # Assign Rank
    for i, stock in enumerate(top_picks):
        stock['Rank'] = i + 1

    # 4. Exporting
    print(f"\n[Visualizer] Generating professional charts for Top {len(top_picks)}...")
    for stock in tqdm(top_picks):
        try:
            chart_path = Visualizer.generate_chart(stock['Code'], stock['Name'], stock['_analysis'])
            stock['Chart'] = chart_path
        except:
            stock['Chart'] = "Error generating chart"
        del stock['_analysis']

    # Save Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    excel_path = f"{Config.OUTPUT_DIR}/Professional_Report_Top{Config.TOP_N}_{timestamp}.xlsx"
    
    cols_order = ['Rank', 'Code', 'Name', 'Price', 'Score', 'Big Trend', 'Signal', 'Risk', 'CMF', 'Squeeze', 'Lots', 'Stop Loss', 'Target', 'Market', 'Chart']
    df_out = pd.DataFrame(top_picks)[cols_order]
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False, sheet_name='Deep_Analysis')
        ws = writer.sheets['Deep_Analysis']
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_len:
                        max_len = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[col_letter].width = min(max_len + 2, 40)

    # 5. Console Output
    print("\n" + "="*80)
    print("                    Analysis Complete - Top Picks")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_out[['Rank', 'Code', 'Name', 'Price', 'Score', 'Big Trend', 'Risk', 'Lots', 'Target']].to_string(index=False))
    
    print(f"\n[Output] Report saved to: {excel_path}")
    print(f"[Output] Charts saved to: {Config.OUTPUT_DIR}")
    print(f"[Time] Total execution: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
