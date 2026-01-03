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
import re

# 忽略第三方库警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 全局配置与常量定义
# ==========================================
class Config:
    """系统配置常量类"""
    # 路径配置
    BASE_DIR = ""
    OUTPUT_FOLDER = "股票分析_Top15"
    
    # 交易配置
    POOL_SIZE = 500          # 扫描池扩大，寻找更多机会
    TOP_N = 20               # 推荐 Top 20
    CAPITAL = 100000         # 资金规模
    RISK_RATIO = 0.015       # 降低单笔风险至 1.5% (更保守)
    
    # 时间周期
    HISTORY_DAYS = 300       # 获取更长时间的历史数据 (用于长期趋势)
    
    # 过滤阈值
    MAX_TURNOVER = 25.0      # 最大换手率 (超过视为过度投机)
    MIN_TURNOVER = 0.5       # 最小换手率 (低于视为流动性枯竭)
    MIN_MARKET_CAP = 40      # 最小市值 40亿 (剔除微盘股操纵风险)
    
    # 绘图样式
    PLT_STYLE = 'dark_background'
    
    # 评分阈值
    MIN_SCORE = 60           # 最小入选分数

# ==========================================
# 2. 工具类与路径管理
# ==========================================
class SystemUtils:
    """系统工具类：处理跨平台路径、目录创建等"""
    
    @staticmethod
    def setup_environment():
        """初始化环境，创建输出目录"""
        system = platform.system()
        home = os.environ.get('USERPROFILE') if system == "Windows" else os.environ.get('HOME')
        desktop = os.path.join(home, 'Desktop') if home else os.getcwd()
        
        if not os.path.exists(desktop):
            desktop = os.getcwd()
            
        output_path = os.path.join(desktop, Config.OUTPUT_FOLDER)
        
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
                print(f"[System] Output directory created: {output_path}")
            except Exception as e:
                print(f"[Error] Failed to create folder: {e}")
                output_path = desktop
                
        Config.BASE_DIR = output_path
        return output_path

    @staticmethod
    def clean_numeric(val):
        """清洗字符串中的数值，如 '10.5亿', '20%' -> (10.5, 20.0)"""
        if pd.isna(val) or val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # 移除百分号、亿、万等文字，保留数字和小数点
            v = re.sub(r'[^\d.]', '', val)
            return float(v) if v else 0.0
        return 0.0
    
    @staticmethod
    def safe_filename(name):
        """清理文件名中的非法字符"""
        return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)

# ==========================================
# 3. 核心算法引擎
# ==========================================
class TechnicalIndicators:
    """技术指标计算引擎"""
    
    @staticmethod
    def calculate_all(df):
        """
        计算全套技术指标
        包含：MA, MACD, BOLL, ATR, RSI, KDJ, OBV, MFI, 
              新增: CMF (Chaikin), VWAP, Williams %R, BIAS, MA5
        """
        df = df.copy()
        
        # --- 基础均线 (添加MA5) ---
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ma120'] = df['close'].rolling(120).mean() # 长期趋势
        
        # --- MACD ---
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = exp12 - exp26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2
        df['macd_slope'] = df['macd_hist'].diff()
        
        # --- BOLL & ATR ---
        df['boll_mid'] = df['close'].rolling(20).mean()
        df['boll_std'] = df['close'].rolling(20).std()
        df['boll_upper'] = df['boll_mid'] + (df['boll_std'] * 2)
        df['boll_lower'] = df['boll_mid'] - (df['boll_std'] * 2)
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        
        # --- KDJ & RSI ---
        low_list = df['low'].rolling(9, min_periods=9).min()
        high_list = df['high'].rolling(9, min_periods=9).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        df['kdj_k'] = rsv.ewm(com=2).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # --- VWAP (成交量加权平均价 - 近似算法) ---
        # 真正的 VWAP 需要分时数据，这里用日线数据模拟一个"中期成本线"
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # --- CMF (Chaikin Money Flow) ---
        # 判断资金压力，范围 -0.5 到 0.5
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfv.fillna(0.5) # 防止除零
        cmf_vol = mfv * df['volume']
        df['cmf'] = cmf_vol.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # --- Williams %R ---
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        wr = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        df['wr'] = wr.fillna(-50)  # 初始值设为-50
        
        # --- MFI & OBV ---
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        pos_flow = money_flow.copy()
        neg_flow = money_flow.copy()
        flow_change = typical_price.diff()
        
        for i in range(1, len(df)):
            if flow_change.iloc[i] > 0:
                neg_flow.iloc[i] = 0
            else:
                pos_flow.iloc[i] = 0
                
        pos_mf = pos_flow.rolling(14).sum()
        neg_mf = neg_flow.rolling(14).sum()
        ratio = np.where(neg_mf == 0, 100, pos_mf / neg_mf)
        df['mfi'] = 100 - (100 / (1 + ratio))
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # --- 新增: BIAS (乖离率) ---
        df['bias'] = (df['close'] - df['ma20']) / df['ma20'] * 100
        
        return df

class MarketStructureAnalyzer:
    """市场结构分析器：市值、换手率、机构行为"""
    
    @staticmethod
    def analyze_market_cap(cap_str):
        """市值分级"""
        cap_num = SystemUtils.clean_numeric(cap_str)
        if cap_num >= 1000: return "Mega Cap (超大盘)", 0.8
        elif cap_num >= 200: return "Large Cap (大盘)", 0.9
        elif cap_num >= 50: return "Mid Cap (中盘)", 1.0
        else: return "Small Cap (小盘)", 1.2 # 小盘股波动大，风险系数高

    @staticmethod
    def detect_manipulation(df_hist, turnover_rate):
        """
        机构操纵局面检测算法
        核心：量价背离 (Volume-Price Divergence)
        """
        if df_hist.empty or len(df_hist) < 20:
            return "Insufficient Data", "Ignore"
            
        curr = df_hist.iloc[-1]
        
        # 1. 换手率异常检测
        if turnover_rate > Config.MAX_TURNOVER:
            return "High Speculation (Hot Money)", "Risk"
        if turnover_rate < Config.MIN_TURNOVER:
            return "Low Liquidity (Dead)", "Ignore"
            
        # 2. 量价背离检测 (Distribution - 机构出货)
        try:
            recent_vol = df_hist['volume'].iloc[-3:]
            vol_avg_20 = df_hist['volume'].rolling(20).mean().iloc[-1]
            
            if vol_avg_20 <= 0:
                return "Volume Data Error", "Ignore"
                
            price_change_3d = (curr['close'] - df_hist['close'].iloc[-4]) / df_hist['close'].iloc[-4]
            vol_spike_ratio = recent_vol.mean() / vol_avg_20
            
            if vol_spike_ratio > 2.0 and price_change_3d < 0.02:
                return "Manipulation: Huge Vol No Price (Distribution)", "Danger"
        except:
            pass
            
        # 3. CMF (资金流) 背离
        try:
            if 'cmf' in curr and not pd.isna(curr['cmf']) and curr['cmf'] < -0.1:
                if curr['close'] > curr.get('ma20', 0):
                    return "Risk: Money Outflow vs Price Rise", "Warning"
        except:
            pass
             
        return "Normal Flow", "Safe"

class AIStrategicAnalyst:
    """AI 战略分析核心类：整合所有逻辑进行评分"""
    
    def __init__(self, code, name, spot_data):
        self.code = code
        self.name = name
        self.spot = spot_data
        self.hist_df = None
        self.result = None

    def fetch_history(self):
        """获取并清洗历史数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=420)).strftime('%Y%m%d')  # 延长到420天确保有足够数据
        try:
            df = ak.stock_zh_a_hist(symbol=self.code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty or len(df) < 60:  # 至少需要60天数据
                return False
                
            # 兼容不同版本的akshare列名
            column_mapping = {}
            for col in df.columns:
                if '开盘' in col or 'open' in col.lower():
                    column_mapping[col] = 'open'
                elif '收盘' in col or 'close' in col.lower():
                    column_mapping[col] = 'close'
                elif '最高' in col or 'high' in col.lower():
                    column_mapping[col] = 'high'
                elif '最低' in col or 'low' in col.lower():
                    column_mapping[col] = 'low'
                elif '成交量' in col or 'volume' in col.lower():
                    column_mapping[col] = 'volume'
                elif '日期' in col or 'date' in col.lower():
                    column_mapping[col] = 'date'
                    
            df.rename(columns=column_mapping, inplace=True)
            
            # 确保所有需要的列都存在
            required_cols = ['open', 'close', 'high', 'low', 'volume', 'date']
            if not all(col in df.columns for col in required_cols):
                return False
                
            df['date'] = pd.to_datetime(df['date'])
            self.hist_df = df
            return True
        except Exception as e:
            # print(f"Error fetching history {self.code}: {e}")
            return False

    def analyze(self, market_env):
        """执行完整分析流程"""
        if not self.fetch_history(): 
            return None
        
        # 1. 计算指标
        df = TechnicalIndicators.calculate_all(self.hist_df)
        
        if df.empty or len(df) < 20:
            return None
            
        curr = df.iloc[-1]
        
        # 2. 市场结构与基本面分析
        # 清理市值字符串
        market_cap_val = self.spot.get('总市值', self.spot.get('总市值(亿)', 0))
        cap_category, cap_risk_factor = MarketStructureAnalyzer.analyze_market_cap(market_cap_val)
        
        # 清理换手率
        turnover_val = self.spot.get('换手率', self.spot.get('换手率(%)', 0))
        turnover = SystemUtils.clean_numeric(turnover_val)
        
        # 3. 机构操纵检测
        manip_status, manip_level = MarketStructureAnalyzer.detect_manipulation(df, turnover)
        
        if manip_level == "Danger":
            return self._construct_result(0, "Risk: Institutional Manipulation", df, curr, 
                                          cap_category, turnover, manip_status)
        if manip_level == "Ignore":
            return None # 流动性太低，直接忽略

        # 4. 三重滤网系统 + 逻辑判断
        # Screen 1: 趋势 (MACD Slope > 0)
        macd_slope = curr.get('macd_slope', np.nan)
        if pd.isna(macd_slope) or macd_slope <= 0:
            return None
            
        # Screen 2: 回调 (RSI < 65)
        rsi_val = curr.get('rsi', np.nan)
        if pd.isna(rsi_val) or rsi_val > 65:
            return None
            
        # Screen 3: 动能 (Price Action)
        ma5_val = curr.get('ma5', 0)
        if curr['close'] < ma5_val: 
            return None
        
        # --- 深度评分算法 (总分 100+) ---
        score = 0.0  # 使用浮点数
        reasons = []
        
        # 基础分：通过三重滤网 (40分)
        score += 40
        
        # 趋势强度 (20分)
        ma20_val = curr.get('ma20', 0)
        ma60_val = curr.get('ma60', 0)
        ma120_val = curr.get('ma120', 0)
        
        if curr['close'] > ma20_val > ma60_val > ma120_val:
            score += 20
            reasons.append("Perfect Uptrend")
        elif curr['close'] > ma20_val > ma60_val:
            score += 10
            reasons.append("Mid-term Uptrend")
            
        # 资金流向 (20分) - 结合 CMF 和 OBV
        cmf_val = curr.get('cmf', np.nan)
        if not pd.isna(cmf_val) and cmf_val > 0.1:
            score += 10
            reasons.append("Strong CMF Inflow")
            
        try:
            obv_val = curr.get('obv', np.nan)
            if not pd.isna(obv_val):
                obv_ma20 = df['obv'].rolling(20).mean().iloc[-1]
                if obv_val > obv_ma20:
                    score += 10
                    reasons.append("OBV Rising")
        except:
            pass
            
        # 机构博弈面 (10分)
        if manip_status == "Normal Flow":
            score += 10
            reasons.append("Healthy Order Flow")
            
        # 估值与乖离 (10分) - VWAP
        vwap_val = curr.get('vwap', curr['close'])
        if not pd.isna(vwap_val) and vwap_val > 0:
            vwap_deviation = abs(curr['close'] - vwap_val) / vwap_val
            if vwap_deviation < 0.05: # 价格贴近成本线
                score += 10
                reasons.append("Near VWAP Support")
        
        # 风险调整：小盘股给予一定风险溢价扣分，大盘股加分
        score = score * cap_risk_factor 
        
        # 动态止损 (ATR + VWAP)
        atr_val = curr.get('atr', 1.0)
        vwap_val = curr.get('vwap', curr['close'])
        
        if pd.isna(vwap_val) or vwap_val <= 0:
            vwap_val = curr['ma20'] if not pd.isna(curr.get('ma20', np.nan)) else curr['close']
        
        stop_loss = min(curr['close'] - (atr_val * 1.5), vwap_val * 0.98) # 止损位取 ATR 和 VWAP 中的较低值
        take_profit = curr['close'] + (atr_val * 3.5)
        
        return self._construct_result(score, "，".join(reasons), df, curr,
                                      cap_category, turnover, manip_status, stop_loss, take_profit)

    def _construct_result(self, score, reason, df, curr, cap_type, turnover, manip, sl=0, tp=0):
        """构建标准化的返回结果字典"""
        return {
            'score': round(float(score), 1),
            'reason': reason,
            'df': df,
            'curr': curr,
            'cap_type': cap_type,
            'turnover': float(turnover),
            'manipulation': manip,
            'stop_loss': float(sl),
            'take_profit': float(tp)
        }

# ==========================================
# 4. 可视化系统
# ==========================================
class Visualizer:
    """高级可视化：展示机构博弈和技术结构"""
    
    @staticmethod
    def plot(stock_code, stock_name, data):
        df = data['df'].tail(Config.HISTORY_DAYS)
        curr = data['curr']
        score = data['score']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
        
        # --- 子图1：价格与结构 (包含 VWAP) ---
        ax1.plot(df['date'], df['close'], label='Price', color='white', linewidth=2)
        ax1.plot(df['date'], df['ma5'], label='MA5', color='orange', linewidth=1, alpha=0.7)
        
        # 处理VWAP的NaN值
        vwap_series = df['vwap'].fillna(df['ma20'])
        ax1.plot(df['date'], vwap_series, label='VWAP (Inst. Cost)', color='cyan', linestyle='--', alpha=0.7)
        
        ma60_series = df['ma60'].fillna(method='ffill')
        ax1.plot(df['date'], ma60_series, label='MA60 Trend', color='yellow', linestyle=':')
        
        current_date = df['date'].iloc[-1]
        ax1.scatter(current_date, curr['close'], color='lime', s=100, zorder=5, label='Entry')
        
        # 标注风险信息
        risk_text = f"Manipulation: {data['manipulation']}\nCap: {data['cap_type']}"
        ax1.text(df['date'].iloc[0], df['close'].max() * 0.95, risk_text, color='orange', 
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='orange'), fontsize=8)

        ax1.set_title(f"{stock_code} {stock_name} - Deep Structure (Score: {score})", color='white', fontsize=14)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # --- 子图2：CMF (Chaikin Money Flow) ---
        cmf_series = df['cmf'].fillna(0)
        colors = ['g' if x > 0 else 'r' for x in cmf_series]
        ax2.bar(df['date'], cmf_series, color=colors, alpha=0.6)
        ax2.axhline(0, color='white', linewidth=0.5)
        ax2.axhline(0.1, color='yellow', linestyle='--', alpha=0.5, label='Strong Inflow')
        ax2.set_title("Chaikin Money Flow (Institutional Pressure)", color='cyan')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # --- 子图3：MACD & Volatility ---
        macd_hist_series = df['macd_hist'].fillna(0)
        colors_macd = ['g' if x > 0 else 'r' for x in macd_hist_series]
        ax3.bar(df['date'], macd_hist_series, color=colors_macd, label='MACD', alpha=0.5)
        
        macd_dif_series = df['macd_dif'].fillna(0)
        ax3.plot(df['date'], macd_dif_series, label='DIF', color='cyan')
        ax3.set_title("Momentum (MACD)", color='white')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
        plt.tight_layout()
        
        # 清理股票名称中的非法字符
        safe_name = SystemUtils.safe_filename(stock_name)
        file_path = os.path.join(Config.BASE_DIR, f"{stock_code}_{safe_name}_Deep.png")
        plt.savefig(file_path, dpi=100, bbox_inches='tight')
        plt.close()
        return file_path

# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    start_time = time.time()
    print("\n" + "="*80)
    print("      A股深度AI量化分析系统 v8.0 (OOP / Institutional Analysis)")
    print("="*80)
    
    # 1. 初始化环境
    SystemUtils.setup_environment()
    print(f"Environment: {platform.system()} | Path: {Config.BASE_DIR}")
    
    # 2. 获取市场数据
    print("\n[Step 1] Scanning Market Data...")
    try:
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df.empty:
            print("[Error] No market data fetched!")
            return
            
        # 基础清洗：去除 ST、停牌、价格异常
        # 确保名称字段存在
        name_col = None
        for col in spot_df.columns:
            if '名称' in col or 'name' in col.lower():
                name_col = col
                break
        
        if name_col is None:
            raise ValueError("Cannot find stock name column in data")
            
        # 去除ST股票
        spot_df = spot_df[~spot_df[name_col].str.contains('ST|退|停', na=False)]
        
        # 价格过滤
        price_col = None
        for col in spot_df.columns:
            if '最新价' in col or 'price' in col.lower():
                price_col = col
                break
                
        if price_col is None:
            raise ValueError("Cannot find price column in data")
            
        spot_df = spot_df[(spot_df[price_col] > 3) & (spot_df[price_col] < 300)]
        
        # 按成交额排序并取前N
        amount_col = None
        for col in spot_df.columns:
            if '成交额' in col or 'amount' in col.lower():
                amount_col = col
                break
                
        if amount_col:
            spot_df = spot_df.sort_values(by=amount_col, ascending=False).head(Config.POOL_SIZE)
        else:
            spot_df = spot_df.head(Config.POOL_SIZE)
            
    except Exception as e:
        print(f"[Error] Data Fetch Error: {e}")
        return

    # 3. 循环深度分析
    print(f"[Step 2] Deep Analyzing {len(spot_df)} Stocks with Institutional Logic...")
    results = []
    success_count = 0
    error_count = 0
    
    # 动态获取列名
    code_col = next((col for col in spot_df.columns if '代码' in col or 'code' in col.lower()), None)
    name_col = next((col for col in spot_df.columns if '名称' in col or 'name' in col.lower()), None)
    price_col = next((col for col in spot_df.columns if '最新价' in col or 'price' in col.lower()), None)
    
    if not all([code_col, name_col, price_col]):
        print("[Error] Required columns not found in market data!")
        return
    
    # 进度条
    for idx, row in tqdm(spot_df.iterrows(), total=len(spot_df), desc="Analyzing"):
        code = row[code_col]
        name = row[name_col]
        
        try:
            analyst = AIStrategicAnalyst(code, name, row)
            res = analyst.analyze("Bullish")
            
            if res and res['score'] >= Config.MIN_SCORE:
                # 添加基础信息
                res['code'] = code
                res['name'] = name
                res['price'] = row[price_col]
                results.append(res)
                success_count += 1
                
        except Exception as e:
            error_count += 1
            # print(f"Error analyzing {code}: {e}")
            continue
        
        # 根据网络状况调整延迟
        if idx % 50 == 0:  # 每50只股票休息一次
            time.sleep(0.05)

    # 4. 排序与生成报告
    print(f"\n[Step 3] Processing Results...")
    print(f"Successfully analyzed: {success_count} | Errors: {error_count}")
    
    if not results:
        print("No stocks met the deep analysis criteria.")
        return
        
    # 按评分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    top_stocks = results[:Config.TOP_N]
    
    # 添加名次
    for i, stock in enumerate(top_stocks):
        stock['rank'] = i + 1
    
    # 生成图表和 Excel
    print(f"[Step 4] Generating Reports for Top {Config.TOP_N}...")
    export_data = []
    
    for stock in tqdm(top_stocks, desc="Generating Viz"):
        try:
            chart_file = Visualizer.plot(stock['code'], stock['name'], stock)
            
            export_data.append({
                '名次': stock['rank'],
                '代码': stock['code'],
                '名称': stock['name'],
                '现价': round(float(stock['price']), 2),
                'AI评分': stock['score'],
                '市值类型': stock['cap_type'],
                '换手率(%)': round(float(stock['turnover']), 2),
                '机构博弈状态': stock['manipulation'],
                '核心逻辑': stock['reason'],
                '止损价': round(float(stock['stop_loss']), 2),
                '目标价': round(float(stock['take_profit']), 2),
                '走势图': os.path.basename(chart_file)
            })
        except Exception as e:
            print(f"Error plotting {stock['name']}: {e}")
            continue

    # 保存 Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    excel_file = os.path.join(Config.BASE_DIR, f"AI_Analysis_v8_{timestamp}.xlsx")
    
    try:
        df_export = pd.DataFrame(export_data)
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Deep_Analysis')
            
            # 自动列宽
            ws = writer.sheets['Deep_Analysis']
            for idx, col in enumerate(ws.columns, 1):
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        if cell.value:
                            max_len = max(max_len, len(str(cell.value)))
                    except:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 2, 50)
    except Exception as e:
        print(f"[Error] Failed to save Excel: {e}")
        # 尝试保存为CSV作为备选
        csv_file = os.path.join(Config.BASE_DIR, f"AI_Analysis_v8_{timestamp}.csv")
        df_export.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV Report saved as alternative: {csv_file}")

    # 输出摘要
    print("\n" + "="*80)
    print("                  Analysis Complete")
    print("="*80)
    
    if export_data:
        display_df = pd.DataFrame(export_data)
        display_cols = ['名次', '代码', '名称', '现价', 'AI评分', '市值类型', '机构博弈状态', '换手率(%)']
        print(display_df[display_cols].to_string(index=False))
    
    print(f"\nExcel Report: {excel_file}")
    print(f"Total Time: {time.time() - start_time:.2f} sec")
    print(f"Avg Time per Stock: {(time.time() - start_time) / len(spot_df):.2f} sec")

if __name__ == "__main__":
    main()
