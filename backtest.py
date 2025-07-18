"""
バックテストモジュール - FVG & レジスタンス突破戦略の過去検証
"""
import warnings
warnings.simplefilter(action='error', category=FutureWarning)
import yfinance as yf
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from curl_cffi import requests
import time

class FVGBreakBacktest:
    """FVG突破戦略のバックテストクラス"""

    def __init__(self, ma_period=200, fvg_min_gap=0.5,
                 resistance_lookback=20, breakout_threshold=1.005,
                 stop_loss_rate=0.02, target_profit_rate=0.05,
                 ma_proximity_percent=0.05):
        self.ma_period = ma_period
        self.fvg_min_gap = fvg_min_gap
        self.resistance_lookback = resistance_lookback
        self.breakout_threshold = breakout_threshold
        self.stop_loss_rate = stop_loss_rate
        self.target_profit_rate = target_profit_rate
        self.ma_proximity_percent = ma_proximity_percent

    def detect_fvg(self, df: pd.DataFrame, index: int) -> Optional[Dict]:
        """FVG検出を改善"""
        if index < 2 or index >= len(df):
            return None
    
        # 3本のローソク足
        candle1 = df.iloc[index - 2]
        candle2 = df.iloc[index - 1]
        candle3 = df.iloc[index]

        # ブルッシュFVG（上昇）
        gap_up = candle3['Low'] - candle1['High']

        # ベアリッシュFVG（下降）も検出
        gap_down = candle1['Low'] - candle3['High']

        # より緩い条件で検出
        if gap_up > 0 and (gap_up / candle1['High']) * 100 >= self.fvg_min_gap:
            return {
                'type': 'bullish',
                'date': df.index[index],
                'gap_top': candle3['Low'],
                'gap_bottom': candle1['High'],
                'gap_size_percent': (gap_up / candle1['High']) * 100,
                'entry_price': candle3['Close']
            }
        elif gap_down > 0 and (gap_down / candle3['High']) * 100 >= self.fvg_min_gap:
            return {
                'type': 'bearish',
                'date': df.index[index],
                'gap_top': candle1['Low'],
                'gap_bottom': candle3['High'],
                'gap_size_percent': (gap_down / candle3['High']) * 100,
                'entry_price': candle3['Close']
            }

        return None

    def find_resistance_levels(self, df: pd.DataFrame, current_index: int) -> List[float]:
        """現在位置から遡ってレジスタンスレベルを検出"""
        start_index = max(0, current_index - self.resistance_lookback)
        df_lookback = df.iloc[start_index:current_index]

        if len(df_lookback) < 5:
            return []

        # 直近の高値を取得
        recent_high = df_lookback['High'].max()

        # ローカル高値も検出
        highs = []
        for i in range(2, len(df_lookback) - 2):
            if (df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i-1] and
                df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i-2] and
                df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i+1] and
                df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i+2]):
                highs.append(df_lookback['High'].iloc[i])

        # 重複を除いて返す
        all_highs = [recent_high] + highs
        unique_highs = []
        for high in sorted(all_highs, reverse=True):
            if not unique_highs or all(abs(high - h) / h > 0.01 for h in unique_highs):
                unique_highs.append(high)

        return unique_highs[:3]

    def run_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """バックテストを実行"""
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        # MA計算のために十分なデータを確保
        fetch_start_date = start_date_dt - timedelta(days=self.ma_period * 2)

        session = requests.Session(impersonate="safari15_5")
        retries = 3
        df_daily_full = pd.DataFrame() # 空のデータフレームで初期化
        for i in range(retries):
            try:
                ticker_obj = yf.Ticker(symbol, session=session)
                df_daily_full = ticker_obj.history(
                    start=fetch_start_date,
                    end=end_date_dt,
                    auto_adjust=False
                )
                if not df_daily_full.empty:
                    break
            except Exception as e:
                if i == retries - 1:
                    return {"error": f"yfinanceダウンロードエラー ({symbol}): {e}"}
                time.sleep(i + 1)

        # 必要なカラムが存在するかを厳密にチェック
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if df_daily_full.empty or not all(col in df_daily_full.columns for col in required_columns):
            return {"error": f"データ取得エラーまたはカラム不足: {symbol}"}

        df_daily_full.index = df_daily_full.index.tz_localize(None)

        # 週次データを作成
        df_weekly_full = df_daily_full.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # 移動平均計算
        df_daily_full['MA200'] = df_daily_full['Close'].rolling(window=self.ma_period).mean()
        df_weekly_full['SMA200'] = df_weekly_full['Close'].rolling(window=self.ma_period).mean()

        # 週次SMAを日次データにマージ (より堅牢な方法)
        if df_daily_full.index.tz is not None:
            df_daily_full.index = df_daily_full.index.tz_localize(None)
        if df_weekly_full.index.tz is not None:
            df_weekly_full.index = df_weekly_full.index.tz_localize(None)

        df_daily_full_reset = df_daily_full.reset_index()
        df_weekly_full_reset = df_weekly_full.reset_index()

        df_daily_full_reset['Week_Start'] = pd.to_datetime(df_daily_full_reset['Date']).dt.to_period('W-MON').apply(lambda r: r.start_time)
        df_weekly_full_reset['Week_Start'] = pd.to_datetime(df_weekly_full_reset['Date']).dt.to_period('W-MON').apply(lambda r: r.start_time)

        df_daily_full = pd.merge(df_daily_full_reset,
                                 df_weekly_full_reset[['Week_Start', 'SMA200']],
                                 on='Week_Start',
                                 how='left',
                                 suffixes=('', '_weekly'))

        df_daily_full = df_daily_full.set_index('Date')
        df_daily_full.rename(columns={'SMA200': 'Weekly_SMA200'}, inplace=True)
        df_daily_full['Weekly_SMA200'] = df_daily_full['Weekly_SMA200'].ffill()
        
        # 元の期間にデータをトリム
        df_daily = df_daily_full.loc[start_date_dt:end_date].copy()
        
        # トレード記録
        strategy1_trades = [] # FVG
        strategy2_trades = [] # FVG + Resistance
        active_s1_trade = None

        # デバッグ情報を記録
        debug_info = {
            'total_days': len(df_daily),
            'days_with_valid_ma': 0,
            'days_above_weekly_sma': 0,
            'days_near_daily_ma': 0,
            'fvg_detected_count': 0,
            'resistance_breaks_detected': 0
        }
        
        # バックテスト実行
        for i in range(1, len(df_daily)):
            current_date = df_daily.index[i]
            current_price = df_daily['Close'].iloc[i]
            daily_ma = df_daily['MA200'].iloc[i]
            weekly_sma = df_daily['Weekly_SMA200'].iloc[i]

            # --- ポジション管理 ---
            if active_s1_trade:
                # 戦略2の条件（レジスタンス突破）をチェック
                if not active_s1_trade.get('s2_triggered'):
                    resistance_levels = self.find_resistance_levels(df_daily, i)
                    for resistance in resistance_levels:
                        if current_price > resistance * self.breakout_threshold and df_daily['Close'].iloc[i-1] <= resistance:
                            debug_info['resistance_breaks_detected'] += 1
                            active_s1_trade['s2_triggered'] = True

                            s2_trade = active_s1_trade.copy()
                            s2_trade['entry_date_s2'] = current_date
                            s2_trade['entry_price_s2'] = current_price
                            s2_trade['resistance'] = resistance
                            strategy2_trades.append(s2_trade)
                            break

                # 決済条件のチェック
                exit_reason = None
                if current_price <= active_s1_trade['stop_loss']:
                    exit_reason = 'ストップロス'
                elif current_price >= active_s1_trade['target']:
                    exit_reason = '利確'

                if exit_reason:
                    active_s1_trade['exit_date'] = current_date
                    active_s1_trade['exit_price'] = current_price
                    active_s1_trade['return'] = (current_price - active_s1_trade['entry_price']) / active_s1_trade['entry_price']
                    active_s1_trade['exit_reason'] = exit_reason
                    strategy1_trades.append(active_s1_trade)
                    active_s1_trade = None

            # --- 新規エントリー条件 ---
            if not active_s1_trade:
                # 基本条件
                if pd.isna(daily_ma) or pd.isna(weekly_sma):
                    continue
                debug_info['days_with_valid_ma'] += 1

                if current_price <= weekly_sma:
                    continue
                debug_info['days_above_weekly_sma'] += 1

                ma_distance = abs(current_price - daily_ma) / daily_ma
                if ma_distance > self.ma_proximity_percent:
                    continue
                debug_info['days_near_daily_ma'] += 1

                # 戦略1のトリガー（FVG検出）
                fvg = self.detect_fvg(df_daily, i)
                if fvg and fvg['type'] == 'bullish': # ブルッシュFVGのみを対象
                    debug_info['fvg_detected_count'] += 1
                    active_s1_trade = {
                        'symbol': symbol,
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'fvg_info': fvg,
                        'stop_loss': fvg['gap_bottom'] * (1 - self.stop_loss_rate),
                        'target': current_price * (1 + self.target_profit_rate),
                        's2_triggered': False
                    }

        # 未決済ポジションの処理
        if active_s1_trade:
            active_s1_trade['exit_date'] = df_daily.index[-1]
            active_s1_trade['exit_price'] = df_daily['Close'].iloc[-1]
            active_s1_trade['return'] = (active_s1_trade['exit_price'] - active_s1_trade['entry_price']) / active_s1_trade['entry_price']
            active_s1_trade['exit_reason'] = '期間終了'
            strategy1_trades.append(active_s1_trade)
        
        # 結果集計
        return self.calculate_statistics(symbol, start_date, end_date, strategy1_trades, strategy2_trades, debug_info)

    def calculate_statistics(self, symbol: str, start_date: str, end_date: str,
                             strategy1_trades: List[Dict], strategy2_trades: List[Dict], debug_info: Dict) -> Dict:
        """トレード結果の統計を計算"""

        # 戦略1 (FVG) の統計
        s1_returns = [t['return'] for t in strategy1_trades if 'return' in t]
        s1_wins = [r for r in s1_returns if r > 0]
        
        # 戦略2 (FVG -> Resistance) の統計
        # 戦略2のトレードは、戦略1のトレード結果を継承する
        s2_final_trades = []
        for s2_trade in strategy2_trades:
            # 対応するs1トレードを見つける
            for s1_trade in strategy1_trades:
                if s1_trade['entry_date'] == s2_trade['entry_date']:
                    if 'return' in s1_trade:
                        final_trade = s2_trade.copy()
                        final_trade['return'] = s1_trade['return']
                        s2_final_trades.append(final_trade)
                    break
        
        s2_returns = [t['return'] for t in s2_final_trades]
        s2_wins = [r for r in s2_returns if r > 0]

        # 全体の統計（戦略1がベース）
        total_trades = len(strategy1_trades)
        all_returns = s1_returns

        return {
            'symbol': symbol,
            'period': f"{start_date} - {end_date}",
            'total_trades': total_trades,
            'avg_return': np.mean(all_returns) * 100 if all_returns else 0,
            'max_profit': max(all_returns) * 100 if all_returns else 0,
            'max_loss': min(all_returns) * 100 if all_returns else 0,

            's1_stats': {
                'count': len(strategy1_trades),
                'win_rate': len(s1_wins) / len(strategy1_trades) * 100 if strategy1_trades else 0,
                'avg_return': np.mean(s1_returns) * 100 if s1_returns else 0,
            },
            's2_stats': {
                'count': len(strategy2_trades),
                'conversion_rate': len(strategy2_trades) / len(strategy1_trades) * 100 if strategy1_trades else 0,
                'win_rate': len(s2_wins) / len(s2_final_trades) * 100 if s2_final_trades else 0,
                'avg_return': np.mean(s2_returns) * 100 if s2_returns else 0,
            },

            'strategy1_trades': strategy1_trades,
            'strategy2_trades': strategy2_trades,
            'debug_info': debug_info
        }

    def create_summary_report(self, result: Dict) -> str:
        """バックテスト結果のサマリーレポートを作成"""
        if result.get('error'):
            return f"エラー: {result['error']}"
    
        s1 = result['s1_stats']
        s2 = result['s2_stats']

        report = f"""
📊 FVGベース戦略 バックテスト結果 - {result['symbol']}
期間: {result['period']}
━━━━━━━━━━━━━━━━━━━━━━━━

📈 戦略1: FVG検出
• トレード数: {s1['count']}回
• 勝率: {s1['win_rate']:.1f}%
• 平均リターン: {s1['avg_return']:.2f}%

🚀 戦略2: FVG検出 → レジスタンス突破
• 転換率 (S1→S2): {s2['conversion_rate']:.1f}%
• トレード数: {s2['count']}回
• 勝率: {s2['win_rate']:.1f}%
• 平均リターン: {s2['avg_return']:.2f}% (S1エントリーからの最終リターン)

💰 全体パフォーマンス (戦略1ベース):
• 総トレード数: {result['total_trades']}回
• 平均リターン: {result['avg_return']:.2f}%
• 最大利益: {result['max_profit']:.2f}%
• 最大損失: {result['max_loss']:.2f}%

📋 最近のトレード例 (戦略1):
"""
        # 最新のトレード例を表示
        for trade in result['strategy1_trades'][-5:]:
            outcome = "✅" if trade.get('return', 0) > 0 else "❌"
            s2_marker = "🚀" if trade.get('s2_triggered') else ""
            report += f"\n• {outcome} {trade['entry_date'].strftime('%Y-%m-%d')} "
            report += f"→ {trade['exit_date'].strftime('%Y-%m-%d')}: "
            report += f"{trade.get('return', 0)*100:.1f}% {s2_marker}"

        return report

# 使用例
if __name__ == "__main__":
    # バックテスト実行
    backtester = FVGBreakBacktest()

    # 動的な日付設定
    end_date_dt = datetime.now()
    start_date_dt = end_date_dt - timedelta(days=10*365) # 10年前

    # NVIDIAのバックテスト
    result = backtester.run_backtest(
        symbol="NVDA",
        start_date=start_date_dt.strftime('%Y-%m-%d'),
        end_date=end_date_dt.strftime('%Y-%m-%d')
    )

    # レポート出力
    print(backtester.create_summary_report(result))
