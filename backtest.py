"""
バックテストモジュール - FVG & レジスタンス突破戦略の過去検証
"""
import warnings
warnings.simplefilter(action='error', category=FutureWarning)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class FVGBreakBacktest:
    """FVG突破戦略のバックテストクラス"""

    def __init__(self, ma_period=200, fvg_min_gap=0.5,
                 resistance_lookback=20, breakout_threshold=1.005,
                 stop_loss_rate=0.02, target_profit_rate=0.05):
        self.ma_period = ma_period
        self.fvg_min_gap = fvg_min_gap
        self.resistance_lookback = resistance_lookback
        self.breakout_threshold = breakout_threshold
        self.stop_loss_rate = stop_loss_rate
        self.target_profit_rate = target_profit_rate

    def detect_fvg(self, df: pd.DataFrame, index: int) -> Optional[Dict]:
        """指定インデックスでFVGを検出"""
        if index < 2 or index >= len(df):
            return None
    
        # 3本のローソク足を取得
        candle1 = df.iloc[index - 2]
        candle2 = df.iloc[index - 1]
        candle3 = df.iloc[index]

        # FVG条件: 3本目の安値 > 1本目の高値
        gap_size = (candle3['Low'] - candle1['High']) / candle1['High'] * 100

        if gap_size >= self.fvg_min_gap:
            return {
                'date': df.index[index],
                'gap_top': candle3['Low'],
                'gap_bottom': candle1['High'],
                'gap_size_percent': gap_size,
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
        # データ取得
        stock = yf.Ticker(symbol)
        df_daily = stock.history(start=start_date, end=end_date)
        df_weekly = stock.history(start=start_date, end=end_date, interval="1wk")
        
        if df_daily.empty or df_weekly.empty:
            return {"error": "データ取得失敗"}
        
        # 移動平均計算
        df_daily['MA200'] = df_daily['Close'].rolling(window=self.ma_period).mean()
        df_weekly['SMA200'] = df_weekly['Close'].rolling(window=self.ma_period).mean()
        
        # 週次SMAを日次データに結合
        df_daily['Weekly_SMA200'] = np.nan
        for current_date_idx in df_daily.index: # Renamed variable to avoid conflict
            week_start = current_date_idx - timedelta(days=current_date_idx.weekday())
            if week_start in df_weekly.index:
                df_daily.loc[current_date_idx, 'Weekly_SMA200'] = df_weekly.loc[week_start, 'SMA200']
        df_daily['Weekly_SMA200'] = df_daily['Weekly_SMA200'].ffill()
        
        # トレード記録
        fvg_trades = []  # FVGエントリー
        resistance_trades = []  # レジスタンス突破エントリー
        active_fvg = None
        active_resistance = None
        
        # バックテスト実行
        for i in range(self.ma_period + 3, len(df_daily)):
            current_date = df_daily.index[i]
            current_price = df_daily['Close'].iloc[i]
            daily_ma = df_daily['MA200'].iloc[i]
            weekly_sma = df_daily['Weekly_SMA200'].iloc[i]

            # ポジション管理
            if active_fvg:
                # ストップロスチェック
                if current_price <= active_fvg['stop_loss']:
                    active_fvg['exit_date'] = current_date
                    active_fvg['exit_price'] = current_price
                    active_fvg['return'] = (current_price - active_fvg['entry_price']) / active_fvg['entry_price']
                    active_fvg['exit_reason'] = 'ストップロス'
                    fvg_trades.append(active_fvg)
                    active_fvg = None
                # 利確チェック
                elif current_price >= active_fvg['target']:
                    active_fvg['exit_date'] = current_date
                    active_fvg['exit_price'] = current_price
                    active_fvg['return'] = (current_price - active_fvg['entry_price']) / active_fvg['entry_price']
                    active_fvg['exit_reason'] = '利確'
                    fvg_trades.append(active_fvg)
                    active_fvg = None

            if active_resistance:
                # ストップロスチェック
                if current_price <= active_resistance['stop_loss']:
                    active_resistance['exit_date'] = current_date
                    active_resistance['exit_price'] = current_price
                    active_resistance['return'] = (current_price - active_resistance['entry_price']) / active_resistance['entry_price']
                    active_resistance['exit_reason'] = 'ストップロス'
                    resistance_trades.append(active_resistance)
                    active_resistance = None
                # 利確チェック
                elif current_price >= active_resistance['target']:
                    active_resistance['exit_date'] = current_date
                    active_resistance['exit_price'] = current_price
                    active_resistance['return'] = (current_price - active_resistance['entry_price']) / active_resistance['entry_price']
                    active_resistance['exit_reason'] = '利確'
                    resistance_trades.append(active_resistance)
                    active_resistance = None

            # 基本条件チェック
            if pd.isna(daily_ma) or pd.isna(weekly_sma):
                continue

            # 条件1: 週足SMA200以上
            if current_price <= weekly_sma:
                continue

            # 条件2: 日足MA200付近（±5%）
            ma_distance = abs(current_price - daily_ma) / daily_ma
            if ma_distance > 0.05:
                continue

            # FVG検出
            if not active_fvg:
                fvg = self.detect_fvg(df_daily, i)
                if fvg:
                    # FVGエントリー
                    active_fvg = {
                        'symbol': symbol,
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'fvg_info': fvg,
                        'stop_loss': fvg['gap_bottom'] * (1 - self.stop_loss_rate),
                        'target': current_price * (1 + self.target_profit_rate),
                        'daily_ma': daily_ma,
                        'weekly_sma': weekly_sma
                    }

            # レジスタンス突破チェック
            if not active_resistance:
                resistance_levels = self.find_resistance_levels(df_daily, i)
                for resistance in resistance_levels:
                    if current_price > resistance * self.breakout_threshold:
                        # 前日がレジスタンス以下かチェック
                        if df_daily['Close'].iloc[i-1] <= resistance:
                            # レジスタンス突破エントリー
                            active_resistance = {
                                'symbol': symbol,
                                'entry_date': current_date,
                                'entry_price': current_price,
                                'resistance': resistance,
                                'stop_loss': resistance * (1 - self.stop_loss_rate),
                                'target': current_price * (1 + self.target_profit_rate),
                                'daily_ma': daily_ma,
                                'weekly_sma': weekly_sma,
                                'had_fvg': active_fvg is not None
                            }
                            break
        
        # 未決済ポジションの処理
        if active_fvg:
            active_fvg['exit_date'] = df_daily.index[-1]
            active_fvg['exit_price'] = df_daily['Close'].iloc[-1]
            active_fvg['return'] = (active_fvg['exit_price'] - active_fvg['entry_price']) / active_fvg['entry_price']
            active_fvg['exit_reason'] = '期間終了'
            fvg_trades.append(active_fvg)
        
        if active_resistance:
            active_resistance['exit_date'] = df_daily.index[-1]
            active_resistance['exit_price'] = df_daily['Close'].iloc[-1]
            active_resistance['return'] = (active_resistance['exit_price'] - active_resistance['entry_price']) / active_resistance['entry_price']
            active_resistance['exit_reason'] = '期間終了'
            resistance_trades.append(active_resistance)
        
        # 結果集計
        return self.calculate_statistics(symbol, start_date, end_date, fvg_trades, resistance_trades)

    def calculate_statistics(self, symbol: str, start_date: str, end_date: str,
                             fvg_trades: List[Dict], resistance_trades: List[Dict]) -> Dict:
        """トレード結果の統計を計算"""
        # FVG統計
        fvg_returns = [t['return'] for t in fvg_trades] if fvg_trades else []
        fvg_wins = [r for r in fvg_returns if r > 0]
        
        # レジスタンス統計
        resistance_returns = [t['return'] for t in resistance_trades] if resistance_trades else []
        resistance_wins = [r for r in resistance_returns if r > 0]
        
        # FVG→レジスタンスの成功率を計算
        fvg_to_resistance = 0
        combined_wins = 0
        for fvg_trade in fvg_trades:
            # FVGエントリー後にレジスタンス突破があったかチェック
            for res_trade in resistance_trades:
                if (res_trade['entry_date'] > fvg_trade['entry_date'] and
                    res_trade['entry_date'] < fvg_trade['exit_date'] and
                    res_trade.get('had_fvg', False)):
                    fvg_to_resistance += 1
                    if fvg_trade['return'] > 0 and res_trade['return'] > 0:
                        combined_wins += 1
                    break
        
        # 全トレードの統計
        all_returns = fvg_returns + resistance_returns

        return {
            'symbol': symbol,
            'period': f"{start_date} - {end_date}",

            # FVG統計
            'fvg_count': len(fvg_trades),
            'fvg_win_rate': len(fvg_wins) / len(fvg_trades) * 100 if fvg_trades else 0,
            'fvg_avg_gain': np.mean(fvg_returns) * 100 if fvg_returns else 0,
            'fvg_to_resistance_rate': fvg_to_resistance / len(fvg_trades) * 100 if fvg_trades else 0,

            # レジスタンス統計
            'resistance_breaks': len(resistance_trades),
            'resistance_success_rate': len(resistance_wins) / len(resistance_trades) * 100 if resistance_trades else 0,
            'resistance_avg_return': np.mean(resistance_returns) * 100 if resistance_returns else 0,

            # 組み合わせ統計
            'combined_win_rate': combined_wins / fvg_to_resistance * 100 if fvg_to_resistance > 0 else 0,
            'total_trades': len(fvg_trades) + len(resistance_trades),
            'avg_return': np.mean(all_returns) * 100 if all_returns else 0,
            'max_profit': max(all_returns) * 100 if all_returns else 0,
            'max_loss': min(all_returns) * 100 if all_returns else 0,

            # 詳細データ
            'fvg_trades': fvg_trades,
            'resistance_trades': resistance_trades
        }

    def create_summary_report(self, result: Dict) -> str:
        """バックテスト結果のサマリーレポートを作成"""
        if result.get('error'):
            return f"エラー: {result['error']}"
    
        report = f"""
📊 FVGブレイク戦略 バックテスト結果 - {result['symbol']}
期間: {result['period']}
━━━━━━━━━━━━━━━━━━━━━━━━

🔵 FVGアラート統計:
• 検出数: {result['fvg_count']}回
• 勝率: {result['fvg_win_rate']:.1f}%
• 平均リターン: {result['fvg_avg_gain']:.2f}%
• レジスタンス到達率: {result['fvg_to_resistance_rate']:.1f}%

🟢 レジスタンス突破統計:
• 突破数: {result['resistance_breaks']}回
• 成功率: {result['resistance_success_rate']:.1f}%
• 平均リターン: {result['resistance_avg_return']:.2f}%

💰 2段階戦略パフォーマンス:
• FVG→突破の勝率: {result['combined_win_rate']:.1f}%
• 総トレード数: {result['total_trades']}回
• 平均リターン: {result['avg_return']:.2f}%
• 最大利益: {result['max_profit']:.2f}%
• 最大損失: {result['max_loss']:.2f}%

📋 最近のトレード例:
"""
        # 最新のトレード例を表示
        all_trades = []
        for trade in result['fvg_trades'][-3:]:
            all_trades.append({
                'type': 'FVG',
                'entry_date': trade['entry_date'],
                'exit_date': trade['exit_date'],
                'return': trade['return']
            })

        for trade in result['resistance_trades'][-3:]:
            all_trades.append({
                'type': 'Resistance',
                'entry_date': trade['entry_date'],
                'exit_date': trade['exit_date'],
                'return': trade['return']
            })

        # 日付順にソート
        all_trades.sort(key=lambda x: x['entry_date'], reverse=True)

        for trade in all_trades[:5]:
            report += f"\n• [{trade['type']}] {trade['entry_date'].strftime('%Y-%m-%d')} "
            report += f"→ {trade['exit_date'].strftime('%Y-%m-%d')}: "
            report += f"{trade['return']*100:.1f}%"

        return report

# 使用例
if __name__ == "__main__":
    # バックテスト実行
    backtester = FVGBreakBacktest()

    # NVIDIAの1年間バックテスト
    result = backtester.run_backtest(
        symbol="NVDA",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )

    # レポート出力
    print(backtester.create_summary_report(result))
