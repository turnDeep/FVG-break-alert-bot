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
    """FVG突破戦略のバックテストクラス（改善版）"""

    def __init__(self, ma_period=200, fvg_min_gap=0.1,  # FVGの最小ギャップを0.5から0.1に緩和
                 resistance_lookback=50,  # レジスタンスルックバックを20から50に拡大
                 breakout_threshold=1.002,  # ブレイクアウト閾値を1.005から1.002に緩和
                 stop_loss_rate=0.02, target_profit_rate=0.05,
                 ma_proximity_percent=0.10):  # MA近接条件を0.05から0.10に緩和
        self.ma_period = ma_period
        self.fvg_min_gap = fvg_min_gap
        self.resistance_lookback = resistance_lookback
        self.breakout_threshold = breakout_threshold
        self.stop_loss_rate = stop_loss_rate
        self.target_profit_rate = target_profit_rate
        self.ma_proximity_percent = ma_proximity_percent

    def detect_fvg(self, df: pd.DataFrame, index: int) -> Optional[Dict]:
        """FVG検出を改善 - より小さなギャップも検出"""
        if index < 2 or index >= len(df):
            return None
    
        # 3本のローソク足
        candle1 = df.iloc[index - 2]
        candle2 = df.iloc[index - 1]
        candle3 = df.iloc[index]

        # ブルッシュFVG（上昇）
        # より緩い条件：candle2の高値がcandle1とcandle3の間にギャップを作る
        if candle3['Low'] > candle1['High']:  # 明確なギャップ
            gap_size = candle3['Low'] - candle1['High']
            gap_percent = (gap_size / candle1['High']) * 100
            
            if gap_percent >= self.fvg_min_gap:
                return {
                    'type': 'bullish',
                    'date': df.index[index],
                    'gap_top': candle3['Low'],
                    'gap_bottom': candle1['High'],
                    'gap_size_percent': gap_percent,
                    'entry_price': candle3['Close'],
                    'candle2_high': candle2['High'],  # 中間ローソクの高値も記録
                    'candle2_low': candle2['Low']
                }

        # ベアリッシュFVG（下降）
        elif candle1['Low'] > candle3['High']:  # 明確なギャップ
            gap_size = candle1['Low'] - candle3['High']
            gap_percent = (gap_size / candle3['High']) * 100
            
            if gap_percent >= self.fvg_min_gap:
                return {
                    'type': 'bearish',
                    'date': df.index[index],
                    'gap_top': candle1['Low'],
                    'gap_bottom': candle3['High'],
                    'gap_size_percent': gap_percent,
                    'entry_price': candle3['Close'],
                    'candle2_high': candle2['High'],
                    'candle2_low': candle2['Low']
                }

        return None

    def find_resistance_levels(self, df: pd.DataFrame, current_index: int) -> List[Dict]:
        """レジスタンスレベルの検出を改善 - 価格帯とボリュームを考慮"""
        start_index = max(0, current_index - self.resistance_lookback)
        df_lookback = df.iloc[start_index:current_index]

        if len(df_lookback) < 10:
            return []

        resistance_levels = []
        
        # 1. 直近の明確な高値
        recent_high = df_lookback['High'].max()
        recent_high_idx = df_lookback['High'].idxmax()
        resistance_levels.append({
            'level': recent_high,
            'type': 'recent_high',
            'date': recent_high_idx,
            'strength': 3  # 強度スコア
        })

        # 2. ローカル高値（スイングハイ）
        for i in range(2, len(df_lookback) - 2):
            if (df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i-1] and
                df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i-2] and
                df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i+1] and
                df_lookback['High'].iloc[i] > df_lookback['High'].iloc[i+2]):
                
                level = df_lookback['High'].iloc[i]
                # 既存のレベルと近すぎない場合のみ追加
                if all(abs(level - r['level']) / r['level'] > 0.005 for r in resistance_levels):
                    resistance_levels.append({
                        'level': level,
                        'type': 'swing_high',
                        'date': df_lookback.index[i],
                        'strength': 2
                    })

        # 3. ボリュームクラスター（高ボリュームエリア）
        if 'Volume' in df_lookback.columns:
            volume_mean = df_lookback['Volume'].mean()
            high_volume_days = df_lookback[df_lookback['Volume'] > volume_mean * 1.5]
            
            for idx, row in high_volume_days.iterrows():
                level = row['High']
                if all(abs(level - r['level']) / r['level'] > 0.005 for r in resistance_levels):
                    resistance_levels.append({
                        'level': level,
                        'type': 'volume_cluster',
                        'date': idx,
                        'strength': 1
                    })

        # 4. 価格の集中帯（コンソリデーションエリア）
        price_counts = pd.cut(df_lookback['High'], bins=20).value_counts()
        if len(price_counts) > 0:
            most_common_range = price_counts.idxmax()
            if pd.notna(most_common_range):
                consolidation_level = most_common_range.mid
                if all(abs(consolidation_level - r['level']) / r['level'] > 0.005 for r in resistance_levels):
                    resistance_levels.append({
                        'level': consolidation_level,
                        'type': 'consolidation',
                        'date': df_lookback.index[-1],
                        'strength': 1
                    })

        # 強度でソートして返す
        resistance_levels.sort(key=lambda x: (-x['strength'], -x['level']))
        return resistance_levels[:5]  # 上位5つのレジスタンスレベルを返す

    def check_fvg_retest_entry(self, df: pd.DataFrame, index: int, fvg: Dict) -> bool:
        """FVGへのリテスト（戻り）を確認してエントリー"""
        if fvg['type'] != 'bullish':
            return False
            
        current_price = df['Close'].iloc[index]
        current_low = df['Low'].iloc[index]
        
        # FVGゾーンにタッチまたは侵入
        if current_low <= fvg['gap_top'] and current_price > fvg['gap_bottom']:
            # 価格がFVGゾーンから上に抜けようとしている
            if index > 0 and df['Close'].iloc[index] > df['Close'].iloc[index-1]:
                return True
        
        return False

    def run_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """バックテストを実行"""
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        # MA計算のために十分なデータを確保（日足）
        fetch_start_date_daily = start_date_dt - timedelta(days=self.ma_period * 2)
        # 週足MAのためにさらに過去のデータを取得
        fetch_start_date_weekly = start_date_dt - timedelta(days=200 * 7 * 2) # 週足200MA

        session = requests.Session(impersonate="safari15_5")
        retries = 3
        df_daily_full = pd.DataFrame()
        df_weekly_full = pd.DataFrame()

        try:
            ticker_obj = yf.Ticker(symbol, session=session)
            # 日足データ取得
            for i in range(retries):
                try:
                    df_daily_full = ticker_obj.history(
                        start=fetch_start_date_daily, end=end_date_dt, interval="1d", auto_adjust=False
                    )
                    if not df_daily_full.empty: break
                except Exception:
                    time.sleep(i + 1)

            # 週足データ取得
            for i in range(retries):
                try:
                    df_weekly_full = ticker_obj.history(
                        start=fetch_start_date_weekly, end=end_date_dt, interval="1wk", auto_adjust=False
                    )
                    if not df_weekly_full.empty: break
                except Exception:
                    time.sleep(i + 1)

        except Exception as e:
            return {"error": f"yfinanceダウンロードエラー ({symbol}): {e}"}

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if df_daily_full.empty or not all(col in required_columns for col in df_daily_full.columns):
            return {"error": f"日足データ取得エラーまたはカラム不足: {symbol}"}
        if df_weekly_full.empty or not all(col in required_columns for col in df_weekly_full.columns):
            return {"error": f"週足データ取得エラーまたはカラム不足: {symbol}"}
        
        df_daily_full.index = df_daily_full.index.tz_localize(None)
        df_weekly_full.index = df_weekly_full.index.tz_localize(None)

        # --- 移動平均の計算 ---
        df_daily_full['MA200_daily'] = df_daily_full['Close'].rolling(window=self.ma_period).mean()
        df_weekly_full['MA200_weekly'] = df_weekly_full['Close'].rolling(window=200).mean()

        # 日足データに週足MAをマージ
        df_weekly_ma = df_weekly_full[['MA200_weekly']].copy()
        df_daily_full = pd.merge_asof(df_daily_full.sort_index(), df_weekly_ma.sort_index(),
                                      left_index=True, right_index=True, direction='backward')

        # 元の期間にデータをトリム
        df_daily = df_daily_full.loc[start_date_dt:end_date].copy()
        
        # --- トレードロジックの初期化 ---
        strategy1_trades = []
        strategy2_trades = []
        active_trade = None # 統合されたトレード管理
        detected_fvgs = []

        # デバッグ情報
        debug_info = {
            'total_days': len(df_daily),
            'days_with_valid_ma': 0,
            'days_above_weekly_ma': 0,
            'days_in_ma_zone': 0,
            'fvg_detected_count': 0,
            'fvg_retest_count': 0,
            'resistance_breaks_detected': 0,
            'strategy1_entries': 0,
            'strategy2_entries': 0,
        }

        # --- バックテスト実行 ---
        for i in range(3, len(df_daily)):
            current_date = df_daily.index[i]
            current_price = df_daily['Close'].iloc[i]
            current_high = df_daily['High'].iloc[i]
            daily_ma = df_daily['MA200_daily'].iloc[i]
            weekly_ma = df_daily['MA200_weekly'].iloc[i]

            if pd.isna(daily_ma) or pd.isna(weekly_ma):
                continue
            debug_info['days_with_valid_ma'] += 1

            # --- ポジション管理 ---
            if active_trade:
                # 戦略2への移行チェック
                if active_trade['status'] == 'strategy1':
                    resistance_levels = self.find_resistance_levels(df_daily, i)
                    for res in resistance_levels:
                        # 終値がレジスタンスを明確にブレイク
                        if current_price > res['level'] * self.breakout_threshold:
                            debug_info['resistance_breaks_detected'] += 1
                            active_trade['status'] = 'strategy2'
                            active_trade['entry_date_s2'] = current_date
                            active_trade['entry_price_s2'] = current_price
                            active_trade['resistance_broken'] = res
                            strategy2_trades.append(active_trade.copy()) # 戦略2トレードとして記録
                            debug_info['strategy2_entries'] += 1
                            break # 最初のブレイクで移行

                # 決済条件
                exit_reason = None
                if current_price <= active_trade['stop_loss']: exit_reason = 'ストップロス'
                elif current_price >= active_trade['target']: exit_reason = '利確'

                if exit_reason:
                    active_trade['exit_date'] = current_date
                    active_trade['exit_price'] = current_price
                    active_trade['return'] = (current_price - active_trade['entry_price']) / active_trade['entry_price']
                    active_trade['exit_reason'] = exit_reason
                    strategy1_trades.append(active_trade) # 最終的なトレードとして記録
                    active_trade = None

            # --- 新規エントリー条件 ---
            if not active_trade:
                # 1. 基本条件: 週足200MA以上
                if current_price <= weekly_ma:
                    continue
                debug_info['days_above_weekly_ma'] += 1

                # 2. セットアップ: 日足200MAでの攻防
                is_in_ma_zone = abs(current_price - daily_ma) / daily_ma < self.ma_proximity_percent
                if is_in_ma_zone:
                    debug_info['days_in_ma_zone'] += 1

                # FVG検出 (日足200MAより上で発生)
                if current_price > daily_ma:
                    fvg = self.detect_fvg(df_daily, i)
                    if fvg and fvg['type'] == 'bullish':
                        debug_info['fvg_detected_count'] += 1
                        detected_fvgs.append({'fvg': fvg, 'index': i, 'tested': False})

                # 3. 戦略1: FVGサポートでのエントリー
                for fvg_data in detected_fvgs:
                    if not fvg_data['tested'] and i > fvg_data['index'] + 2: # 検出から2日後以降
                        # FVGへのリテスト（タッチ＆反発）を確認
                        if self.check_fvg_retest_entry(df_daily, i, fvg_data['fvg']):
                            # 前日の終値より今日の終値が高い（反発の確認）
                            if i > 0 and df_daily['Close'].iloc[i] > df_daily['Close'].iloc[i-1]:
                                debug_info['fvg_retest_count'] += 1
                                fvg_data['tested'] = True

                                # エントリー
                                active_trade = {
                                    'symbol': symbol,
                                    'status': 'strategy1',
                                    'entry_date': current_date,
                                    'entry_price': current_price,
                                    'fvg_info': fvg_data['fvg'],
                                    'stop_loss': fvg_data['fvg']['gap_bottom'] * (1 - self.stop_loss_rate),
                                    'target': current_price * (1 + self.target_profit_rate),
                                }
                                debug_info['strategy1_entries'] += 1
                                break
                if active_trade: continue

        # 未決済ポジションの処理
        if active_trade:
            active_trade['exit_date'] = df_daily.index[-1]
            active_trade['exit_price'] = df_daily['Close'].iloc[-1]
            active_trade['return'] = (active_trade['exit_price'] - active_trade['entry_price']) / active_trade['entry_price']
            active_trade['exit_reason'] = '期間終了'
            strategy1_trades.append(active_trade)
        
        # 結果集計
        return self.calculate_statistics(symbol, start_date, end_date, strategy1_trades, strategy2_trades, debug_info)

    def calculate_statistics(self, symbol: str, start_date: str, end_date: str,
                             strategy1_trades: List[Dict], strategy2_trades: List[Dict], debug_info: Dict) -> Dict:
        """トレード結果の統計を計算"""
        
        # --- 全トレード（最終結果）の統計 ---
        all_returns = [t['return'] for t in strategy1_trades if 'return' in t]
        wins = [r for r in all_returns if r > 0]
        total_trades = len(strategy1_trades)

        # --- 戦略1で始まり、戦略1で終了したトレード ---
        s1_only_trades = [t for t in strategy1_trades if t.get('status') == 'strategy1' and 'return' in t]
        s1_only_returns = [t['return'] for t in s1_only_trades]
        s1_only_wins = [r for r in s1_only_returns if r > 0]

        # --- 戦略2に移行したトレードの統計 ---
        # strategy2_tradesには移行した時点のデータが入っている
        # 最終的なリターンはstrategy1_tradesから取得する
        s2_final_trades = []
        for s2_trade in strategy2_trades:
            # 対応する最終トレードを見つける
            for final_trade in strategy1_trades:
                if final_trade['entry_date'] == s2_trade['entry_date'] and 'return' in final_trade:
                    # s2移行時の情報と最終結果をマージ
                    merged_trade = final_trade.copy()
                    merged_trade.update(s2_trade)
                    s2_final_trades.append(merged_trade)
                    break
        
        s2_returns = [t['return'] for t in s2_final_trades]
        s2_wins = [r for r in s2_returns if r > 0]

        return {
            'symbol': symbol,
            'period': f"{start_date} - {end_date}",
            'total_trades': total_trades,
            'win_rate': len(wins) / total_trades * 100 if total_trades > 0 else 0,
            'avg_return': np.mean(all_returns) * 100 if all_returns else 0,
            'max_profit': max(all_returns) * 100 if all_returns else 0,
            'max_loss': min(all_returns) * 100 if all_returns else 0,

            's1_stats': {
                'entry_count': debug_info.get('strategy1_entries', 0),
                'win_rate_s1_only': len(s1_only_wins) / len(s1_only_trades) * 100 if s1_only_trades else 0,
                'avg_return_s1_only': np.mean(s1_only_returns) * 100 if s1_only_returns else 0,
            },
            's2_stats': {
                'entry_count': debug_info.get('strategy2_entries', 0),
                'conversion_rate': debug_info.get('strategy2_entries', 0) / debug_info.get('strategy1_entries', 1) * 100,
                'win_rate': len(s2_wins) / len(s2_final_trades) * 100 if s2_final_trades else 0,
                'avg_return': np.mean(s2_returns) * 100 if s2_returns else 0,
            },

            'strategy1_final_trades': strategy1_trades,
            'strategy2_transition_trades': strategy2_trades,
            'debug_info': debug_info
        }

    def create_summary_report(self, result: Dict) -> str:
        """バックテスト結果のサマリーレポートを作成"""
        if result.get('error'):
            return f"エラー: {result['error']}"
    
        s1 = result['s1_stats']
        s2 = result['s2_stats']
        debug = result['debug_info']

        report = f"""
📊 新戦略 バックテスト結果 - {result['symbol']}
期間: {result['period']}
━━━━━━━━━━━━━━━━━━━━━━━━

📈 戦略1: FVGサポートでのエントリー
• エントリー数: {s1['entry_count']}回
• (S1のみで決済) 勝率: {s1['win_rate_s1_only']:.1f}%
• (S1のみで決済) 平均リターン: {s1['avg_return_s1_only']:.2f}%

🚀 戦略2: レジスタンス・ブレイクアウト
• 移行数: {s2['entry_count']}回 (S1からの転換率: {s2['conversion_rate']:.1f}%)
• (S2移行後) 勝率: {s2['win_rate']:.1f}%
• (S2移行後) 平均リターン: {s2['avg_return']:.2f}%

💰 全体パフォーマンス (最終結果):
• 総トレード数: {result['total_trades']}回
• 勝率: {result['win_rate']:.1f}%
• 平均リターン: {result['avg_return']:.2f}%
• 最大利益: {result['max_profit']:.2f}%
• 最大損失: {result['max_loss']:.2f}%

🔍 デバッグ情報:
• 分析日数: {debug['total_days']}日
• MA有効日数: {debug['days_with_valid_ma']}日
• 週足MA上日数: {debug['days_above_weekly_ma']}日 (条件合致)
• 日足MA攻防日数: {debug['days_in_ma_zone']}日 (セットアップ)
• FVG検出数: {debug['fvg_detected_count']}回
• FVGリテストエントリー数: {debug['strategy1_entries']}回
• レジスタンス突破数: {debug['strategy2_entries']}回

📋 最近のトレード例:
"""
        # 最新のトレード例を表示
        for trade in result['strategy1_final_trades'][-5:]:
            outcome = "✅" if trade.get('return', 0) > 0 else "❌"
            # statusは最終的な状態を示す
            s2_marker = "🚀S2" if trade.get('status') == 'strategy2' else "S1"
            report += f"\n• {outcome} {trade['entry_date'].strftime('%Y-%m-%d')} "
            report += f"→ {trade['exit_date'].strftime('%Y-%m-%d')}: "
            report += f"{trade.get('return', 0)*100:.1f}% ({s2_marker})"

        return report


# 使用例
if __name__ == "__main__":
    # バックテスト実行
    backtester = FVGBreakBacktest(
        ma_period=50,             # 日足MA期間
        fvg_min_gap=0.1,          # FVG検出の最小ギャップ
        resistance_lookback=50,   # レジスタンス探索期間
        breakout_threshold=1.005, # ブレイクアウトの強さ
        stop_loss_rate=0.03,      # ストップロス率
        target_profit_rate=0.1,   # 利確率
        ma_proximity_percent=0.03 # 日足MAへの近接度
    )

    # テスト対象の銘柄
    symbols = ["NVDA", "AAPL", "MSFT"]
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}")
        print('='*50)
        
        result = backtester.run_backtest(
            symbol=symbol,
            start_date="2023-01-01",
            end_date="2024-07-15"
        )

        # レポート出力
        print(backtester.create_summary_report(result))
        
        # 詳細なトレード情報
        if not result.get('error'):
            print(f"\n最終トレード詳細 (最新5件):")
            for trade in result['strategy1_final_trades'][-5:]:
                s2_info = ""
                if trade.get('status') == 'strategy2':
                    s2_date = trade.get('entry_date_s2', trade['entry_date']).strftime('%Y-%m-%d')
                    s2_price = trade.get('entry_price_s2', trade['entry_price'])
                    s2_info = f" -> S2移行: {s2_date} @ ${s2_price:.2f}"

                print(f"  エントリー: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}{s2_info}")
                print(f"  エグジット: {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f} ({trade['exit_reason']})")
                print(f"  リターン: {trade.get('return', 0)*100:.2f}%")
                print("-" * 20)
