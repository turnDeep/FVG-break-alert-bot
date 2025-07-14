"""
機械学習によるFVG Break Alert Botパラメータ最適化
S&P500全銘柄で学習し、過学習を防ぎながら最適なパラメータを探索
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from io import BytesIO
warnings.filterwarnings('ignore')

from backtest import FVGBreakBacktest

class FVGParameterOptimizer:
    """FVGパラメータの機械学習最適化クラス"""

    def __init__(self, n_trials=100, n_jobs=4):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.best_params = None
        self.optimization_results = []

    def get_sp500_symbols(self):
        """S&P500銘柄リストを取得"""
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            symbols = sp500['Symbol'].str.replace('.', '-').tolist()
            return symbols[:100]  # デモ用に100銘柄に制限（全銘柄なら[:100]を削除）
        except:
            # フォールバック
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']

    def calculate_score(self, result):
        """戦略1と2の両方を評価し、戦略2を重視するスコア計算"""
        s1_stats = result['s1_stats']
        s2_stats = result['s2_stats']

        # 戦略1が発生しない場合は0点以下
        if s1_stats['count'] == 0:
            return -1000

        # --- 各戦略のコアスコアを計算 ---
        # 「平均リターン(%) × 勝率(0-1)」を基本スコアとする
        s1_score = s1_stats['avg_return'] * (s1_stats['win_rate'] / 100)

        s2_score = 0
        if s2_stats['count'] > 0:
            s2_score = s2_stats['avg_return'] * (s2_stats['win_rate'] / 100)

        # --- 最終スコアを比重付けして合算 ---
        # 戦略1: 40%, 戦略2: 60%
        final_score = (s1_score * 0.4) + (s2_score * 0.6)

        # --- 調整ファクター ---
        # 1. 戦略2への転換率ボーナス
        # 転換率が高いほど、そのパラメータは有望とみなし加点
        final_score += (s2_stats['conversion_rate'] / 100) * 5 # 最大5点

        # 2. トレード数ペナルティ
        # 戦略1のトレードが少なすぎると信頼性が低いとみなし減点
        if s1_stats['count'] < 10:
            final_score -= (10 - s1_stats['count']) * 2 # 10回未満は最大-18点

        # 3. 最大損失ペナルティ
        # 全体での最大損失が大きい場合は減点
        if result['max_loss'] < -15:  # 15%以上の損失
            final_score -= abs(result['max_loss'])

        return final_score

    def evaluate_parameters(self, params, symbols, start_date, end_date):
        """パラメータセットを複数銘柄で評価"""
        scores = []
        trade_counts = []

        # バックテスターを作成
        backtester = FVGBreakBacktest(
            ma_period=params['ma_period'],
            fvg_min_gap=params['fvg_min_gap'],
            resistance_lookback=params['resistance_lookback'],
            breakout_threshold=params['breakout_threshold'],
            stop_loss_rate=params['stop_loss_rate'],
            target_profit_rate=params['target_profit_rate'],
            ma_proximity_percent=params['ma_proximity_percent']
        )

        # 各銘柄でバックテスト
        for symbol in symbols:
            try:
                result = backtester.run_backtest(symbol, start_date, end_date)
                if not result.get('error'):
                    score = self.calculate_score(result)
                    scores.append(score)
                    trade_counts.append(result['total_trades'])
            except Exception as e:
                print(f"Error with {symbol}: {e}")
                continue

        if not scores:
            return -1000

        # 平均スコアを返す（外れ値の影響を減らすため中央値も考慮）
        avg_score = np.mean(scores)
        median_score = np.median(scores)
        total_trades = sum(trade_counts)
        
        # 最終スコア（平均と中央値の組み合わせ）
        final_score = avg_score * 0.7 + median_score * 0.3

        # トレード数が少なすぎる場合はペナルティ
        if total_trades < len(symbols) * 2:
            final_score *= 0.8

        return final_score

    def objective(self, trial, train_symbols, val_symbols, start_date, end_date):
        """探索範囲を拡大"""
        params = {
            'ma_period': trial.suggest_int('ma_period', 20, 200, step=10),
            'fvg_min_gap': trial.suggest_float('fvg_min_gap', 0.1, 1.0, step=0.1),  # 上限を下げる
            'resistance_lookback': trial.suggest_int('resistance_lookback', 5, 30, step=5),
            'breakout_threshold': trial.suggest_float('breakout_threshold', 1.0, 1.01, step=0.001),  # より小さい値も試す
            'stop_loss_rate': trial.suggest_float('stop_loss_rate', 0.01, 0.05, step=0.005),
            'target_profit_rate': trial.suggest_float('target_profit_rate', 0.01, 0.1, step=0.01),  # 小さい利益も狙う
            'ma_proximity_percent': trial.suggest_float('ma_proximity_percent', 0.05, 0.20, step=0.05),  # 新パラメータ
        }

        # 訓練セットで評価
        train_score = self.evaluate_parameters(params, train_symbols, start_date, end_date)

        # 検証セットで評価（過学習チェック）
        val_score = self.evaluate_parameters(params, val_symbols, start_date, end_date)

        # 訓練と検証のバランスを取る
        combined_score = train_score * 0.6 + val_score * 0.4

        # 結果を保存
        self.optimization_results.append({
            'params': params,
            'train_score': train_score,
            'val_score': val_score,
            'combined_score': combined_score
        })

        return combined_score

    def optimize(self, start_date='2022-01-01', end_date='2024-01-01'):
        """メイン最適化処理"""
        print("S&P500銘柄リストを取得中...")
        all_symbols = self.get_sp500_symbols()
        print(f"取得銘柄数: {len(all_symbols)}")

        # 銘柄を訓練用と検証用に分割（8:2）
        np.random.shuffle(all_symbols)
        split_idx = int(len(all_symbols) * 0.8)
        train_symbols = all_symbols[:split_idx]
        val_symbols = all_symbols[split_idx:]

        print(f"訓練銘柄数: {len(train_symbols)}, 検証銘柄数: {len(val_symbols)}")

        # Optuna最適化
        print("パラメータ最適化を開始...")
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        # 目的関数のラッパー
        def wrapped_objective(trial):
            return self.objective(trial, train_symbols, val_symbols, start_date, end_date)

        study.optimize(
            wrapped_objective,
            n_trials=self.n_trials,
            n_jobs=1  # 並列実行時の問題を避けるため
        )

        # 最適パラメータを保存
        self.best_params = study.best_params
        print(f"\n最適化完了！最高スコア: {study.best_value:.2f}")
        print(f"最適パラメータ: {self.best_params}")

        return self.best_params

    def validate_best_params(self, test_period_start='2024-01-01', test_period_end='2024-12-31'):
        """最適パラメータを別期間でテストし、戦略別の合算結果を返す"""
        if not self.best_params:
            raise ValueError("最適化を先に実行してください")

        print(f"\nテスト期間（{test_period_start} - {test_period_end}）で検証中...")

        test_symbols = self.get_sp500_symbols()
        print(f"全{len(test_symbols)}銘柄でテストを実行します...")

        backtester = FVGBreakBacktest(**self.best_params)
        all_s1_trades = []
        all_s2_trades = []
        symbols_with_errors = 0

        for symbol in test_symbols:
            try:
                result = backtester.run_backtest(symbol, test_period_start, test_period_end)
                if not result.get('error'):
                    all_s1_trades.extend(result['strategy1_trades'])
                    all_s2_trades.extend(result['strategy2_trades'])
            except Exception as e:
                print(f"エラー: {symbol} のバックテスト中に問題が発生 - {e}")
                symbols_with_errors += 1
                continue

        if not all_s1_trades:
            return {'error': 'テスト期間中にトレードが一件もありませんでした。'}

        # 戦略1の合算パフォーマンス
        s1_returns = [t['return'] for t in all_s1_trades if 'return' in t]
        s1_wins = [r for r in s1_returns if r > 0]

        # 戦略2の合算パフォーマンス
        s2_final_trades = [t for t in all_s1_trades if t.get('s2_triggered') and 'return' in t]
        s2_returns = [t['return'] for t in s2_final_trades]
        s2_wins = [r for r in s2_returns if r > 0]

        # 全体サマリー
        summary = {
            'symbols_tested': len(test_symbols),
            'symbols_with_errors': symbols_with_errors,
            's1_stats': {
                'count': len(all_s1_trades),
                'win_rate': len(s1_wins) / len(all_s1_trades) * 100 if all_s1_trades else 0,
                'avg_return': np.mean(s1_returns) * 100 if s1_returns else 0,
            },
            's2_stats': {
                'count': len(all_s2_trades),
                'conversion_rate': len(all_s2_trades) / len(all_s1_trades) * 100 if all_s1_trades else 0,
                'win_rate': len(s2_wins) / len(s2_final_trades) * 100 if s2_final_trades else 0,
                'avg_return': np.mean(s2_returns) * 100 if s2_returns else 0,
            }
        }
        return summary

    def save_results(self, filename='optimized_params.json'):
        """最適化結果を保存"""
        results = {
            'best_params': self.best_params,
            'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_history': sorted(
                self.optimization_results,
                key=lambda x: x['combined_score'],
                reverse=True
            )[:10]  # トップ10を保存
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n結果を{filename}に保存しました")

        # .envファイル用の設定も出力
        print("\n.envファイルに以下を設定してください:")
        print(f"MA_PERIOD={self.best_params['ma_period']}")
        print(f"FVG_MIN_GAP_PERCENT={self.best_params['fvg_min_gap']}")
        print(f"RESISTANCE_LOOKBACK={self.best_params['resistance_lookback']}")
        print(f"BREAKOUT_THRESHOLD={self.best_params['breakout_threshold']}")
        print(f"STOP_LOSS_RATE={self.best_params['stop_loss_rate']}")
        print(f"TARGET_PROFIT_RATE={self.best_params['target_profit_rate']}")
        print(f"MA_PROXIMITY_PERCENT={self.best_params['ma_proximity_percent']}")

    def plot_optimization_history(self):
        """最適化履歴をプロット"""
        import matplotlib.pyplot as plt

        if not self.optimization_results:
            print("最適化結果がありません")
            return

        # スコアの推移
        scores = [r['combined_score'] for r in self.optimization_results]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Optimization Score Progress')
        plt.xlabel('Trial')
        plt.ylabel('Score')
        plt.grid(True)

        # パラメータ分布
        plt.subplot(1, 2, 2)
        top_results = sorted(self.optimization_results, key=lambda x: x['combined_score'], reverse=True)[:20]
        ma_periods = [r['params']['ma_period'] for r in top_results]
        fvg_gaps = [r['params']['fvg_min_gap'] for r in top_results]

        plt.scatter(ma_periods, fvg_gaps, c=[r['combined_score'] for r in top_results], cmap='viridis')
        plt.colorbar(label='Score')
        plt.xlabel('MA Period')
        plt.ylabel('FVG Min Gap %')
        plt.title('Top 20 Parameter Combinations')

        plt.tight_layout()
        plt.savefig('optimization_results.png')
        print("最適化結果をoptimization_results.pngに保存しました")

    def create_example_chart(self, test_period_start, test_period_end):
        """戦略2の成功例チャートを作成"""
        if not self.best_params:
            print("チャート作成スキップ: 最適パラメータがありません。")
            return

        print("\n📈 戦略2の成功例を探してチャートを作成中...")

        # 戦略2に到達した成功トレード（利益が出たもの）を探す
        test_symbols = self.get_sp500_symbols()
        backtester = FVGBreakBacktest(**self.best_params)
        successful_s2_trades = []

        for symbol in test_symbols[:20]:  # 最初の20銘柄のみチェック（効率化）
            try:
                result = backtester.run_backtest(symbol, test_period_start, test_period_end)
                if not result.get('error'):
                    for trade in result.get('strategy1_trades', []):
                        if trade.get('s2_triggered') and trade.get('return', 0) > 0:
                            # シンボル情報も追加
                            trade['symbol'] = symbol
                            successful_s2_trades.append(trade)
            except Exception as e:
                print(f"Warning: チャート用データ取得エラー ({symbol}): {e}")
                continue

        if not successful_s2_trades:
            print("チャート作成スキップ: 戦略2の成功例が見つかりませんでした。")
            return

        # ランダムに1つ選ぶ
        import random
        trade_example = random.choice(successful_s2_trades)
        symbol = trade_example['symbol']

        try:
            # チャート用のデータを取得
            stock = yf.Ticker(symbol)
            
            # 日付範囲を検証
            entry_date = pd.to_datetime(trade_example['entry_date'])
            exit_date = pd.to_datetime(trade_example['exit_date'])
            chart_start = entry_date - timedelta(days=60)
            chart_end = exit_date + timedelta(days=10)
            
            # テスト期間内に収まるよう調整
            period_start = pd.to_datetime(test_period_start)
            period_end = pd.to_datetime(test_period_end)
            chart_start = max(chart_start, period_start - timedelta(days=30))
            chart_end = min(chart_end, period_end + timedelta(days=10))
            
            df = stock.history(start=chart_start, end=chart_end)

            if df.empty:
                print(f"チャート作成エラー: {symbol}のデータ取得に失敗しました。")
                return

            # タイムゾーンを統一（全てtz-naiveに変換）
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 日付をtz-naiveに変換
            def to_naive_date(date_obj):
                if isinstance(date_obj, str):
                    date_obj = pd.to_datetime(date_obj)
                if hasattr(date_obj, 'tz') and date_obj.tz is not None:
                    return date_obj.tz_localize(None)
                return date_obj

            entry_date = to_naive_date(entry_date)
            exit_date = to_naive_date(exit_date)

            # 戦略2トリガー日を探す
            s2_trigger_date = None
            try:
                s2_result = backtester.run_backtest(symbol, test_period_start, test_period_end)
                for s2_trade in s2_result.get('strategy2_trades', []):
                    if pd.to_datetime(s2_trade['entry_date']) == entry_date:
                        s2_trigger_date = to_naive_date(s2_trade.get('entry_date_s2'))
                        break
            except Exception as e:
                print(f"Warning: S2トリガー日取得エラー: {e}")

            # スタイル設定
            mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick={'up':'green', 'down':'red'}, volume='in')
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)

            # プロット作成
            fig, axes = mpf.plot(df, type='candle', style=s, volume=True,
                                 title=f"Example: {symbol} - Strategy 2 Success",
                                 returnfig=True, figsize=(15, 10))

            ax = axes[0]

            # FVGゾーン
            fvg = trade_example['fvg_info']
            try:
                fvg_date = to_naive_date(fvg['date'])
                if fvg_date in df.index:
                    fvg_start_loc = df.index.get_loc(fvg_date) - 2
                    rect = patches.Rectangle((fvg_start_loc, fvg['gap_bottom']), 2, fvg['gap_top'] - fvg['gap_bottom'],
                                             linewidth=1, edgecolor='cyan', facecolor='cyan', alpha=0.3)
                    ax.add_patch(rect)
            except Exception as e:
                print(f"Warning: FVGゾーン描画エラー: {e}")

            # アノテーション
            def annotate_point(date, text, y_price, color):
                try:
                    if date in df.index:
                        x_loc = df.index.get_loc(date)
                        ax.annotate(text, (x_loc, y_price),
                                    xytext=(x_loc + 5, y_price * 1.05),
                                    arrowprops=dict(facecolor=color, shrink=0.05),
                                    fontsize=12, color='white',
                                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec='none', alpha=0.8))
                except Exception as e:
                    print(f"Warning: アノテーション描画エラー ({text}): {e}")

            # アノテーション追加
            annotate_point(entry_date, 'S1 Entry', trade_example['entry_price'], 'blue')
            if s2_trigger_date and s2_trigger_date in df.index:
                s2_price = df.loc[s2_trigger_date]['Close']
                annotate_point(s2_trigger_date, 'S2 Trigger', s2_price, 'orange')
            annotate_point(exit_date, 'Exit', trade_example['exit_price'], 'purple')

            # 保存
            save_path = 'example_trade_chart.png'
            fig.savefig(save_path)
            print(f"✅ サンプルチャートを {save_path} に保存しました。")
            
        except Exception as e:
            print(f"チャート作成エラー: {e}")
            print("チャート作成をスキップして続行します。")


def main():
    """メイン実行関数"""
    # 最適化インスタンスを作成
    optimizer = FVGParameterOptimizer(
        n_trials=50,  # デモ用に50回（本番は100-200推奨）
        n_jobs=4
    )

    # 最適化実行（2年間のデータで学習）
    best_params = optimizer.optimize(
        start_date='2022-01-01',
        end_date='2024-01-01'
    )

    # 別期間でテスト
    test_results = optimizer.validate_best_params(
        test_period_start='2024-01-01',
        test_period_end='2024-12-31'
    )

    # 結果を保存
    optimizer.save_results()

    # 可視化
    optimizer.plot_optimization_history()

    # 詳細結果を表示
    print("\n=== テスト期間の上位銘柄パフォーマンス ===")
    for result in test_results['detailed_results'][:5]:
        print(f"{result['symbol']}: 勝率 {result['win_rate']:.1f}%, "
              f"平均リターン {result['avg_return']:.2f}%, "
              f"トレード数 {result['total_trades']}")

if __name__ == "__main__":
    main()
