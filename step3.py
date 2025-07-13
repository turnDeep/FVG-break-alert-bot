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
        """最適化のスコアを計算（改訂案）"""
        if result['total_trades'] == 0:
            return -1000  # トレードなしは変わらず大きなペナルティ

        # トレードがあった場合の基本スコア
        base_score = 10

        avg_return = result['avg_return']
        win_rate = result['combined_win_rate'] / 100

        # 勝率とリターンでスコアを調整
        score = base_score + (avg_return * 100) + (win_rate * 20)

        # トレード回数に応じたボーナス/ペナルティ
        if result['total_trades'] < 5:
            score *= 0.8 # 少なすぎる場合は少し減点

        if result['max_loss'] > 10:
            score *= 0.7 # 大きな損失は減点

        return score

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
            target_profit_rate=params['target_profit_rate']
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
        """Optunaの目的関数"""
        # パラメータの探索範囲
        params = {
            'ma_period': trial.suggest_int('ma_period', 20, 200, step=10), # 範囲を少し下に広げる
            'fvg_min_gap': trial.suggest_float('fvg_min_gap', 0.05, 1.5, step=0.05), # 下限を大幅に下げる
            'resistance_lookback': trial.suggest_int('resistance_lookback', 5, 50, step=5), # 範囲を広げる
            'breakout_threshold': trial.suggest_float('breakout_threshold', 1.0005, 1.015, step=0.0005), # 下限を下げ、刻みを細かくする
            'stop_loss_rate': trial.suggest_float('stop_loss_rate', 0.01, 0.1, step=0.005), # 範囲を広げる
            'target_profit_rate': trial.suggest_float('target_profit_rate', 0.02, 0.2, step=0.01) # 範囲を広げる
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
        """最適パラメータを別期間でテスト"""
        if not self.best_params:
            raise ValueError("最適化を先に実行してください")

        print(f"\nテスト期間（{test_period_start} - {test_period_end}）で検証中...")

        # テスト用銘柄（ランダムに50銘柄選択、ただし取得可能な銘柄数を超えないようにする）
        available_symbols = self.get_sp500_symbols()
        sample_size = min(50, len(available_symbols))
        test_symbols = np.random.choice(available_symbols, sample_size, replace=False)

        # バックテスト実行
        test_score = self.evaluate_parameters(
            self.best_params,
            test_symbols,
            test_period_start,
            test_period_end
        )

        print(f"テスト期間のスコア: {test_score:.2f}")

        # 個別銘柄の詳細結果
        backtester = FVGBreakBacktest(**self.best_params)
        detailed_results = []

        for symbol in test_symbols[:10]:  # 上位10銘柄の詳細
            try:
                result = backtester.run_backtest(symbol, test_period_start, test_period_end)
                if not result.get('error'):
                    detailed_results.append({
                        'symbol': symbol,
                        'win_rate': result['combined_win_rate'],
                        'avg_return': result['avg_return'],
                        'total_trades': result['total_trades']
                    })
            except:
                continue

        return {
            'test_score': test_score,
            'detailed_results': detailed_results
        }

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

def main():
    """メイン実行関数"""
    # 最適化インスタンスを作成
    optimizer = FVGParameterOptimizer(
        n_trials=5,
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
