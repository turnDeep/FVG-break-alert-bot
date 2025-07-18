import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import yfinance as yf
import optuna
from backtest import FVGBreakBacktest
from indicators import AdvancedIndicators, FVGAnalyzer
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
import random
warnings.filterwarnings('ignore')

class OptimizationManager:
    """
    ML最適化、バックテスト、レポート生成を管理するクラス
    """
    def __init__(self, n_trials=100, n_jobs=-1):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study = None
        self.best_params = None
        self.results_dir = "optimization_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def get_sp500_symbols(self):
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return sp500['Symbol'].str.replace('.', '-').tolist()
        except Exception as e:
            print(f"S&P500リスト取得エラー: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    def objective(self, trial, symbols, start_date, end_date):
        params = {
            'ma_period': trial.suggest_int('ma_period', 50, 200, step=10),
            'fvg_min_gap': trial.suggest_float('fvg_min_gap', 0.2, 1.0, step=0.1),
            'resistance_lookback': trial.suggest_int('resistance_lookback', 10, 40, step=5),
            'breakout_threshold': trial.suggest_float('breakout_threshold', 1.001, 1.01, step=0.001),
            'stop_loss_rate': trial.suggest_float('stop_loss_rate', 0.01, 0.05, step=0.005),
            'target_profit_rate': trial.suggest_float('target_profit_rate', 0.02, 0.1, step=0.01),
            'ma_proximity_percent': trial.suggest_float('ma_proximity_percent', 0.03, 0.15, step=0.01),
        }

        backtester = FVGBreakBacktest(**params)
        scores = []
        trade_counts = []

        for symbol in symbols:
            try:
                result = backtester.run_backtest(symbol, start_date, end_date)
                if not result.get('error') and result['total_trades'] > 0:
                    score = self._calculate_score(result)
                    scores.append(score)
                    trade_counts.append(result['total_trades'])
            except Exception:
                continue

        if not scores:
            return -1000.0

        avg_score = np.mean(scores)
        # トレードが少ない場合はペナルティ
        if sum(trade_counts) < len(symbols):
            avg_score *= 0.8

        return avg_score

    def _calculate_score(self, result):
        s1 = result['s1_stats']
        s2 = result['s2_stats']

        if s1['count'] == 0: return -1000

        # シャープレシオを計算 (簡易版)
        returns = [t['return'] for t in result['strategy1_trades']]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        score = (s1['avg_return'] * (s1['win_rate'] / 100) * 0.5 +
                 s2['avg_return'] * (s2['win_rate'] / 100) * 0.5)

        score += s2['conversion_rate'] / 10 # 転換率ボーナス
        score += sharpe_ratio * 5 # シャープレシオボーナス

        if result['max_loss'] < -15:
            score -= abs(result['max_loss']) # 大きな損失はペナルティ

        return score

    def run_optimization(self, start_date, end_date, n_trials=None):
        self.n_trials = n_trials if n_trials else self.n_trials
        all_symbols = self.get_sp500_symbols()
        np.random.shuffle(all_symbols)
        # 銘柄数を絞る (高速化のため)
        symbols_to_use = all_symbols[:50]

        self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        self.study.optimize(
            lambda trial: self.objective(trial, symbols_to_use, start_date, end_date),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        self.best_params = self.study.best_params
        return self.best_params

    def run_full_backtest(self, params, start_date, end_date):
        """全S&P500銘柄で詳細なバックテストを実行"""
        all_symbols = self.get_sp500_symbols()
        backtester = FVGBreakBacktest(**params)

        all_s1_trades, all_s2_trades = [], []
        trade_examples = {'s1': [], 's2': []}

        for i, symbol in enumerate(all_symbols):
            print(f"Backtesting {symbol} ({i+1}/{len(all_symbols)})...")
            try:
                result = backtester.run_backtest(symbol, start_date, end_date)
                if result.get('error') or result['total_trades'] == 0:
                    continue

                all_s1_trades.extend(result['strategy1_trades'])
                all_s2_trades.extend(result['strategy2_trades'])

                # トレード例を収集 (exit_dateとreturnが存在するもののみ)
                s1_wins = [t for t in result['strategy1_trades'] if t.get('return', 0) > 0.05 and 'exit_date' in t]
                if s1_wins and len(trade_examples['s1']) < 10:
                    trade_examples['s1'].append(random.choice(s1_wins))

                s2_trades_with_data = [
                    t for t in result['strategy2_trades']
                    if any(s1['entry_date'] == t['entry_date'] and s1.get('return', 0) > 0.08 and 'exit_date' in s1 for s1 in result['strategy1_trades'])
                ]
                if s2_trades_with_data and len(trade_examples['s2']) < 10:
                    trade_examples['s2'].append(random.choice(s2_trades_with_data))

            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                continue

        return self._compile_backtest_results(all_s1_trades, all_s2_trades, trade_examples, start_date, end_date)

    def _compile_backtest_results(self, s1_trades, s2_trades, examples, start, end):
        # ... (設計書通りの詳細なバックテスト結果を生成するロジック)
        def calc_stats(trades):
            if not trades:
                return {'total_trades': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0}
            returns = [t['return'] for t in trades if 'return' in t]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            return {
                'total_trades': len(trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'avg_win': np.mean(wins) * 100 if wins else 0,
                'avg_loss': np.mean(losses) * 100 if losses else 0,
                'profit_factor': abs(sum(wins) / sum(losses)) if sum(losses) != 0 else 999
            }

        s1_final_trades = [t for t in s1_trades if 'return' in t]
        s2_final_trades = [t for t in s1_trades if t.get('s2_triggered') and 'return' in t]

        return {
            'test_period': {'start': start, 'end': end},
            'strategy1_detailed': {
                'performance_summary': calc_stats(s1_final_trades),
                'trade_examples': examples['s1']
            },
            'strategy2_detailed': {
                'performance_summary': calc_stats(s2_final_trades),
                'trade_examples': examples['s2']
            }
        }


    def save_optimization_results(self, backtest_results):
        """設計書に従って最適化結果をJSONに保存"""
        if not self.study:
            raise ValueError("Optimization has not been run yet.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"optimization_{timestamp}.json")

        results = {
            'timestamp': timestamp,
            'optimization_summary': {
                'total_trials': len(self.study.trials),
                'best_score': self.study.best_value,
                'best_params': self.study.best_params,
            },
            'backtest_results': backtest_results,
            'all_trials': [t.values for t in self.study.trials],
        }

        # Convert datetime objects to strings for JSON serialization
        def default_converter(o):
            if isinstance(o, (datetime, pd.Timestamp)):
                return o.isoformat()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=default_converter)

        print(f"Optimization results saved to {filepath}")
        return filepath

    def generate_html_report(self, results_filepath):
        """設計書に従ってHTMLダッシュボードを生成"""
        with open(results_filepath, 'r') as f:
            data = json.load(f)

        timestamp = data['timestamp']
        summary = data['optimization_summary']
        params = summary['best_params']
        s1_results = data['backtest_results']['strategy1_detailed']
        s2_results = data['backtest_results']['strategy2_detailed']

        def format_metrics(metrics):
            return f"""
            <ul>
                <li>Total Trades: {metrics['total_trades']}</li>
                <li>Win Rate: {metrics['win_rate']:.2f}%</li>
                <li>Avg Win: {metrics['avg_win']:.2f}%</li>
                <li>Avg Loss: {metrics['avg_loss']:.2f}%</li>
                <li>Profit Factor: {metrics['profit_factor']:.2f}</li>
            </ul>
            """

        def format_trades(trades, s_type='s1'):
            html = ""
            if not trades:
                return "<p>No trade examples found.</p>"
            for t in trades[:5]: # Show top 5 examples
                outcome = 'win' if t.get('return', 0) > 0 else 'loss'
                entry_date_str = t.get('entry_date', 'N/A')
                exit_date_str = t.get('exit_date', 'N/A')
                # 日付オブジェクトを文字列に変換
                if isinstance(entry_date_str, datetime):
                    entry_date_str = entry_date_str.strftime('%Y-%m-%d')
                if isinstance(exit_date_str, datetime):
                    exit_date_str = exit_date_str.strftime('%Y-%m-%d')

                html += f"""
                <div class="trade-example {outcome}">
                    <p><b>{t.get('symbol', 'N/A')}</b> ({entry_date_str})</p>
                    <p>Return: {t.get('return', 0)*100:.2f}% | Exit: {exit_date_str}</p>
                </div>
                """
            return html

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FVG Bot ML Optimization Results - {timestamp}</title>
            <style>
                body {{ font-family: sans-serif; margin: 2em; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .win {{ background-color: #e8f5e9; }}
                .loss {{ background-color: #ffebee; }}
                .trade-example {{ border: 1px solid #eee; padding: 10px; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>FVG Bot ML Optimization Report</h1>
                <p><b>Generated:</b> {timestamp}</p>

                <h2>Optimization Summary</h2>
                <div class="card">
                    <p><b>Best Score:</b> {summary['best_score']:.4f}</p>
                    <p><b>Total Trials:</b> {summary['total_trials']}</p>
                    <h3>Best Parameters:</h3>
                    <pre>{json.dumps(params, indent=2)}</pre>
                </div>

                <h2>Backtest Results</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Strategy 1 (FVG Detect)</h3>
                        {format_metrics(s1_results['performance_summary'])}
                    </div>
                    <div class="card">
                        <h3>Strategy 2 (Resistance Break)</h3>
                        {format_metrics(s2_results['performance_summary'])}
                    </div>
                </div>

                <h2>Trade Examples</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Strategy 1 Examples</h3>
                        {format_trades(s1_results['trade_examples'], 's1')}
                    </div>
                    <div class="card">
                        <h3>Strategy 2 Examples</h3>
                        {format_trades(s2_results['trade_examples'], 's2')}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        report_path = os.path.join(self.results_dir, f"optimization_report_{timestamp}.html")
        with open(report_path, 'w') as f:
            f.write(html_template)

        print(f"HTML report saved to {report_path}")
        return report_path

if __name__ == '__main__':
    # --- 使用例 ---
    import random

    manager = OptimizationManager(n_trials=20) # デモ用に試行回数を減らす

    # 1. 最適化実行
    print("Running optimization...")
    best_params = manager.run_optimization(
        start_date="2023-01-01",
        end_date="2023-06-30"
    )
    print(f"Best parameters found: {best_params}")

    # 2. フルバックテスト実行
    print("\nRunning full backtest with best parameters...")
    backtest_results = manager.run_full_backtest(
        params=best_params,
        start_date="2023-07-01",
        end_date="2023-12-31"
    )

    # 3. 結果を保存
    print("\nSaving results...")
    json_path = manager.save_optimization_results(backtest_results)

    # 4. HTMLレポートを生成
    print("\nGenerating HTML report...")
    html_path = manager.generate_html_report(json_path)

    print(f"\n✅ Process complete. See {html_path} for details.")
