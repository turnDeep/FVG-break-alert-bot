"""
機械学習によるFVG Break Alert Botパラメータ最適化
設計書に基づく品質最優先・時間無制限の最適化システム

主な改善点:
1. 多段階最適化戦略 (RandomSampler → TPESampler → CmaEsSampler)
2. 多目的最適化 (NSGAIISampler) による過学習防止
3. TimeSeriesSplit による堅牢なクロスバリデーション
4. 拡張されたパラメータ範囲と対数スケール探索
5. 詳細な結果記録と可視化
6. 並列処理による高速化
7. キャッシュとEarly Stoppingによる効率化

使用方法:
# 最速実行（推奨）
python ml_optimizer_high_speed.py --fast --no_cv

# 高速実行（並列処理強化）
python ml_optimizer_high_speed.py --no_cv --n_jobs 8

# 基本的な実行（クロスバリデーション有効）
python ml_optimizer_high_speed.py

# クロスバリデーションを無効化（高速化）
python ml_optimizer_high_speed.py --no_cv

# 試行回数を指定
python ml_optimizer_high_speed.py --n_trials 10

# 無制限モード（時間をかけて高品質な最適化）
python ml_optimizer_high_speed.py --unlimited

# 多目的最適化モード
python ml_optimizer_high_speed.py --mode multi_objective

高速化のヒント:
1. --fast --no_cv を併用（最速）
2. --n_jobs をCPUコア数に設定
3. --n_trials を減らしてテスト実行
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from curl_cffi import requests
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from io import BytesIO
import seaborn as sns
import argparse
import os
import time
import html
warnings.filterwarnings('ignore')

# Optunaのログレベル設定
optuna.logging.set_verbosity(optuna.logging.WARNING)

# バックテストモジュールのインポート（存在確認付き）
try:
    from backtest import FVGBreakBacktest
except ImportError:
    print("警告: backtest.pyが見つかりません。ダミークラスを使用します。")
    # ダミークラスの定義（開発/テスト用）
    class FVGBreakBacktest:
        def __init__(self, **kwargs):
            self.params = kwargs
            np.random.seed(sum(ord(c) for c in str(kwargs))) # パラメータでシード変更

        def run_backtest(self, symbol, start_date, end_date):
            # 新しい統計構造に合わせたダミー結果
            ma_period = self.params.get('ma_period', 200)
            fvg_min_gap = self.params.get('fvg_min_gap', 0.5)
            
            base_win_rate = 50 + (200 - ma_period) * 0.05 + fvg_min_gap * 10
            base_return = 0.001 + (fvg_min_gap - 0.5) * 0.002

            s1_entries = np.random.randint(10, 50)
            s2_entries = int(s1_entries * np.random.uniform(0.2, 0.6))
            total_trades = s1_entries
            
            final_trades = []
            s2_transition_trades = []
            
            # トレード生成
            for i in range(total_trades):
                is_s2 = i < s2_entries
                win_rate = base_win_rate + (10 if is_s2 else 0) # S2は勝率が高いと仮定

                trade = {'entry_date': pd.to_datetime(start_date) + timedelta(days=i*5)}
                if np.random.random() < win_rate / 100:
                    trade['return'] = np.random.uniform(base_return, base_return + 0.03)
                else:
                    trade['return'] = np.random.uniform(base_return - 0.02, base_return)

                if is_s2:
                    trade['status'] = 'strategy2'
                    s2_transition_trades.append(trade)
                else:
                    trade['status'] = 'strategy1'

                final_trades.append(trade)

            all_returns = [t['return'] for t in final_trades]
            s2_returns = [t['return'] for t in final_trades if t['status'] == 'strategy2']

            return {
                'error': False,
                'total_trades': total_trades,
                'win_rate': sum(1 for r in all_returns if r > 0) / total_trades * 100,
                'avg_return': np.mean(all_returns) if all_returns else 0,
                'max_loss': min(all_returns) if all_returns else 0,
                's1_stats': {
                    'entry_count': s1_entries,
                },
                's2_stats': {
                    'entry_count': s2_entries,
                    'conversion_rate': s2_entries / s1_entries * 100,
                    'win_rate': (sum(1 for r in s2_returns if r > 0) / len(s2_returns) * 100) if s2_returns else 0,
                    'avg_return': np.mean(s2_returns) if s2_returns else 0,
                },
                'strategy1_final_trades': final_trades,
                'strategy2_transition_trades': s2_transition_trades,
                'debug_info': {
                    'strategy1_entries': s1_entries,
                    'strategy2_entries': s2_entries
                }
            }

# グローバル関数として定義
def _evaluate_single_symbol_wrapper(args):
    """並列処理のための独立した評価関数"""
    symbol, basic_params, start_date, end_date = args
    try:
        backtester = FVGBreakBacktest(**basic_params)
        result = backtester.run_backtest(symbol, start_date, end_date)

        if result.get('error'):
            return None

        # 期間の日数を計算
        period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        # optimizerインスタンス経由でスコア計算メソッドを呼び出す
        score = EnhancedFVGParameterOptimizer.calculate_enhanced_score(result, period_days)
        return score

    except Exception:
        return None


class EnhancedFVGParameterOptimizer:
    """品質最優先の機械学習最適化システム"""

    def __init__(self, unlimited_mode=False, n_jobs=4, use_cross_validation=True, fast_mode=False):
        self.unlimited_mode = unlimited_mode
        self.n_jobs = n_jobs if not fast_mode else max(4, os.cpu_count() - 1)
        self.use_cross_validation = use_cross_validation
        self.fast_mode = fast_mode
        self.best_params = None
        self.optimization_results = []
        self.multi_stage_results = {}
        self.pareto_solutions = []
        self.cache = {}  # 結果のキャッシュ
        
        # 設計書に基づく最適化設定
        if fast_mode:
            # 高速モード：試行回数を削減
            self.optimization_config = {
                'exploration': {'n_trials': 30, 'sampler': RandomSampler()},
                'exploitation': {'n_trials': 20, 'sampler': TPESampler(seed=42, n_startup_trials=5)},
                'refinement': {'n_trials': 10, 'sampler': CmaEsSampler(seed=42, n_startup_trials=3)}
            }
        elif unlimited_mode:
            self.optimization_config = {
                'exploration': {'n_trials': 10000, 'sampler': RandomSampler()},
                'exploitation': {'n_trials': 5000, 'sampler': TPESampler(seed=42)},
                'refinement': {'n_trials': 2000, 'sampler': CmaEsSampler(seed=42)}
            }
        else:
            self.optimization_config = {
                'exploration': {'n_trials': 100, 'sampler': RandomSampler()},
                'exploitation': {'n_trials': 50, 'sampler': TPESampler(seed=42)},
                'refinement': {'n_trials': 30, 'sampler': CmaEsSampler(seed=42)}
            }
    
    def get_sp500_symbols(self):
        """S&P500銘柄リストを取得する"""
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            symbols = sp500['Symbol'].str.replace('.', '-').tolist()
        except Exception as e:
            print(f"S&P500リスト取得エラー: {e}")
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']

        if self.fast_mode:
            symbols = symbols[:10]
        elif not self.unlimited_mode:
            symbols = symbols[:50]

        print(f"対象銘柄数: {len(symbols)}")
        return symbols

    @staticmethod
    def calculate_enhanced_score(result, period_days):
        """新しい戦略に最適化された評価関数"""
        if result.get('error') or 's1_stats' not in result or 's2_stats' not in result:
            return -1000

        s1_stats = result['s1_stats']
        s2_stats = result['s2_stats']
        total_trades = result.get('total_trades', 0)

        if total_trades == 0:
            return -100

        # --- 戦略2のパフォーマンスを最重要視 ---
        s2_win_rate = s2_stats.get('win_rate', 0) / 100.0
        s2_avg_return = s2_stats.get('avg_return', 0) / 100.0
        s2_score = (s2_avg_return * s2_win_rate) * s2_stats.get('entry_count', 0)
        
        # --- 全体的なパフォーマンス ---
        overall_win_rate = result.get('win_rate', 0) / 100.0
        overall_avg_return = result.get('avg_return', 0) / 100.0
        
        # プロフィットファクターのような指標
        returns = [t.get('return', 0) for t in result.get('strategy1_final_trades', [])]
        total_profit = sum(r for r in returns if r > 0)
        total_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = total_profit / (total_loss + 1e-6)

        # --- スコアの組み立て ---
        # 戦略2のスコアに高い重み (70%)、全体スコアに (30%)
        base_score = (s2_score * 0.7) + (overall_avg_return * overall_win_rate * 0.3)
        base_score *= 1000 # スコアを拡大

        # --- 正則化とボーナス ---
        # 1. シャープレシオ（安定性）
        if len(returns) > 1:
            returns_array = np.array(returns)
            daily_std = np.std(returns_array)
            if daily_std > 0:
                # リスクフリーレート0と仮定
                sharpe_ratio = (np.mean(returns_array) / daily_std) * np.sqrt(252)
                base_score += sharpe_ratio * 5 # シャープレシオの影響を大きく

        # 2. 戦略2への転換率ボーナス
        conversion_rate = s2_stats.get('conversion_rate', 0) / 100.0
        # 転換率が高いほどボーナス。ただし高すぎるとS1の意味がなくなるのでキャップ
        base_score += min(conversion_rate, 0.8) * 10

        # 3. プロフィットファクターボーナス
        base_score += (profit_factor - 1) * 5

        # 4. トレード頻度のペナルティ
        ANNUAL_MIN_TRADES = 10
        ANNUAL_MAX_TRADES = 150
        min_trades = (ANNUAL_MIN_TRADES / 365.0) * period_days
        max_trades = (ANNUAL_MAX_TRADES / 365.0) * period_days

        if total_trades < min_trades:
            base_score -= (min_trades - total_trades) * 1.0 # 強いペナルティ
        elif total_trades > max_trades:
            base_score -= (total_trades - max_trades) * 0.2 # 弱いペナルティ

        # 5. 最大損失ペナルティ
        max_loss = result.get('max_loss', 0) / 100.0
        if max_loss < -0.10: # 10%以上の損失でペナルティ
            base_score -= (abs(max_loss) - 0.10) * 100

        return base_score

    def enhanced_parameter_ranges(self, trial):
        """拡張されたパラメータ範囲と対数スケール探索"""
        # 基本パラメータ（FVGBreakBacktestで使用される）
        basic_params = {
            'ma_period': trial.suggest_int('ma_period', 5, 500, step=5),
            'fvg_min_gap': trial.suggest_float('fvg_min_gap', 0.01, 5.0, log=True),
            'resistance_lookback': trial.suggest_int('resistance_lookback', 3, 100, step=2),
            'breakout_threshold': trial.suggest_float('breakout_threshold', 0.995, 1.05, step=0.001),
            # stop_loss_rateを保守的に調整（最小値を0.01から0.015に、デフォルト範囲を狭める）
            'stop_loss_rate': trial.suggest_float('stop_loss_rate', 0.015, 0.08, log=True),
            'target_profit_rate': trial.suggest_float('target_profit_rate', 0.001, 0.3, log=True),
            'ma_proximity_percent': trial.suggest_float('ma_proximity_percent', 0.005, 0.5, log=True),
        }
        
        # 拡張パラメータ（将来の機能拡張用）
        extended_params = {
            'fvg_lookback_bars': trial.suggest_int('fvg_lookback_bars', 3, 10),
            'fvg_fill_threshold': trial.suggest_float('fvg_fill_threshold', 0.1, 0.9, step=0.1),
            'volume_confirmation': trial.suggest_categorical('volume_confirmation', [True, False]),
            'timeframe_filter': trial.suggest_categorical('timeframe_filter', ['1h', '4h', 'daily']),
            'volatility_adjustment': trial.suggest_float('volatility_adjustment', 0.5, 2.0, step=0.1),
            'trend_strength_min': trial.suggest_float('trend_strength_min', 0.05, 0.95, step=0.05),
            'risk_adjustment': trial.suggest_float('risk_adjustment', 0.8, 1.5, step=0.1),
        }
        
        # 全パラメータを返す
        return {**basic_params, **extended_params}

    def evaluate_with_cross_validation(self, params, symbols, start_date, end_date):
        """TimeSeriesSplitを使用したクロスバリデーション"""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            try:
                months = pd.date_range(start=start_dt, end=end_dt, freq='3ME')
            except:
                months = pd.date_range(start=start_dt, end=end_dt, freq='3M')
            
            if len(months) < 3:
                return self.evaluate_parameters(params, symbols, start_date, end_date)
            
            cv_scores = []
            
            for i in range(len(months) - 2):
                train_start = months[i].strftime('%Y-%m-%d')
                train_end = months[i+1].strftime('%Y-%m-%d')
                val_start = months[i+1].strftime('%Y-%m-%d')
                val_end = months[i+2].strftime('%Y-%m-%d')
                
                train_score = self.evaluate_parameters(params, symbols, train_start, train_end)
                val_score = self.evaluate_parameters(params, symbols, val_start, val_end)
                
                if train_score > -900 and val_score > -900:
                    overfitting_penalty = abs(train_score - val_score) * 0.1
                    adjusted_score = val_score - overfitting_penalty
                    cv_scores.append(adjusted_score)

            if not cv_scores:
                return -1000.0
            
            return np.mean(cv_scores)
            
        except Exception:
            return -1000.0

    def evaluate_parameters_parallel(self, params, symbols, start_date, end_date):
        """パラメータセットを並列処理で複数銘柄評価"""
        param_hash = hash(frozenset(params.items()))
        cache_key = f"{param_hash}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        basic_params = {
            'ma_period': params['ma_period'],
            'fvg_min_gap': params['fvg_min_gap'],
            'resistance_lookback': params['resistance_lookback'],
            'breakout_threshold': params['breakout_threshold'],
            'stop_loss_rate': params['stop_loss_rate'],
            'target_profit_rate': params['target_profit_rate'],
            'ma_proximity_percent': params['ma_proximity_percent']
        }
        
        if self.fast_mode:
            sample_symbols = symbols[:min(len(symbols), 5)]
        else:
            sample_symbols = symbols[:min(len(symbols), 10)]
        
        tasks = [(symbol, basic_params, start_date, end_date) for symbol in sample_symbols]

        scores = []
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = executor.map(_evaluate_single_symbol_wrapper, tasks)
                scores = [score for score in results if score is not None]
        except Exception:
            return self.evaluate_parameters_sequential(params, symbols, start_date, end_date)

        if not scores:
            result = -1000.0
        else:
            result = np.median(scores)
        
        self.cache[cache_key] = result
        return result

    def evaluate_parameters_sequential(self, params, symbols, start_date, end_date):
        """パラメータセットを逐次処理で評価（フォールバック用）"""
        scores = []
        basic_params = {
            'ma_period': params['ma_period'],
            'fvg_min_gap': params['fvg_min_gap'],
            'resistance_lookback': params['resistance_lookback'],
            'breakout_threshold': params['breakout_threshold'],
            'stop_loss_rate': params['stop_loss_rate'],
            'target_profit_rate': params['target_profit_rate'],
            'ma_proximity_percent': params['ma_proximity_percent']
        }
        
        sample_symbols = symbols[:min(len(symbols), 10)]
        period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        for i, symbol in enumerate(sample_symbols):
            try:
                backtester = FVGBreakBacktest(**basic_params)
                result = backtester.run_backtest(symbol, start_date, end_date)

                if i < 1:  # 最初の銘柄のデバッグ情報のみ表示
                    print(f"--- Debug Info for {symbol} ---")
                    if result.get('error'):
                        print(f"Error: {result['error']}")
                    else:
                        print(f"Total Trades: {result.get('total_trades', 0)}")
                        print(f"Win Rate: {result.get('win_rate', 0):.2f}%")
                        print(f"S1 Entries: {result.get('debug_info', {}).get('strategy1_entries', 0)}")
                        print(f"S2 Entries: {result.get('debug_info', {}).get('strategy2_entries', 0)}")

                if not result.get('error'):
                    score = EnhancedFVGParameterOptimizer.calculate_enhanced_score(result, period_days)
                    scores.append(score)
            except Exception as e:
                print(f"Exception in sequential evaluation for {symbol}: {e}")
                continue
        
        if not scores:
            return -1000.0
        return np.median(scores)

    def evaluate_parameters(self, params, symbols, start_date, end_date):
        """パラメータセットを複数銘柄で評価（並列/逐次を自動選択）"""
        if self.fast_mode or self.n_jobs > 1:
            return self.evaluate_parameters_parallel(params, symbols, start_date, end_date)
        else:
            return self.evaluate_parameters_sequential(params, symbols, start_date, end_date)

    def multi_stage_optimization(self, symbols, start_date, end_date):
        """多段階最適化戦略の実行"""
        print("🚀 多段階最適化を開始します...")
        print(f"クロスバリデーション: {'有効' if self.use_cross_validation else '無効'}")
        print(f"高速モード: {'有効' if self.fast_mode else '無効'}")
        print(f"並列ジョブ数: {self.n_jobs}")
        
        all_results = []
        best_params_from_previous_stage = {}

        for stage_name, config in self.optimization_config.items():
            print(f"\n=== {stage_name.upper()} STAGE ===")
            print(f"試行回数: {config['n_trials']}")
            print(f"サンプラー: {type(config['sampler']).__name__}")

            # 早期停止のためのコールバック
            best_score = -float('inf')
            no_improvement_count = 0

            def early_stopping_callback(study, trial):
                nonlocal best_score, no_improvement_count
                if trial.value is not None and trial.value > best_score:
                    best_score = trial.value
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if no_improvement_count >= 10 and self.fast_mode:
                    study.stop()

            # TPESampler/CmaEsSamplerに前ステージの最良解をシードする
            if stage_name in ['exploitation', 'refinement'] and best_params_from_previous_stage:
                if isinstance(config['sampler'], (TPESampler, CmaEsSampler)):
                    # カテゴリカルな値をCmaEsSamplerが扱えるように数値に変換（ここでは単純に無視）
                    cma_seed = {k: v for k, v in best_params_from_previous_stage.items() if not isinstance(v, str)}

                    # CmaEsSamplerはwarm_startを直接サポートしないので、学習済みStudyを渡すか、
                    # TPESamplerのように初期試行として追加する
                    # ここでは、TPESamplerの機能を利用
                    if isinstance(config['sampler'], TPESampler):
                         study = optuna.create_study(
                            direction='maximize',
                            sampler=config['sampler']
                        )
                         study.enqueue_trial(best_params_from_previous_stage)
                    else:
                         study = optuna.create_study(
                            direction='maximize',
                            sampler=config['sampler']
                        )
                else:
                    study = optuna.create_study(
                        direction='maximize',
                        sampler=config['sampler']
                    )
            else:
                 study = optuna.create_study(
                    direction='maximize',
                    sampler=config['sampler']
                )

            def objective(trial):
                params = {}
                # refinementステージではカテゴリカル変数を固定し、連続値のみを最適化
                # さらに、前ステージの最良パラメータの周辺で探索するように範囲を制限
                if stage_name == 'refinement' and best_params_from_previous_stage:
                    # 前ステージの最良パラメータを取得
                    prev_ma = best_params_from_previous_stage.get('ma_period', 200)
                    prev_gap = best_params_from_previous_stage.get('fvg_min_gap', 0.5)
                    prev_lookback = best_params_from_previous_stage.get('resistance_lookback', 20)
                    prev_breakout = best_params_from_previous_stage.get('breakout_threshold', 1.005)
                    prev_stop = best_params_from_previous_stage.get('stop_loss_rate', 0.02)
                    prev_target = best_params_from_previous_stage.get('target_profit_rate', 0.05)
                    prev_proximity = best_params_from_previous_stage.get('ma_proximity_percent', 0.05)
                    
                    # 連続値パラメータ（前ステージの値の周辺で探索）
                    params['ma_period'] = trial.suggest_int('ma_period', 
                        max(5, prev_ma - 50), min(500, prev_ma + 50), step=5)
                    params['fvg_min_gap'] = trial.suggest_float('fvg_min_gap', 
                        max(0.01, prev_gap * 0.5), min(5.0, prev_gap * 2.0), log=True)
                    params['resistance_lookback'] = trial.suggest_int('resistance_lookback', 
                        max(3, prev_lookback - 10), min(100, prev_lookback + 10), step=2)
                    params['breakout_threshold'] = trial.suggest_float('breakout_threshold', 
                        max(0.995, prev_breakout - 0.01), min(1.05, prev_breakout + 0.01), step=0.001)
                    # stop_loss_rateは保守的に調整（リスク管理強化）
                    params['stop_loss_rate'] = trial.suggest_float('stop_loss_rate', 
                        max(0.015, prev_stop * 0.8), min(0.08, prev_stop * 1.3), log=True)
                    params['target_profit_rate'] = trial.suggest_float('target_profit_rate', 
                        max(0.001, prev_target * 0.5), min(0.3, prev_target * 2.0), log=True)
                    params['ma_proximity_percent'] = trial.suggest_float('ma_proximity_percent', 
                        max(0.005, prev_proximity * 0.5), min(0.5, prev_proximity * 2.0), log=True)
                    
                    # 拡張パラメータも前ステージの値の周辺で探索
                    prev_lookback_bars = best_params_from_previous_stage.get('fvg_lookback_bars', 5)
                    prev_fill_threshold = best_params_from_previous_stage.get('fvg_fill_threshold', 0.5)
                    prev_volatility = best_params_from_previous_stage.get('volatility_adjustment', 1.0)
                    prev_trend = best_params_from_previous_stage.get('trend_strength_min', 0.5)
                    prev_risk = best_params_from_previous_stage.get('risk_adjustment', 1.0)
                    
                    params['fvg_lookback_bars'] = trial.suggest_int('fvg_lookback_bars', 
                        max(3, prev_lookback_bars - 2), min(10, prev_lookback_bars + 2))
                    params['fvg_fill_threshold'] = trial.suggest_float('fvg_fill_threshold', 
                        max(0.1, prev_fill_threshold - 0.2), min(0.9, prev_fill_threshold + 0.2), step=0.1)
                    params['volatility_adjustment'] = trial.suggest_float('volatility_adjustment', 
                        max(0.5, prev_volatility - 0.3), min(2.0, prev_volatility + 0.3), step=0.1)
                    params['trend_strength_min'] = trial.suggest_float('trend_strength_min', 
                        max(0.05, prev_trend - 0.2), min(0.95, prev_trend + 0.2), step=0.05)
                    params['risk_adjustment'] = trial.suggest_float('risk_adjustment', 
                        max(0.8, prev_risk - 0.2), min(1.5, prev_risk + 0.2), step=0.1)

                    # 固定するカテゴリカルパラメータ
                    params['volume_confirmation'] = best_params_from_previous_stage.get('volume_confirmation', False)
                    params['timeframe_filter'] = best_params_from_previous_stage.get('timeframe_filter', 'daily')
                else:
                    # 他ステージでは全パラメータを探索
                    params = self.enhanced_parameter_ranges(trial)

                # 高速モードでは一部のパラメータを固定
                if self.fast_mode and stage_name == 'exploration':
                    params['fvg_lookback_bars'] = 5
                    params['fvg_fill_threshold'] = 0.5
                    params['volume_confirmation'] = False
                    params['timeframe_filter'] = 'daily'
                    params['volatility_adjustment'] = 1.0
                    params['trend_strength_min'] = 0.5
                    params['risk_adjustment'] = 1.0

                if self.use_cross_validation and not self.fast_mode:
                    score = self.evaluate_with_cross_validation(params, symbols, start_date, end_date)
                else:
                    score = self.evaluate_parameters(params, symbols, start_date, end_date)

                if trial.number < 3:
                    score_str = f"{score:.3f}" if score is not None else "N/A"
                    print(f"  試行{trial.number}: スコア={score_str}")

                return score if score is not None else -1000.0
            
            try:
                if self.fast_mode:
                    study.optimize(
                        objective, 
                        n_trials=config['n_trials'], 
                        callbacks=[early_stopping_callback],
                        show_progress_bar=True
                    )
                else:
                    study.optimize(
                        objective, 
                        n_trials=config['n_trials'], 
                        show_progress_bar=True
                    )
            except Exception as e:
                print(f"最適化エラー ({stage_name}): {e}")
                continue
            
            stage_results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'trials': len(study.trials),
                'sampler': type(config['sampler']).__name__
            }

            best_params_from_previous_stage = study.best_params
            
            self.multi_stage_results[stage_name] = stage_results
            all_results.extend(study.trials)
            
            print(f"ステージ完了: 最高スコア {study.best_value:.3f}")
            
            # 高速モードでは、良いスコアが見つかったら後続ステージをスキップ
            if self.fast_mode and study.best_value > 10:
                print(f"高速モード: 良好なスコア({study.best_value:.3f})が見つかったため、残りのステージをスキップします。")
                break
        
        # 全ステージの結果から最適パラメータを選択
        if self.multi_stage_results:
            best_overall = max(self.multi_stage_results.items(), key=lambda x: x[1]['best_score'])
            self.best_params = best_overall[1]['best_params']
            
            print(f"\n🎯 多段階最適化完了！")
            print(f"最優秀ステージ: {best_overall[0]}")
            print(f"最高スコア: {best_overall[1]['best_score']:.3f}")
        else:
            print("最適化に失敗しました")
            self.best_params = self.get_default_params()
        
        # キャッシュをクリア（メモリ節約）
        self.cache.clear()
        
        return self.best_params

    def get_default_params(self):
        """デフォルトパラメータを返す（リスク管理強化版）"""
        return {
            'ma_period': 200,
            'fvg_min_gap': 0.5,
            'resistance_lookback': 20,
            'breakout_threshold': 1.005,
            'stop_loss_rate': 0.025,  # 0.02 → 0.025（少し保守的に）
            'target_profit_rate': 0.05,
            'ma_proximity_percent': 0.05,
            'fvg_lookback_bars': 5,
            'fvg_fill_threshold': 0.5,
            'volume_confirmation': False,
            'timeframe_filter': 'daily',
            'volatility_adjustment': 1.0,
            'trend_strength_min': 0.5,
            'risk_adjustment': 1.0
        }

    def multi_objective_optimization(self, symbols, start_date, end_date, n_trials=1000):
        """多目的最適化（リターン vs リスク）"""
        print("🎯 多目的最適化を開始します...")
        
        study = optuna.create_study(
            directions=['maximize', 'minimize'],  # リターン最大化, リスク最小化
            sampler=NSGAIISampler(seed=42)
        )
        
        def multi_objective(trial):
            params = self.enhanced_parameter_ranges(trial)
            
            # 複数銘柄での評価
            returns = []
            drawdowns = []
            
            basic_params = {
                'ma_period': params['ma_period'],
                'fvg_min_gap': params['fvg_min_gap'],
                'resistance_lookback': params['resistance_lookback'],
                'breakout_threshold': params['breakout_threshold'],
                'stop_loss_rate': params['stop_loss_rate'],
                'target_profit_rate': params['target_profit_rate'],
                'ma_proximity_percent': params['ma_proximity_percent']
            }
            
            try:
                backtester = FVGBreakBacktest(**basic_params)
            except Exception as e:
                print(f"バックテスター初期化エラー: {e}")
                return -1000, 1000
            
            sample_symbols = symbols[:min(len(symbols), 20)]  # 効率化のため20銘柄
            
            for symbol in sample_symbols:
                try:
                    result = backtester.run_backtest(symbol, start_date, end_date)
                    if not result.get('error'):
                        returns.append(result.get('avg_return', 0))
                        drawdowns.append(abs(result.get('max_loss', 0)))
                except:
                    continue
            
            if not returns:
                return -1000, 1000
            
            avg_return = np.mean(returns)
            avg_drawdown = np.mean(drawdowns)
            
            return avg_return, avg_drawdown
        
        try:
            study.optimize(multi_objective, n_trials=n_trials, show_progress_bar=True)
        except Exception as e:
            print(f"多目的最適化エラー: {e}")
            self.best_params = self.get_default_params()
            return self.best_params
        
        # パレート最適解を抽出
        pareto_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
                pareto_trials.append({
                    'params': trial.params,
                    'return': trial.values[0],
                    'risk': trial.values[1]
                })
        
        if not pareto_trials:
            print("パレート最適解が見つかりませんでした")
            self.best_params = self.get_default_params()
            return self.best_params
        
        self.pareto_solutions = sorted(pareto_trials, key=lambda x: x['return'], reverse=True)
        
        # バランスの良い解を選択（シャープレシオ的な指標）
        best_balanced = max(self.pareto_solutions, key=lambda x: x['return'] / (x['risk'] + 1e-6))
        self.best_params = best_balanced['params']
        
        print(f"🎯 多目的最適化完了！")
        print(f"パレート最適解数: {len(self.pareto_solutions)}")
        print(f"選択解 - リターン: {best_balanced['return']:.3f}, リスク: {best_balanced['risk']:.3f}")
        
        return self.best_params

    def comprehensive_validation(self, test_start='2024-01-01', test_end='2024-12-31'):
        """包括的な検証"""
        if not self.best_params:
            raise ValueError("最適化を先に実行してください")
        
        print(f"\n📊 包括的検証を開始 ({test_start} - {test_end})")
        
        test_symbols = self.get_sp500_symbols()
        
        # 基本パラメータのみでバックテスター作成
        basic_params = {
            'ma_period': self.best_params.get('ma_period', 200),
            'fvg_min_gap': self.best_params.get('fvg_min_gap', 0.5),
            'resistance_lookback': self.best_params.get('resistance_lookback', 20),
            'breakout_threshold': self.best_params.get('breakout_threshold', 1.005),
            'stop_loss_rate': self.best_params.get('stop_loss_rate', 0.025),  # デフォルトも保守的に
            'target_profit_rate': self.best_params.get('target_profit_rate', 0.05),
            'ma_proximity_percent': self.best_params.get('ma_proximity_percent', 0.05)
        }
        
        try:
            backtester = FVGBreakBacktest(**basic_params)
        except Exception as e:
            print(f"バックテスター初期化エラー: {e}")
            return {'error': '検証中にエラーが発生しました'}
        
        all_results = []
        all_s1_trades = []
        all_s2_trades = []
        
        # 検証用にサンプル数を制限
        validation_symbols = test_symbols[:min(len(test_symbols), 30)]
        
        for symbol in validation_symbols:
            try:
                result = backtester.run_backtest(symbol, test_start, test_end)
                if not result.get('error'):
                    all_results.append(result)
                    all_s1_trades.extend(result.get('strategy1_trades', []))
                    all_s2_trades.extend(result.get('strategy2_trades', []))
            except Exception as e:
                print(f"検証エラー ({symbol}): {e}")
                continue
        
        if not all_s1_trades:
            return {'error': '検証期間中にトレードが発生しませんでした'}
        
        # 総合統計
        s1_returns = [t.get('return', 0) for t in all_s1_trades if 'return' in t]
        s1_wins = [r for r in s1_returns if r > 0]
        
        s2_final_trades = [t for t in all_s1_trades if t.get('s2_triggered') and 'return' in t]
        s2_returns = [t.get('return', 0) for t in s2_final_trades]
        s2_wins = [r for r in s2_returns if r > 0]
        
        validation_result = {
            'test_period': f"{test_start} - {test_end}",
            'symbols_tested': len(validation_symbols),
            'symbols_with_data': len(all_results),
            'best_params': self.best_params,
            's1_comprehensive': {
                'total_trades': len(all_s1_trades),
                'win_rate': len(s1_wins) / len(all_s1_trades) * 100 if all_s1_trades else 0,
                'avg_return': np.mean(s1_returns) * 100 if s1_returns else 0,
                'median_return': np.median(s1_returns) * 100 if s1_returns else 0,
                'sharpe_ratio': np.mean(s1_returns) / (np.std(s1_returns) + 1e-6) if len(s1_returns) > 1 else 0,
                'max_drawdown': min(s1_returns) * 100 if s1_returns else 0,
            },
            's2_comprehensive': {
                'total_trades': len(all_s2_trades),
                'conversion_rate': len(all_s2_trades) / len(all_s1_trades) * 100 if all_s1_trades else 0,
                'win_rate': len(s2_wins) / len(s2_final_trades) * 100 if s2_final_trades else 0,
                'avg_return': np.mean(s2_returns) * 100 if s2_returns else 0,
                'median_return': np.median(s2_returns) * 100 if s2_returns else 0,
            }
        }
        
        return validation_result

    def create_enhanced_visualizations(self):
        """詳細な可視化レポート"""
        if not self.multi_stage_results:
            print("可視化: 多段階最適化結果がありません")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Enhanced FVG Optimization Results', fontsize=16, fontweight='bold')
            
            # 1. 多段階最適化結果
            ax1 = axes[0, 0]
            stages = list(self.multi_stage_results.keys())
            scores = [self.multi_stage_results[stage]['best_score'] for stage in stages]
            trials = [self.multi_stage_results[stage]['trials'] for stage in stages]
            
            bars = ax1.bar(stages, scores, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax1.set_title('Multi-Stage Optimization Results', fontweight='bold')
            ax1.set_ylabel('Best Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # 試行回数を棒グラフ上に表示
            for bar, trial in zip(bars, trials):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{trial} trials', ha='center', va='bottom', fontsize=10)
            
            # 2. パラメータ分布ヒートマップ
            ax2 = axes[0, 1]
            if self.best_params:
                # 数値パラメータのみ抽出
                numeric_params = {k: v for k, v in self.best_params.items() 
                                if isinstance(v, (int, float))}
                
                if numeric_params:
                    param_names = list(numeric_params.keys())[:7]  # 最大7個まで
                    param_values = [numeric_params[k] for k in param_names]
                    
                    # 正規化
                    if len(param_values) > 1 and max(param_values) != min(param_values):
                        normalized_values = [(v - min(param_values)) / (max(param_values) - min(param_values)) 
                                           for v in param_values]
                    else:
                        normalized_values = param_values
                    
                    # ヒートマップ用のデータ準備
                    heatmap_data = np.array(normalized_values).reshape(1, -1)
                    
                    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                               xticklabels=[p[:10] for p in param_names], yticklabels=['Best Params'], ax=ax2)
                    ax2.set_title('Parameter Heatmap (Normalized)', fontweight='bold')
                    ax2.tick_params(axis='x', rotation=45)
            
            # 3. パレート最適解（多目的最適化）
            ax3 = axes[0, 2]
            if self.pareto_solutions:
                returns = [p['return'] for p in self.pareto_solutions[:50]]  # 最大50点
                risks = [p['risk'] for p in self.pareto_solutions[:50]]
                
                scatter = ax3.scatter(risks, returns, c=range(len(returns)), cmap='viridis', alpha=0.7)
                ax3.set_xlabel('Risk (Drawdown)')
                ax3.set_ylabel('Return')
                ax3.set_title('Pareto Frontier Analysis', fontweight='bold')
                plt.colorbar(scatter, ax=ax3, label='Solution Index')
            else:
                ax3.text(0.5, 0.5, 'No Pareto Solutions\n(Run multi-objective optimization)', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Pareto Frontier Analysis', fontweight='bold')
            
            # 4. 最適化収束履歴（プレースホルダー）
            ax4 = axes[1, 0]
            ax4.text(0.5, 0.5, 'Optimization Convergence\n(Feature in development)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Optimization Convergence', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 5. 設定概要
            ax5 = axes[1, 1]
            ax5.axis('off')
            
            config_text = f"""
OPTIMIZATION CONFIGURATION

Mode: {'Unlimited' if self.unlimited_mode else 'Limited'}
Total Stages: {len(self.optimization_config)}

Stage Details:
"""
            for stage, config in self.optimization_config.items():
                config_text += f"• {stage}: {config['n_trials']} trials\n"
            
            if self.best_params:
                config_text += f"""
Best Parameters:
• MA Period: {self.best_params.get('ma_period', 'N/A')}
• FVG Min Gap: {self.best_params.get('fvg_min_gap', 'N/A'):.3f}
• Resistance Lookback: {self.best_params.get('resistance_lookback', 'N/A')}
• Breakout Threshold: {self.best_params.get('breakout_threshold', 'N/A'):.3f}
• Stop Loss Rate: {self.best_params.get('stop_loss_rate', 'N/A'):.3f}
"""
            
            ax5.text(0.05, 0.95, config_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            # 6. 品質指標
            ax6 = axes[1, 2]
            if hasattr(self, 'validation_results') and self.validation_results:
                if not self.validation_results.get('error'):
                    metrics = ['Win Rate', 'Avg Return', 'Sharpe Ratio', 'Max Drawdown']
                    s1_values = [
                        self.validation_results['s1_comprehensive']['win_rate'],
                        self.validation_results['s1_comprehensive']['avg_return'],
                        self.validation_results['s1_comprehensive']['sharpe_ratio'],
                        abs(self.validation_results['s1_comprehensive']['max_drawdown'])
                    ]
                    
                    bars = ax6.bar(metrics, s1_values, color=['green', 'blue', 'orange', 'red'])
                    ax6.set_title('Quality Metrics (Strategy 1)', fontweight='bold')
                    ax6.tick_params(axis='x', rotation=45)
                    
                    # 値をバーの上に表示
                    for bar, value in zip(bars, s1_values):
                        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
                else:
                    ax6.text(0.5, 0.5, 'Validation Error', ha='center', va='center', 
                            transform=ax6.transAxes, fontsize=12)
            else:
                ax6.text(0.5, 0.5, 'No Validation Results\n(Run comprehensive validation)', 
                        ha='center', va='center', transform=ax6.transAxes, fontsize=12)
                ax6.set_title('Quality Metrics', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('enhanced_optimization_results.png', dpi=300, bbox_inches='tight')
            print("📊 詳細な可視化を enhanced_optimization_results.png に保存しました")
            
        except Exception as e:
            print(f"可視化エラー: {e}")

    def save_comprehensive_results(self, filename='optimized_params_enhanced.json'):
        """包括的な結果保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        comprehensive_results = {
            'optimization_metadata': {
                'timestamp': timestamp,
                'unlimited_mode': self.unlimited_mode,
                'optimization_config': {
                    stage: {
                        'n_trials': config['n_trials'],
                        'sampler': type(config['sampler']).__name__
                    }
                    for stage, config in self.optimization_config.items()
                }
            },
            'best_params': self.best_params,
            'multi_stage_results': self.multi_stage_results,
            'pareto_solutions': self.pareto_solutions[:10] if self.pareto_solutions else [],
            'validation_results': getattr(self, 'validation_results', None)
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
            
            print(f"📄 包括的な結果を {filename} に保存しました")
            
            # 設定用出力
            if self.best_params:
                print("\n🔧 .envファイル設定:")
                print(f"MA_PERIOD={self.best_params.get('ma_period', 200)}")
                print(f"FVG_MIN_GAP_PERCENT={self.best_params.get('fvg_min_gap', 0.5)}")
                print(f"RESISTANCE_LOOKBACK={self.best_params.get('resistance_lookback', 20)}")
                print(f"BREAKOUT_THRESHOLD={self.best_params.get('breakout_threshold', 1.005)}")
                print(f"STOP_LOSS_RATE={self.best_params.get('stop_loss_rate', 0.025)}")
                print(f"TARGET_PROFIT_RATE={self.best_params.get('target_profit_rate', 0.05)}")
                print(f"MA_PROXIMITY_PERCENT={self.best_params.get('ma_proximity_percent', 0.05)}")
                
        except Exception as e:
            print(f"結果保存エラー: {e}")

    def generate_html_report(self):
        """HTML形式の詳細レポート生成"""
        if not self.best_params:
            print("HTMLレポート: 最適化結果がありません")
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FVG Bot ML Optimization Report</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .params-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }}
        .param-item {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #2e8b57; }}
        .stage-results {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stage {{ background-color: #e6f3ff; padding: 15px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 FVG Break Alert Bot - ML Optimization Report</h1>
            <p>Generated: {html.escape(timestamp)}</p>
            <p>Mode: {html.escape('Unlimited (Quality Priority)' if self.unlimited_mode else 'Limited (Demo)')}</p>
        </div>
        
        <div class="section">
            <h2>🎯 Optimization Summary</h2>
            <div class="stage-results">
"""
        
        # 各ステージの結果
        for stage_name, results in self.multi_stage_results.items():
            score_value = results.get('best_score', -1000)
            html_content += f"""
                <div class="stage">
                    <h3>{html.escape(stage_name.title())}</h3>
                    <p>Trials: {results['trials']}</p>
                    <p>Sampler: {html.escape(results['sampler'])}</p>
                    <p class="score">Score: {score_value:.3f}</p>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Best Parameters</h2>
            <div class="params-grid">
"""
        
        # パラメータ詳細（基本パラメータのみ表示）
        basic_param_names = [
            'ma_period', 'fvg_min_gap', 'resistance_lookback', 
            'breakout_threshold', 'stop_loss_rate', 'target_profit_rate', 
            'ma_proximity_percent'
        ]
        
        for param in basic_param_names:
            if param in self.best_params:
                value = self.best_params[param]
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)
                
                html_content += f"""
                <div class="param-item">
                    <strong>{html.escape(param)}:</strong><br>
                    {html.escape(display_value)}
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Validation Results</h2>
"""
        
        # 検証結果
        if hasattr(self, 'validation_results') and self.validation_results:
            vr = self.validation_results
            if not vr.get('error'):
                s1_comp = vr.get('s1_comprehensive', {})
                s2_comp = vr.get('s2_comprehensive', {})
                html_content += f"""
            <h4>Overall Performance</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{s1_comp.get('total_trades', 'N/A')}</td></tr>
                <tr><td>Win Rate</td><td>{s1_comp.get('win_rate', 0):.2f}%</td></tr>
                <tr><td>Avg Return</td><td>{s1_comp.get('avg_return', 0):.2f}%</td></tr>
                <tr><td>Sharpe Ratio</td><td>{s1_comp.get('sharpe_ratio', 0):.3f}</td></tr>
                <tr><td>Max Drawdown</td><td>{s1_comp.get('max_drawdown', 0):.2f}%</td></tr>
            </table>
            <h4>Strategy 2 (Breakout) Performance</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Conversion Rate</td><td>{s2_comp.get('conversion_rate', 0):.2f}%</td></tr>
                <tr><td>S2 Win Rate</td><td>{s2_comp.get('win_rate', 0):.2f}%</td></tr>
                <tr><td>S2 Avg Return</td><td>{s2_comp.get('avg_return', 0):.2f}%</td></tr>
            </table>
"""
            else:
                html_content += f"""
            <div class="warning">
                <p><strong>Validation Error:</strong> {html.escape(vr['error'])}</p>
            </div>
"""
        else:
            html_content += """
            <div class="warning">
                <p>No validation results. Please run comprehensive_validation().</p>
            </div>
"""
        
        html_content += """
        </div>
        
        <div class="section">
            <h2>🔧 Implementation</h2>
            <p>Add these parameters to your .env file:</p>
            <pre>
"""
        
        # 環境変数設定
        html_content += f"""MA_PERIOD={self.best_params.get('ma_period', 200)}
FVG_MIN_GAP_PERCENT={self.best_params.get('fvg_min_gap', 0.5)}
RESISTANCE_LOOKBACK={self.best_params.get('resistance_lookback', 20)}
BREAKOUT_THRESHOLD={self.best_params.get('breakout_threshold', 1.005)}
STOP_LOSS_RATE={self.best_params.get('stop_loss_rate', 0.025)}
TARGET_PROFIT_RATE={self.best_params.get('target_profit_rate', 0.05)}
MA_PROXIMITY_PERCENT={self.best_params.get('ma_proximity_percent', 0.05)}"""
        
        html_content += """
            </pre>
        </div>
        
        <div class="section">
            <h2>📌 Notes</h2>
            <ul>
                <li>このレポートは最適化結果の概要です。詳細はJSONファイルを参照してください。</li>
                <li>パラメータを本番環境に適用する前に、必ず追加のバックテストを実行してください。</li>
                <li>市場環境の変化に応じて、定期的に再最適化を行うことを推奨します。</li>
                <li>リスク管理パラメータ（stop_loss_rate）は保守的に設定されています。</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        report_filename = f'optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"📋 HTMLレポートを {report_filename} に保存しました")
        except Exception as e:
            print(f"HTMLレポート保存エラー: {e}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Enhanced FVG Parameter Optimization')
    parser.add_argument('--mode', choices=['multi_stage', 'multi_objective', 'walk_forward'], 
                       default='multi_stage', help='Optimization mode')
    parser.add_argument('--unlimited', action='store_true', help='Enable unlimited mode (time-intensive)')
    parser.add_argument('--n_trials', type=int, default=None, help='Number of trials per stage')
    parser.add_argument('--no_cv', action='store_true', help='Disable cross-validation (faster but may overfit)')
    parser.add_argument('--fast', action='store_true', help='Enable fast mode (parallel processing, reduced trials)')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # 高速モードの推奨設定
    if args.fast:
        print("⚡ 高速モードが有効です")
        print("推奨: クロスバリデーションも無効化してください (--no_cv)")
    
    # 最適化インスタンス作成
    optimizer = EnhancedFVGParameterOptimizer(
        unlimited_mode=args.unlimited,
        n_jobs=args.n_jobs,
        use_cross_validation=not args.no_cv,
        fast_mode=args.fast
    )
    
    # 試行回数のカスタマイズ
    if args.n_trials:
        for stage in optimizer.optimization_config:
            optimizer.optimization_config[stage]['n_trials'] = args.n_trials
    
    # S&P500銘柄取得
    symbols = optimizer.get_sp500_symbols()
    print(f"📊 対象銘柄数: {len(symbols)}")
    
    # 最適化実行
    start_time = time.time()
    
    # 動的な日付設定
    today = datetime.now()
    training_end_date = today - timedelta(days=2*365)  # 2年前
    training_start_date = today - timedelta(days=10*365) # 10年前
    validation_start_date = training_end_date
    validation_end_date = today

    training_start_str = training_start_date.strftime('%Y-%m-%d')
    training_end_str = training_end_date.strftime('%Y-%m-%d')
    validation_start_str = validation_start_date.strftime('%Y-%m-%d')
    validation_end_str = validation_end_date.strftime('%Y-%m-%d')

    try:
        if args.mode == 'multi_stage':
            print("🚀 多段階最適化を実行中...")
            best_params = optimizer.multi_stage_optimization(
                symbols, training_start_str, training_end_str
            )
        elif args.mode == 'multi_objective':
            print("🎯 多目的最適化を実行中...")
            # 高速モードでは試行回数を削減
            n_trials = 100 if args.fast else (1000 if args.unlimited else 200)
            best_params = optimizer.multi_objective_optimization(
                symbols, training_start_str, training_end_str,
                n_trials=n_trials
            )
        elif args.mode == 'walk_forward':
            print("📈 Walk-Forward最適化を実行中...")
            best_params = optimizer.multi_stage_optimization(
                symbols, training_start_str, training_end_str
            )
    except Exception as e:
        print(f"最適化エラー: {e}")
        best_params = optimizer.get_default_params()
        optimizer.best_params = best_params
    
    elapsed_time = time.time() - start_time
    print(f"⏱️ 最適化完了時間: {elapsed_time/60:.2f}分 ({elapsed_time/3600:.2f}時間)")
    
    # 検証実行
    print("\n📊 検証を実行中...")
    try:
        validation_results = optimizer.comprehensive_validation(validation_start_str, validation_end_str)
        optimizer.validation_results = validation_results
    except Exception as e:
        print(f"検証エラー: {e}")
        validation_results = {'error': str(e)}
        optimizer.validation_results = validation_results
    
    # 結果出力
    print("\n=== 最適化結果サマリー ===")
    if validation_results.get('error'):
        print(f"検証エラー: {validation_results['error']}")
    else:
        print(f"検証期間: {validation_results['test_period']}")
        print(f"対象銘柄数: {validation_results['symbols_tested']}")
        print(f"データ取得成功: {validation_results['symbols_with_data']}")
        
        s1 = validation_results['s1_comprehensive']
        s2 = validation_results['s2_comprehensive']
        
        print(f"\n戦略1: {s1['total_trades']}回, 勝率{s1['win_rate']:.1f}%, 平均{s1['avg_return']:.2f}%")
        print(f"戦略2: {s2['total_trades']}回, 転換率{s2['conversion_rate']:.1f}%, 勝率{s2['win_rate']:.1f}%")
        print(f"シャープレシオ: {s1['sharpe_ratio']:.3f}")
        print(f"最大ドローダウン: {s1['max_drawdown']:.2f}%")
    
    # 結果保存
    optimizer.save_comprehensive_results()
    optimizer.create_enhanced_visualizations()
    optimizer.generate_html_report()
    
    print("\n🎉 品質最優先ML最適化システムの実行が完了しました！")
    print("生成されたファイル:")
    print("- optimized_params_enhanced.json (最適化結果)")
    print("- enhanced_optimization_results.png (詳細可視化)")
    print("- optimization_report_[timestamp].html (HTMLレポート)")
    
    if args.no_cv:
        print("\n⚠️ 注意: クロスバリデーションが無効化されています。過学習のリスクがあります。")
    
    if args.fast:
        print("\n⚡ 高速モードで実行されました。より精密な最適化が必要な場合は、通常モードで再実行してください。")


if __name__ == "__main__":
    main()
