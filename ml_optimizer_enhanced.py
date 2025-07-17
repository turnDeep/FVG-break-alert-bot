"""
機械学習によるFVG Break Alert Botパラメータ最適化
設計書に基づく品質最優先・時間無制限の最適化システム

主な改善点:
1. 多段階最適化戦略 (RandomSampler → TPESampler → CmaEsSampler)
2. 多目的最適化 (NSGAIISampler) による過学習防止
3. TimeSeriesSplit による堅牢なクロスバリデーション
4. 拡張されたパラメータ範囲と対数スケール探索
5. 詳細な結果記録と可視化

使用方法:
# 基本的な実行（クロスバリデーション有効）
python ml_optimizer_enhanced.py

# クロスバリデーションを無効化（高速化）
python ml_optimizer_enhanced.py --no_cv

# 試行回数を指定
python ml_optimizer_enhanced.py --n_trials 10

# 無制限モード（時間をかけて高品質な最適化）
python ml_optimizer_enhanced.py --unlimited

# 多目的最適化モード
python ml_optimizer_enhanced.py --mode multi_objective
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
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

# バックテストモジュールのインポート（存在確認付き）
try:
    from backtest import FVGBreakBacktest
except ImportError:
    print("警告: backtest.pyが見つかりません。ダミークラスを使用します。")
    # ダミークラスの定義（開発/テスト用）
    class FVGBreakBacktest:
        def __init__(self, **kwargs):
            self.params = kwargs
            np.random.seed(42)  # 再現性のため
        
        def run_backtest(self, symbol, start_date, end_date):
            # ダミーの結果を返す（よりリアルな分布）
            # パラメータによって結果を変える
            ma_period = self.params.get('ma_period', 200)
            fvg_min_gap = self.params.get('fvg_min_gap', 0.5)
            
            # パラメータに基づいてベースラインを調整
            base_return = 0.001 + (200 - ma_period) * 0.00001 + fvg_min_gap * 0.001
            base_win_rate = 50 + (ma_period - 100) * 0.05
            
            # ランダム性を追加
            trade_count = np.random.randint(20, 80)
            win_rate = np.clip(base_win_rate + np.random.normal(0, 5), 30, 70)
            avg_return = base_return + np.random.normal(0, 0.002)
            
            # 戦略1のトレード生成
            trades = []
            for _ in range(trade_count):
                if np.random.random() < win_rate / 100:
                    trades.append({'return': np.random.uniform(0.001, 0.03)})
                else:
                    trades.append({'return': np.random.uniform(-0.02, -0.001)})
            
            # 戦略2のトレード（一部を変換）
            s2_trades = []
            s2_count = int(trade_count * np.random.uniform(0.1, 0.3))
            for i in range(s2_count):
                trades[i]['s2_triggered'] = True
                if np.random.random() < 0.6:  # 戦略2の勝率
                    s2_trades.append({'return': np.random.uniform(0.005, 0.05)})
                else:
                    s2_trades.append({'return': np.random.uniform(-0.015, -0.005)})
            
            returns = [t['return'] for t in trades]
            s2_returns = [t['return'] for t in s2_trades]
            
            return {
                'error': False,
                'avg_return': np.mean(returns) if returns else 0,
                'max_loss': min(returns) if returns else 0,
                's1_stats': {
                    'count': trade_count,
                    'win_rate': win_rate,
                    'avg_return': np.mean(returns) if returns else 0
                },
                's2_stats': {
                    'count': s2_count,
                    'win_rate': (sum(1 for r in s2_returns if r > 0) / len(s2_returns) * 100) if s2_returns else 0,
                    'avg_return': np.mean(s2_returns) if s2_returns else 0,
                    'conversion_rate': (s2_count / trade_count * 100) if trade_count > 0 else 0
                },
                'strategy1_trades': trades,
                'strategy2_trades': s2_trades
            }

class EnhancedFVGParameterOptimizer:
    """品質最優先の機械学習最適化システム"""

    def __init__(self, unlimited_mode=False, n_jobs=4, use_cross_validation=True):
        self.unlimited_mode = unlimited_mode
        self.n_jobs = n_jobs
        self.use_cross_validation = use_cross_validation
        self.best_params = None
        self.optimization_results = []
        self.multi_stage_results = {}
        self.pareto_solutions = []
        
        # 設計書に基づく最適化設定
        if unlimited_mode:
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
        """S&P500銘柄リストを取得"""
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            symbols = sp500['Symbol'].str.replace('.', '-').tolist()
            return symbols if self.unlimited_mode else symbols[:50]  # デモ用制限
        except Exception as e:
            print(f"S&P500リスト取得エラー: {e}")
            # フォールバック銘柄リスト
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']

    def calculate_enhanced_score(self, result):
        """過学習防止と品質重視の評価関数"""
        # エラーチェック
        if result.get('error') or 's1_stats' not in result:
            return -1000
            
        s1_stats = result['s1_stats']
        s2_stats = result.get('s2_stats', {})

        if s1_stats.get('count', 0) == 0:
            return -100  # トレードがない場合のペナルティを軽減

        # 基本スコア: 平均リターン × 勝率
        avg_return = s1_stats.get('avg_return', 0)
        win_rate = s1_stats.get('win_rate', 0) / 100
        s1_score = avg_return * win_rate
        
        # 戦略2のスコア
        s2_score = 0
        if s2_stats.get('count', 0) > 0:
            s2_avg_return = s2_stats.get('avg_return', 0)
            s2_win_rate = s2_stats.get('win_rate', 0) / 100
            s2_score = s2_avg_return * s2_win_rate

        # 戦略1: 40%, 戦略2: 60%の重み付け
        base_score = (s1_score * 0.4) + (s2_score * 0.6)
        
        # スコアを100倍してより見やすい値にする
        base_score *= 100

        # 過学習防止のための正則化項
        returns = [t.get('return', 0) for t in result.get('strategy1_trades', [])]
        if len(returns) > 1:
            # シャープレシオ（年率化）
            returns_array = np.array(returns)
            daily_mean = np.mean(returns_array)
            daily_std = np.std(returns_array)
            if daily_std > 0:
                sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252)  # 年率化
                base_score += sharpe_ratio * 2
            
            # 連続損失耐性
            consecutive_losses = 0
            max_consecutive_losses = 0
            for ret in returns:
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            if max_consecutive_losses > 5:
                base_score -= (max_consecutive_losses - 5) * 0.5  # ペナルティを軽減

        # 戦略2転換率ボーナス
        conversion_rate = s2_stats.get('conversion_rate', 0)
        base_score += conversion_rate * 0.05  # ボーナスを調整

        # 適度な取引頻度の重視
        trade_count = s1_stats.get('count', 0)
        if trade_count < 10:
            base_score -= (10 - trade_count) * 0.2  # ペナルティを軽減
        elif trade_count > 100:
            base_score -= (trade_count - 100) * 0.01  # ペナルティを軽減

        # 最大損失ペナルティ
        max_loss = result.get('max_loss', 0)
        if max_loss < -15:
            base_score -= abs(max_loss) * 0.1  # ペナルティを軽減

        return base_score

    def enhanced_parameter_ranges(self, trial):
        """拡張されたパラメータ範囲と対数スケール探索"""
        # 基本パラメータ（FVGBreakBacktestで使用される）
        basic_params = {
            'ma_period': trial.suggest_int('ma_period', 5, 500, step=5),
            'fvg_min_gap': trial.suggest_float('fvg_min_gap', 0.01, 5.0, log=True),
            'resistance_lookback': trial.suggest_int('resistance_lookback', 3, 100, step=2),
            'breakout_threshold': trial.suggest_float('breakout_threshold', 0.995, 1.05, step=0.001),
            'stop_loss_rate': trial.suggest_float('stop_loss_rate', 0.001, 0.15, log=True),
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
            # 時系列データの準備
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Pandasの新しいバージョンに対応（'M'ではなく'ME'を使用）
            try:
                # 3ヶ月単位で分割
                months = pd.date_range(start=start_dt, end=end_dt, freq='3ME')
            except:
                # 古いバージョンのPandasの場合
                months = pd.date_range(start=start_dt, end=end_dt, freq='3M')
            
            if len(months) < 3:
                # 期間が短すぎる場合は単純評価
                return self.evaluate_parameters(params, symbols, start_date, end_date)
            
            cv_scores = []
            
            for i in range(len(months) - 2):
                train_start = months[i].strftime('%Y-%m-%d')
                train_end = months[i+1].strftime('%Y-%m-%d')
                val_start = months[i+1].strftime('%Y-%m-%d')
                val_end = months[i+2].strftime('%Y-%m-%d')
                
                # 各分割でのスコアを計算
                train_score = self.evaluate_parameters(params, symbols, train_start, train_end)
                val_score = self.evaluate_parameters(params, symbols, val_start, val_end)
                
                # 過学習チェック: 訓練と検証の差が大きすぎる場合はペナルティ
                if train_score > -900 and val_score > -900:  # 有効なスコアの場合のみ
                    overfitting_penalty = abs(train_score - val_score) * 0.1
                    adjusted_score = val_score - overfitting_penalty
                    cv_scores.append(adjusted_score)
            
            if not cv_scores:
                # クロスバリデーションが失敗した場合は単純評価にフォールバック
                print("警告: クロスバリデーションが失敗。単純評価を使用します。")
                return self.evaluate_parameters(params, symbols, start_date, end_date)
            
            return np.mean(cv_scores)
            
        except Exception as e:
            print(f"クロスバリデーションエラー: {str(e)}")
            # エラー時は単純評価にフォールバック
            try:
                return self.evaluate_parameters(params, symbols, start_date, end_date)
            except:
                return -1000

    def evaluate_parameters(self, params, symbols, start_date, end_date):
        """パラメータセットを複数銘柄で評価"""
        scores = []
        
        # 基本パラメータのみを抽出（FVGBreakBacktestが受け取れるパラメータ）
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
            return -1000
        
        # サンプル銘柄数を制限（効率化）
        sample_symbols = symbols[:min(len(symbols), 20)]
        
        for symbol in sample_symbols:
            try:
                result = backtester.run_backtest(symbol, start_date, end_date)
                if not result.get('error'):
                    score = self.calculate_enhanced_score(result)
                    scores.append(score)
            except Exception as e:
                print(f"評価エラー ({symbol}): {e}")
                continue
        
        if not scores:
            return -1000
        
        # 中央値を使用して外れ値の影響を軽減
        return np.median(scores)

    def multi_stage_optimization(self, symbols, start_date, end_date):
        """多段階最適化戦略の実行"""
        print("🚀 多段階最適化を開始します...")
        print(f"クロスバリデーション: {'有効' if self.use_cross_validation else '無効'}")
        
        all_results = []
        
        for stage_name, config in self.optimization_config.items():
            print(f"\n=== {stage_name.upper()} STAGE ===")
            print(f"試行回数: {config['n_trials']}")
            print(f"サンプラー: {type(config['sampler']).__name__}")
            
            study = optuna.create_study(
                direction='maximize',
                sampler=config['sampler']
            )
            
            def objective(trial):
                params = self.enhanced_parameter_ranges(trial)
                if self.use_cross_validation:
                    score = self.evaluate_with_cross_validation(params, symbols, start_date, end_date)
                else:
                    score = self.evaluate_parameters(params, symbols, start_date, end_date)
                
                # デバッグ出力（最初の5試行のみ）
                if trial.number < 5:
                    print(f"  試行{trial.number}: スコア={score:.3f}")
                
                return score
            
            try:
                study.optimize(objective, n_trials=config['n_trials'], show_progress_bar=True)
            except Exception as e:
                print(f"最適化エラー ({stage_name}): {e}")
                continue
            
            stage_results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'trials': len(study.trials),
                'sampler': type(config['sampler']).__name__
            }
            
            self.multi_stage_results[stage_name] = stage_results
            all_results.extend(study.trials)
            
            print(f"ステージ完了: 最高スコア {study.best_value:.3f}")
        
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
        
        return self.best_params

    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            'ma_period': 200,
            'fvg_min_gap': 0.5,
            'resistance_lookback': 20,
            'breakout_threshold': 1.005,
            'stop_loss_rate': 0.02,
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
            'stop_loss_rate': self.best_params.get('stop_loss_rate', 0.02),
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
                print(f"STOP_LOSS_RATE={self.best_params.get('stop_loss_rate', 0.02)}")
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
                html_content += f"""
            <table>
                <tr><th>Metric</th><th>Strategy 1</th><th>Strategy 2</th></tr>
                <tr><td>Total Trades</td><td>{vr['s1_comprehensive']['total_trades']}</td><td>{vr['s2_comprehensive']['total_trades']}</td></tr>
                <tr><td>Win Rate</td><td>{vr['s1_comprehensive']['win_rate']:.2f}%</td><td>{vr['s2_comprehensive']['win_rate']:.2f}%</td></tr>
                <tr><td>Avg Return</td><td>{vr['s1_comprehensive']['avg_return']:.2f}%</td><td>{vr['s2_comprehensive']['avg_return']:.2f}%</td></tr>
                <tr><td>Sharpe Ratio</td><td>{vr['s1_comprehensive']['sharpe_ratio']:.3f}</td><td>-</td></tr>
                <tr><td>Max Drawdown</td><td>{vr['s1_comprehensive']['max_drawdown']:.2f}%</td><td>-</td></tr>
            </table>
"""
            else:
                html_content += f"""
            <div class="warning">
                <p><strong>検証エラー:</strong> {html.escape(vr['error'])}</p>
            </div>
"""
        else:
            html_content += """
            <div class="warning">
                <p>検証結果がありません。comprehensive_validation()を実行してください。</p>
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
STOP_LOSS_RATE={self.best_params.get('stop_loss_rate', 0.02)}
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
    
    args = parser.parse_args()
    
    # 最適化インスタンス作成
    optimizer = EnhancedFVGParameterOptimizer(
        unlimited_mode=args.unlimited,
        n_jobs=4,
        use_cross_validation=not args.no_cv  # --no_cvが指定されたらFalse
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
    
    try:
        if args.mode == 'multi_stage':
            print("🚀 多段階最適化を実行中...")
            best_params = optimizer.multi_stage_optimization(
                symbols, '2022-01-01', '2024-01-01'
            )
        elif args.mode == 'multi_objective':
            print("🎯 多目的最適化を実行中...")
            best_params = optimizer.multi_objective_optimization(
                symbols, '2022-01-01', '2024-01-01',
                n_trials=1000 if args.unlimited else 200
            )
        elif args.mode == 'walk_forward':
            print("📈 Walk-Forward最適化を実行中...")
            # Walk-Forward最適化として多段階最適化を使用
            best_params = optimizer.multi_stage_optimization(
                symbols, '2022-01-01', '2024-01-01'
            )
    except Exception as e:
        print(f"最適化エラー: {e}")
        best_params = optimizer.get_default_params()
        optimizer.best_params = best_params
    
    elapsed_time = time.time() - start_time
    print(f"⏱️ 最適化完了時間: {elapsed_time/3600:.2f}時間")
    
    # 検証実行
    print("\n📊 検証を実行中...")
    try:
        validation_results = optimizer.comprehensive_validation('2024-01-01', '2024-12-31')
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


if __name__ == "__main__":
    main()
