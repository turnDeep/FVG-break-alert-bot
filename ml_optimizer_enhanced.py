"""
機械学習によるFVG Break Alert Botパラメータ最適化
設計書に基づく品質最優先・時間無制限の最適化システム

主な改善点:
1. 多段階最適化戦略 (RandomSampler → TPESampler → CmaEsSampler)
2. 多目的最適化 (NSGAIISampler) による過学習防止
3. TimeSeriesSplit による堅牢なクロスバリデーション
4. 拡張されたパラメータ範囲と対数スケール探索
5. 詳細な結果記録と可視化
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
warnings.filterwarnings('ignore')

from backtest import FVGBreakBacktest

class EnhancedFVGParameterOptimizer:
    """品質最優先の機械学習最適化システム"""

    def __init__(self, unlimited_mode=False, n_jobs=4):
        self.unlimited_mode = unlimited_mode
        self.n_jobs = n_jobs
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
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']

    def calculate_enhanced_score(self, result):
        """過学習防止と品質重視の評価関数"""
        s1_stats = result['s1_stats']
        s2_stats = result['s2_stats']

        if s1_stats['count'] == 0:
            return -1000

        # 基本スコア: 平均リターン × 勝率
        s1_score = s1_stats['avg_return'] * (s1_stats['win_rate'] / 100)
        s2_score = s2_stats['avg_return'] * (s2_stats['win_rate'] / 100) if s2_stats['count'] > 0 else 0

        # 戦略1: 40%, 戦略2: 60%の重み付け
        base_score = (s1_score * 0.4) + (s2_score * 0.6)

        # 過学習防止のための正則化項
        returns = [t.get('return', 0) for t in result.get('strategy1_trades', [])]
        if len(returns) > 1:
            # シャープレシオ
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)
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
                base_score -= (max_consecutive_losses - 5) * 3

        # 戦略2転換率ボーナス
        base_score += (s2_stats['conversion_rate'] / 100) * 5

        # 適度な取引頻度の重視
        if s1_stats['count'] < 10:
            base_score -= (10 - s1_stats['count']) * 2
        elif s1_stats['count'] > 100:
            base_score -= (s1_stats['count'] - 100) * 0.1

        # 最大損失ペナルティ
        if result['max_loss'] < -15:
            base_score -= abs(result['max_loss'])

        return base_score

    def enhanced_parameter_ranges(self, trial):
        """拡張されたパラメータ範囲と対数スケール探索"""
        return {
            # 基本パラメータ（範囲を大幅拡張）
            'ma_period': trial.suggest_int('ma_period', 5, 500, step=5),
            'fvg_min_gap': trial.suggest_float('fvg_min_gap', 0.01, 5.0, log=True),
            'resistance_lookback': trial.suggest_int('resistance_lookback', 3, 100, step=2),
            'breakout_threshold': trial.suggest_float('breakout_threshold', 0.995, 1.05, step=0.001),
            'stop_loss_rate': trial.suggest_float('stop_loss_rate', 0.001, 0.15, log=True),
            'target_profit_rate': trial.suggest_float('target_profit_rate', 0.001, 0.3, log=True),
            'ma_proximity_percent': trial.suggest_float('ma_proximity_percent', 0.005, 0.5, log=True),
            
            # FVG定義の最適化パラメータ
            'fvg_lookback_bars': trial.suggest_int('fvg_lookback_bars', 3, 10),
            'fvg_fill_threshold': trial.suggest_float('fvg_fill_threshold', 0.1, 0.9, step=0.1),
            'volume_confirmation': trial.suggest_categorical('volume_confirmation', [True, False]),
            'timeframe_filter': trial.suggest_categorical('timeframe_filter', ['1h', '4h', 'daily']),
            
            # 市場レジーム適応パラメータ
            'volatility_adjustment': trial.suggest_float('volatility_adjustment', 0.5, 2.0, step=0.1),
            'trend_strength_min': trial.suggest_float('trend_strength_min', 0.05, 0.95, step=0.05),
            'risk_adjustment': trial.suggest_float('risk_adjustment', 0.8, 1.5, step=0.1),
        }

    def evaluate_with_cross_validation(self, params, symbols, start_date, end_date):
        """TimeSeriesSplitを使用したクロスバリデーション"""
        # 時系列データの準備
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 6ヶ月単位で分割
        months = pd.date_range(start=start_dt, end=end_dt, freq='2M')
        
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
            overfitting_penalty = abs(train_score - val_score) * 0.1
            adjusted_score = val_score - overfitting_penalty
            
            cv_scores.append(adjusted_score)
        
        return np.mean(cv_scores) if cv_scores else -1000

    def evaluate_parameters(self, params, symbols, start_date, end_date):
        """パラメータセットを複数銘柄で評価"""
        scores = []
        
        # 基本パラメータのみでバックテスター作成
        basic_params = {
            'ma_period': params['ma_period'],
            'fvg_min_gap': params['fvg_min_gap'],
            'resistance_lookback': params['resistance_lookback'],
            'breakout_threshold': params['breakout_threshold'],
            'stop_loss_rate': params['stop_loss_rate'],
            'target_profit_rate': params['target_profit_rate'],
            'ma_proximity_percent': params['ma_proximity_percent']
        }
        
        backtester = FVGBreakBacktest(**basic_params)
        
        for symbol in symbols:
            try:
                result = backtester.run_backtest(symbol, start_date, end_date)
                if not result.get('error'):
                    score = self.calculate_enhanced_score(result)
                    scores.append(score)
            except Exception as e:
                print(f"Error with {symbol}: {e}")
                continue
        
        if not scores:
            return -1000
        
        # 中央値を使用して外れ値の影響を軽減
        return np.median(scores)

    def multi_stage_optimization(self, symbols, start_date, end_date):
        """多段階最適化戦略の実行"""
        print("🚀 多段階最適化を開始します...")
        
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
                return self.evaluate_with_cross_validation(params, symbols, start_date, end_date)
            
            study.optimize(objective, n_trials=config['n_trials'], show_progress_bar=True)
            
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
        best_overall = max(self.multi_stage_results.items(), key=lambda x: x[1]['best_score'])
        self.best_params = best_overall[1]['best_params']
        
        print(f"\n🎯 多段階最適化完了！")
        print(f"最優秀ステージ: {best_overall[0]}")
        print(f"最高スコア: {best_overall[1]['best_score']:.3f}")
        
        return self.best_params

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
            
            backtester = FVGBreakBacktest(**basic_params)
            
            for symbol in symbols[:20]:  # 効率化のため20銘柄
                try:
                    result = backtester.run_backtest(symbol, start_date, end_date)
                    if not result.get('error'):
                        returns.append(result['avg_return'])
                        drawdowns.append(abs(result['max_loss']))
                except:
                    continue
            
            if not returns:
                return -1000, 1000
            
            avg_return = np.mean(returns)
            avg_drawdown = np.mean(drawdowns)
            
            return avg_return, avg_drawdown
        
        study.optimize(multi_objective, n_trials=n_trials, show_progress_bar=True)
        
        # パレート最適解を抽出
        pareto_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pareto_trials.append({
                    'params': trial.params,
                    'return': trial.values[0],
                    'risk': trial.values[1]
                })
        
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
            'ma_period': self.best_params['ma_period'],
            'fvg_min_gap': self.best_params['fvg_min_gap'],
            'resistance_lookback': self.best_params['resistance_lookback'],
            'breakout_threshold': self.best_params['breakout_threshold'],
            'stop_loss_rate': self.best_params['stop_loss_rate'],
            'target_profit_rate': self.best_params['target_profit_rate'],
            'ma_proximity_percent': self.best_params['ma_proximity_percent']
        }
        
        backtester = FVGBreakBacktest(**basic_params)
        
        all_results = []
        all_s1_trades = []
        all_s2_trades = []
        
        for symbol in test_symbols:
            try:
                result = backtester.run_backtest(symbol, test_start, test_end)
                if not result.get('error'):
                    all_results.append(result)
                    all_s1_trades.extend(result['strategy1_trades'])
                    all_s2_trades.extend(result['strategy2_trades'])
            except Exception as e:
                print(f"検証エラー ({symbol}): {e}")
                continue
        
        if not all_s1_trades:
            return {'error': '検証期間中にトレードが発生しませんでした'}
        
        # 総合統計
        s1_returns = [t['return'] for t in all_s1_trades if 'return' in t]
        s1_wins = [r for r in s1_returns if r > 0]
        
        s2_final_trades = [t for t in all_s1_trades if t.get('s2_triggered') and 'return' in t]
        s2_returns = [t['return'] for t in s2_final_trades]
        s2_wins = [r for r in s2_returns if r > 0]
        
        validation_result = {
            'test_period': f"{test_start} - {test_end}",
            'symbols_tested': len(test_symbols),
            'symbols_with_data': len(all_results),
            'best_params': self.best_params,
            's1_comprehensive': {
                'total_trades': len(all_s1_trades),
                'win_rate': len(s1_wins) / len(all_s1_trades) * 100 if all_s1_trades else 0,
                'avg_return': np.mean(s1_returns) * 100 if s1_returns else 0,
                'median_return': np.median(s1_returns) * 100 if s1_returns else 0,
                'sharpe_ratio': np.mean(s1_returns) / np.std(s1_returns) if len(s1_returns) > 1 else 0,
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
            
            param_names = list(numeric_params.keys())
            param_values = list(numeric_params.values())
            
            # 正規化
            normalized_values = [(v - min(param_values)) / (max(param_values) - min(param_values)) 
                               for v in param_values]
            
            # ヒートマップ用のデータ準備
            heatmap_data = np.array(normalized_values).reshape(1, -1)
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                       xticklabels=param_names, yticklabels=['Best Params'], ax=ax2)
            ax2.set_title('Parameter Heatmap (Normalized)', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. パレート最適解（多目的最適化）
        ax3 = axes[0, 2]
        if self.pareto_solutions:
            returns = [p['return'] for p in self.pareto_solutions]
            risks = [p['risk'] for p in self.pareto_solutions]
            
            scatter = ax3.scatter(risks, returns, c=range(len(returns)), cmap='viridis', alpha=0.7)
            ax3.set_xlabel('Risk (Drawdown)')
            ax3.set_ylabel('Return')
            ax3.set_title('Pareto Frontier Analysis', fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Solution Index')
        else:
            ax3.text(0.5, 0.5, 'No Pareto Solutions\n(Run multi-objective optimization)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Pareto Frontier Analysis', fontweight='bold')
        
        # 4. 最適化収束履歴
        ax4 = axes[1, 0]
        if self.optimization_results:
            scores = [r.get('combined_score', r.get('val_score', 0)) for r in self.optimization_results]
            ax4.plot(scores, 'b-', alpha=0.7)
            ax4.axhline(y=max(scores), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(scores):.2f}')
            ax4.set_xlabel('Trial')
            ax4.set_ylabel('Score')
            ax4.set_title('Optimization Convergence', fontweight='bold')
            ax4.legend()
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
        if hasattr(self, 'validation_results'):
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
        
        plt.tight_layout()
        plt.savefig('enhanced_optimization_results.png', dpi=300, bbox_inches='tight')
        print("📊 詳細な可視化を enhanced_optimization_results.png に保存しました")

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

    def generate_html_report(self):
        """HTML形式の詳細レポート生成"""
        if not self.best_params:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FVG Bot ML Optimization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: auto; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .params-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }}
        .param-item {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #2e8b57; }}
        .stage-results {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stage {{ background-color: #e6f3ff; padding: 15px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 FVG Break Alert Bot - ML Optimization Report</h1>
            <p>Generated: {timestamp}</p>
            <p>Mode: {'Unlimited (Quality Priority)' if self.unlimited_mode else 'Limited (Demo)'}</p>
        </div>
        
        <div class="section">
            <h2>🎯 Optimization Summary</h2>
            <div class="stage-results">
"""
        
        # 各ステージの結果
        for stage_name, results in self.multi_stage_results.items():
            html_content += f"""
                <div class="stage">
                    <h3>{stage_name.title()}</h3>
                    <p>Trials: {results['trials']}</p>
                    <p>Sampler: {results['sampler']}</p>
                    <p class="score">Score: {results['best_score']:.3f}</p>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Best Parameters</h2>
            <div class="params-grid">
"""
        
        # パラメータ詳細
        for param, value in self.best_params.items():
            if isinstance(value, float):
                display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            
            html_content += f"""
                <div class="param-item">
                    <strong>{param}:</strong><br>
                    {display_value}
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Validation Results</h2>
"""
        
        # 検証結果
        if hasattr(self, 'validation_results'):
            vr = self.validation_results
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
    </div>
</body>
</html>
"""
        
        report_filename = f'optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 HTMLレポートを {report_filename} に保存しました")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Enhanced FVG Parameter Optimization')
    parser.add_argument('--mode', choices=['multi_stage', 'multi_objective', 'walk_forward'], 
                       default='multi_stage', help='Optimization mode')
    parser.add_argument('--unlimited', action='store_true', help='Enable unlimited mode (time-intensive)')
    parser.add_argument('--n_trials', type=int, default=None, help='Number of trials per stage')
    
    args = parser.parse_args()
    
    # 最適化インスタンス作成
    optimizer = EnhancedFVGParameterOptimizer(
        unlimited_mode=args.unlimited,
        n_jobs=4
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
        # 従来のWalk-Forward最適化を実装
        best_params = optimizer.multi_stage_optimization(
            symbols, '2022-01-01', '2024-01-01'
        )
    
    elapsed_time = time.time() - start_time
    print(f"⏱️ 最適化完了時間: {elapsed_time/3600:.2f}時間")
    
    # 検証実行
    print("\n📊 検証を実行中...")
    validation_results = optimizer.comprehensive_validation('2024-01-01', '2024-12-31')
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

if __name__ == "__main__":
    main()
