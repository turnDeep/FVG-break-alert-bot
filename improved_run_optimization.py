"""
改善版 機械学習によるFVG Break Alert Botパラメータ最適化
早期収束を防ぎ、より良い最適解を探索する
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import random
warnings.filterwarnings('ignore')

from backtest import FVGBreakBacktest


class ImprovedFVGOptimizer:
    """改善版FVGパラメータ最適化クラス"""
    
    def __init__(self, n_trials=200, n_jobs=4, exploration_ratio=0.3):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.exploration_ratio = exploration_ratio  # 探索に使う試行の割合
        self.best_params = None
        self.optimization_results = []
        self.param_history = []  # パラメータの履歴を保存
        
    def get_sp500_symbols(self):
        """S&P500銘柄リストを取得"""
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            symbols = sp500['Symbol'].str.replace('.', '-').tolist()
            return symbols
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']
    
    def calculate_multi_objective_score(self, result: Dict) -> Tuple[float, Dict]:
        """多目的スコア計算（複数の評価軸を考慮）"""
        s1_stats = result['s1_stats']
        s2_stats = result['s2_stats']
        
        # 戦略1が発生しない場合
        if s1_stats['count'] == 0:
            return -1000, {'reason': 'no_trades'}
        
        # 複数の評価指標を計算
        metrics = {
            # 収益性指標
            's1_return': s1_stats['avg_return'],
            's2_return': s2_stats['avg_return'] if s2_stats['count'] > 0 else 0,
            's1_win_rate': s1_stats['win_rate'] / 100,
            's2_win_rate': s2_stats['win_rate'] / 100 if s2_stats['count'] > 0 else 0,
            
            # リスク指標
            'max_drawdown': abs(result.get('max_loss', 0)),
            'risk_reward_ratio': abs(s1_stats['avg_return'] / result.get('max_loss', -1)) if result.get('max_loss', 0) < 0 else 1,
            
            # 頻度・転換指標
            'trade_frequency': s1_stats['count'],
            'conversion_rate': s2_stats['conversion_rate'] / 100,
            
            # 安定性指標（仮想的に計算）
            'consistency': 1 - (abs(s1_stats['win_rate'] - 50) / 50)  # 50%に近いほど安定
        }
        
        # 動的な重み付け（試行回数に応じて変化）
        trial_progress = len(self.optimization_results) / self.n_trials
        
        if trial_progress < 0.3:  # 初期段階：多様性重視
            weights = {
                's1_return': 0.15,
                's2_return': 0.15,
                's1_win_rate': 0.1,
                's2_win_rate': 0.1,
                'max_drawdown': -0.1,
                'risk_reward_ratio': 0.1,
                'trade_frequency': 0.2,  # 頻度を重視
                'conversion_rate': 0.1,
                'consistency': 0.1
            }
        elif trial_progress < 0.7:  # 中期段階：バランス重視
            weights = {
                's1_return': 0.2,
                's2_return': 0.25,
                's1_win_rate': 0.15,
                's2_win_rate': 0.15,
                'max_drawdown': -0.15,
                'risk_reward_ratio': 0.15,
                'trade_frequency': 0.05,
                'conversion_rate': 0.15,
                'consistency': 0.05
            }
        else:  # 後期段階：収益性重視
            weights = {
                's1_return': 0.25,
                's2_return': 0.35,
                's1_win_rate': 0.1,
                's2_win_rate': 0.1,
                'max_drawdown': -0.2,
                'risk_reward_ratio': 0.2,
                'trade_frequency': 0.0,
                'conversion_rate': 0.1,
                'consistency': 0.0
            }
        
        # スコア計算
        score = sum(metrics.get(key, 0) * weight for key, weight in weights.items())
        
        # ペナルティとボーナス
        if s1_stats['count'] < 5:
            score *= 0.5
        if s2_stats['count'] > 10:
            score *= 1.2
        if metrics['max_drawdown'] > 20:
            score *= 0.3
        
        # 多様性ボーナス（既存パラメータとの距離）
        if self.param_history:
            diversity_bonus = self._calculate_diversity_bonus(result.get('params', {}))
            score += diversity_bonus
        
        return score, metrics
    
    def _calculate_diversity_bonus(self, params: Dict) -> float:
        """パラメータの多様性ボーナスを計算"""
        if not self.param_history:
            return 0
        
        # 最近のパラメータとの距離を計算
        recent_params = self.param_history[-20:]  # 最近20個
        
        distances = []
        for past_params in recent_params:
            distance = 0
            for key in params:
                if key in past_params:
                    # 正規化された距離
                    if isinstance(params[key], (int, float)):
                        param_range = self._get_param_range(key)
                        normalized_dist = abs(params[key] - past_params[key]) / (param_range[1] - param_range[0])
                        distance += normalized_dist
            distances.append(distance)
        
        # 平均距離が大きいほどボーナス
        avg_distance = np.mean(distances) if distances else 0
        return avg_distance * 5  # 最大5ポイントのボーナス
    
    def _get_param_range(self, param_name: str) -> Tuple[float, float]:
        """パラメータの範囲を返す"""
        ranges = {
            'ma_period': (20, 200),
            'fvg_min_gap': (0.1, 1.0),
            'resistance_lookback': (5, 30),
            'breakout_threshold': (1.0, 1.01),
            'stop_loss_rate': (0.01, 0.05),
            'target_profit_rate': (0.01, 0.1),
            'ma_proximity_percent': (0.05, 0.2)
        }
        return ranges.get(param_name, (0, 1))
    
    def create_multi_stage_study(self, stage: int) -> optuna.Study:
        """段階に応じた最適化スタディを作成"""
        
        if stage == 1:  # 粗い探索
            sampler = RandomSampler(seed=42)  # 完全ランダム
            pruner = None  # 枝刈りなし
        elif stage == 2:  # 中間探索
            sampler = TPESampler(
                n_startup_trials=50,  # より多くのランダム試行
                n_ei_candidates=50,   # より多くの候補
                gamma=lambda x: int(0.25 * x),  # より探索的
                seed=42
            )
            pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=5)
        else:  # 精密探索
            sampler = CmaEsSampler(
                n_startup_trials=30,
                seed=42
            )
            pruner = HyperbandPruner()
        
        return optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
    
    def adaptive_objective(self, trial, train_symbols, val_symbols, start_date, end_date):
        """適応的な目的関数（探索範囲が動的に変化）"""
        
        # 現在の最良パラメータ周辺を探索するか、広範囲を探索するか
        explore_widely = (
            len(self.optimization_results) < self.n_trials * self.exploration_ratio or
            random.random() < 0.2  # 20%の確率で常に広範囲探索
        )
        
        if explore_widely or not self.best_params:
            # 広範囲探索
            params = {
                'ma_period': trial.suggest_int('ma_period', 20, 200, step=10),
                'fvg_min_gap': trial.suggest_float('fvg_min_gap', 0.1, 1.0, step=0.05),
                'resistance_lookback': trial.suggest_int('resistance_lookback', 5, 30, step=5),
                'breakout_threshold': trial.suggest_float('breakout_threshold', 1.0, 1.01, step=0.001),
                'stop_loss_rate': trial.suggest_float('stop_loss_rate', 0.01, 0.05, step=0.005),
                'target_profit_rate': trial.suggest_float('target_profit_rate', 0.01, 0.1, step=0.01),
                'ma_proximity_percent': trial.suggest_float('ma_proximity_percent', 0.05, 0.2, step=0.05)
            }
        else:
            # 最良パラメータ周辺の探索
            best = self.best_params
            params = {
                'ma_period': trial.suggest_int('ma_period', 
                    max(20, best['ma_period'] - 30), 
                    min(200, best['ma_period'] + 30), 
                    step=10),
                'fvg_min_gap': trial.suggest_float('fvg_min_gap', 
                    max(0.1, best['fvg_min_gap'] - 0.2), 
                    min(1.0, best['fvg_min_gap'] + 0.2), 
                    step=0.05),
                'resistance_lookback': trial.suggest_int('resistance_lookback', 
                    max(5, best['resistance_lookback'] - 10), 
                    min(30, best['resistance_lookback'] + 10), 
                    step=5),
                'breakout_threshold': trial.suggest_float('breakout_threshold', 
                    max(1.0, best['breakout_threshold'] - 0.003), 
                    min(1.01, best['breakout_threshold'] + 0.003), 
                    step=0.001),
                'stop_loss_rate': trial.suggest_float('stop_loss_rate', 
                    max(0.01, best['stop_loss_rate'] - 0.01), 
                    min(0.05, best['stop_loss_rate'] + 0.01), 
                    step=0.005),
                'target_profit_rate': trial.suggest_float('target_profit_rate', 
                    max(0.01, best['target_profit_rate'] - 0.02), 
                    min(0.1, best['target_profit_rate'] + 0.02), 
                    step=0.01),
                'ma_proximity_percent': trial.suggest_float('ma_proximity_percent', 
                    max(0.05, best['ma_proximity_percent'] - 0.05), 
                    min(0.2, best['ma_proximity_percent'] + 0.05), 
                    step=0.05)
            }
        
        # パラメータ履歴に追加
        self.param_history.append(params)
        
        # 評価実行
        train_results = self._evaluate_with_sampling(params, train_symbols, start_date, end_date)
        val_results = self._evaluate_with_sampling(params, val_symbols, start_date, end_date)
        
        # スコア計算
        train_score, train_metrics = self.calculate_multi_objective_score(train_results)
        val_score, val_metrics = self.calculate_multi_objective_score(val_results)
        
        # 結果を保存
        combined_score = train_score * 0.6 + val_score * 0.4
        
        self.optimization_results.append({
            'params': params,
            'train_score': train_score,
            'val_score': val_score,
            'combined_score': combined_score,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'exploration_type': 'wide' if explore_widely else 'local'
        })
        
        # 最良パラメータを更新
        if combined_score > getattr(self, '_best_score', -float('inf')):
            self._best_score = combined_score
            self.best_params = params.copy()
        
        return combined_score
    
    def _evaluate_with_sampling(self, params, symbols, start_date, end_date):
        """サンプリングを使った効率的な評価"""
        # 銘柄数が多い場合はサンプリング
        if len(symbols) > 30:
            sampled_symbols = np.random.choice(symbols, min(30, len(symbols)), replace=False)
        else:
            sampled_symbols = symbols
        
        backtester = FVGBreakBacktest(**params)
        
        aggregated_results = {
            's1_stats': {'count': 0, 'win_rate': 0, 'avg_return': 0},
            's2_stats': {'count': 0, 'win_rate': 0, 'avg_return': 0, 'conversion_rate': 0},
            'total_trades': 0,
            'max_loss': 0,
            'avg_return': 0,
            'params': params
        }
        
        valid_results = []
        
        for symbol in sampled_symbols:
            try:
                result = backtester.run_backtest(symbol, start_date, end_date)
                if not result.get('error'):
                    valid_results.append(result)
            except:
                continue
        
        if not valid_results:
            return aggregated_results
        
        # 結果を集計
        total_s1_trades = sum(r['s1_stats']['count'] for r in valid_results)
        total_s2_trades = sum(r['s2_stats']['count'] for r in valid_results)
        
        if total_s1_trades > 0:
            aggregated_results['s1_stats']['count'] = total_s1_trades
            aggregated_results['s1_stats']['win_rate'] = np.mean([r['s1_stats']['win_rate'] for r in valid_results if r['s1_stats']['count'] > 0])
            aggregated_results['s1_stats']['avg_return'] = np.mean([r['s1_stats']['avg_return'] for r in valid_results if r['s1_stats']['count'] > 0])
        
        if total_s2_trades > 0:
            aggregated_results['s2_stats']['count'] = total_s2_trades
            aggregated_results['s2_stats']['win_rate'] = np.mean([r['s2_stats']['win_rate'] for r in valid_results if r['s2_stats']['count'] > 0])
            aggregated_results['s2_stats']['avg_return'] = np.mean([r['s2_stats']['avg_return'] for r in valid_results if r['s2_stats']['count'] > 0])
            aggregated_results['s2_stats']['conversion_rate'] = (total_s2_trades / total_s1_trades * 100) if total_s1_trades > 0 else 0
        
        aggregated_results['total_trades'] = total_s1_trades
        aggregated_results['max_loss'] = min(r['max_loss'] for r in valid_results)
        aggregated_results['avg_return'] = aggregated_results['s1_stats']['avg_return']
        
        return aggregated_results
    
    def optimize_multi_stage(self, start_date='2022-01-01', end_date='2024-01-01'):
        """多段階最適化の実行"""
        print("S&P500銘柄リストを取得中...")
        all_symbols = self.get_sp500_symbols()
        print(f"取得銘柄数: {len(all_symbols)}")
        
        # 銘柄を分割
        np.random.shuffle(all_symbols)
        split_idx = int(len(all_symbols) * 0.8)
        train_symbols = all_symbols[:split_idx]
        val_symbols = all_symbols[split_idx:]
        
        print(f"訓練銘柄数: {len(train_symbols)}, 検証銘柄数: {len(val_symbols)}")
        
        # 3段階の最適化
        stages = [
            {'name': '探索段階', 'trials': int(self.n_trials * 0.4)},
            {'name': '改善段階', 'trials': int(self.n_trials * 0.4)},
            {'name': '収束段階', 'trials': int(self.n_trials * 0.2)}
        ]
        
        all_results = []
        
        for stage_idx, stage_info in enumerate(stages, 1):
            print(f"\n=== {stage_info['name']} ({stage_info['trials']}試行) ===")
            
            study = self.create_multi_stage_study(stage_idx)
            
            def wrapped_objective(trial):
                return self.adaptive_objective(trial, train_symbols, val_symbols, start_date, end_date)
            
            study.optimize(
                wrapped_objective,
                n_trials=stage_info['trials'],
                n_jobs=1
            )
            
            # 段階の最良結果を記録
            stage_best = study.best_params
            stage_best_score = study.best_value
            
            print(f"{stage_info['name']}の最高スコア: {stage_best_score:.2f}")
            
            all_results.extend(self.optimization_results[-stage_info['trials']:])
        
        print(f"\n最終的な最適パラメータ: {self.best_params}")
        return self.best_params
    
    def ensemble_optimize(self, start_date='2022-01-01', end_date='2024-01-01', n_optimizers=3):
        """アンサンブル最適化（複数の最適化を並列実行）"""
        print("アンサンブル最適化を開始...")
        
        all_symbols = self.get_sp500_symbols()
        ensemble_results = []
        
        for i in range(n_optimizers):
            print(f"\nOptimizer {i+1}/{n_optimizers}")
            
            # 各最適化で異なる銘柄分割
            np.random.shuffle(all_symbols)
            split_idx = int(len(all_symbols) * 0.8)
            train_symbols = all_symbols[:split_idx]
            val_symbols = all_symbols[split_idx:]
            
            # 異なるシードで最適化
            optimizer = ImprovedFVGOptimizer(
                n_trials=self.n_trials // n_optimizers,
                n_jobs=self.n_jobs
            )
            
            best_params = optimizer.optimize_multi_stage(start_date, end_date)
            ensemble_results.append({
                'params': best_params,
                'score': optimizer._best_score,
                'results': optimizer.optimization_results
            })
        
        # アンサンブル結果から最良を選択
        best_ensemble = max(ensemble_results, key=lambda x: x['score'])
        self.best_params = best_ensemble['params']
        
        # 全結果を統合
        for result in ensemble_results:
            self.optimization_results.extend(result['results'])
        
        return self.best_params
    
    def plot_enhanced_results(self, save_path='enhanced_optimization_results.png'):
        """拡張された結果の可視化"""
        if not self.optimization_results:
            print("最適化結果がありません")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Optimization Analysis', fontsize=16)
        
        # 1. スコアの推移（探索タイプ別）
        ax = axes[0, 0]
        wide_scores = [(i, r['combined_score']) for i, r in enumerate(self.optimization_results) if r.get('exploration_type') == 'wide']
        local_scores = [(i, r['combined_score']) for i, r in enumerate(self.optimization_results) if r.get('exploration_type') == 'local']
        
        if wide_scores:
            ax.scatter(*zip(*wide_scores), alpha=0.6, label='Wide exploration', color='blue')
        if local_scores:
            ax.scatter(*zip(*local_scores), alpha=0.6, label='Local exploration', color='red')
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Score')
        ax.set_title('Score Progress by Exploration Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. パラメータ分布（ヒートマップ）
        ax = axes[0, 1]
        top_results = sorted(self.optimization_results, key=lambda x: x['combined_score'], reverse=True)[:30]
        param_matrix = []
        param_names = ['ma_period', 'fvg_min_gap', 'resistance_lookback', 'breakout_threshold']
        
        for r in top_results:
            normalized_params = []
            for pname in param_names:
                prange = self._get_param_range(pname)
                normalized = (r['params'][pname] - prange[0]) / (prange[1] - prange[0])
                normalized_params.append(normalized)
            param_matrix.append(normalized_params)
        
        im = ax.imshow(np.array(param_matrix).T, aspect='auto', cmap='viridis')
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names)
        ax.set_xlabel('Top 30 Results')
        ax.set_title('Parameter Distribution Heatmap')
        plt.colorbar(im, ax=ax)
        
        # 3. 収束曲線
        ax = axes[0, 2]
        scores = [r['combined_score'] for r in self.optimization_results]
        best_scores = np.maximum.accumulate(scores)
        ax.plot(best_scores, 'g-', linewidth=2, label='Best score')
        ax.plot(scores, 'b-', alpha=0.3, label='Current score')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Best Score')
        ax.set_title('Convergence Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. メトリクス相関
        ax = axes[1, 0]
        if any('train_metrics' in r for r in self.optimization_results[-50:]):
            returns = []
            win_rates = []
            for r in self.optimization_results[-50:]:
                if 'train_metrics' in r:
                    returns.append(r['train_metrics'].get('s2_return', 0))
                    win_rates.append(r['train_metrics'].get('s2_win_rate', 0))
            
            if returns and win_rates:
                ax.scatter(returns, win_rates, alpha=0.6)
                ax.set_xlabel('S2 Return (%)')
                ax.set_ylabel('S2 Win Rate')
                ax.set_title('Return vs Win Rate Correlation')
                ax.grid(True, alpha=0.3)
        
        # 5. パラメータ重要度（簡易版）
        ax = axes[1, 1]
        param_impacts = {}
        for pname in ['ma_period', 'fvg_min_gap', 'resistance_lookback']:
            values = [r['params'][pname] for r in self.optimization_results]
            scores = [r['combined_score'] for r in self.optimization_results]
            if len(set(values)) > 1:
                correlation = np.corrcoef(values, scores)[0, 1]
                param_impacts[pname] = abs(correlation)
        
        if param_impacts:
            ax.bar(param_impacts.keys(), param_impacts.values())
            ax.set_ylabel('Importance (|correlation|)')
            ax.set_title('Parameter Importance')
            ax.tick_params(axis='x', rotation=45)
        
        # 6. 探索の多様性
        ax = axes[1, 2]
        diversity_scores = []
        window_size = 20
        
        for i in range(window_size, len(self.param_history)):
            recent = self.param_history[i-window_size:i]
            # パラメータの標準偏差を計算
            param_stds = []
            for pname in ['ma_period', 'fvg_min_gap']:
                values = [p[pname] for p in recent]
                prange = self._get_param_range(pname)
                normalized_std = np.std(values) / (prange[1] - prange[0])
                param_stds.append(normalized_std)
            diversity_scores.append(np.mean(param_stds))
        
        if diversity_scores:
            ax.plot(range(window_size, len(self.param_history)), diversity_scores)
            ax.set_xlabel('Trial')
            ax.set_ylabel('Parameter Diversity')
            ax.set_title('Exploration Diversity Over Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"拡張された最適化結果を{save_path}に保存しました")
    
    def save_detailed_results(self, filename='enhanced_optimization_results.json'):
        """詳細な結果を保存"""
        # 上位10%の結果を詳細に分析
        top_10_percent = int(len(self.optimization_results) * 0.1)
        top_results = sorted(self.optimization_results, key=lambda x: x['combined_score'], reverse=True)[:top_10_percent]
        
        # パラメータの統計
        param_stats = {}
        for pname in self.best_params.keys():
            values = [r['params'][pname] for r in top_results]
            param_stats[pname] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values),
                'best': self.best_params[pname]
            }
        
        results = {
            'best_params': self.best_params,
            'best_score': getattr(self, '_best_score', None),
            'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trials': len(self.optimization_results),
            'param_statistics': param_stats,
            'top_results': top_results[:20],
            'exploration_summary': {
                'wide_exploration_count': sum(1 for r in self.optimization_results if r.get('exploration_type') == 'wide'),
                'local_exploration_count': sum(1 for r in self.optimization_results if r.get('exploration_type') == 'local'),
                'average_score': np.mean([r['combined_score'] for r in self.optimization_results]),
                'score_std': np.std([r['combined_score'] for r in self.optimization_results])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n詳細な結果を{filename}に保存しました")
        
        # 推奨パラメータ範囲も出力
        print("\n推奨パラメータ範囲（上位10%の統計）:")
        for pname, stats in param_stats.items():
            print(f"{pname}: {stats['mean']:.3f} ± {stats['std']:.3f} (最良: {stats['best']})")


def main():
    """メイン実行関数"""
    # 改善版オプティマイザーを作成
    optimizer = ImprovedFVGOptimizer(
        n_trials=100,  # 総試行回数
        n_jobs=4,
        exploration_ratio=0.3  # 30%は探索に使用
    )
    
    # アンサンブル最適化を実行
    best_params = optimizer.ensemble_optimize(
        start_date='2022-01-01',
        end_date='2024-01-01',
        n_optimizers=3  # 3つの最適化を並列実行
    )
    
    # 詳細な結果を保存
    optimizer.save_detailed_results()
    
    # 拡張された可視化
    optimizer.plot_enhanced_results()
    
    print("\n最適化完了！")
    print("生成されたファイル:")
    print("- enhanced_optimization_results.json (詳細な結果)")
    print("- enhanced_optimization_results.png (拡張された可視化)")


if __name__ == "__main__":
    main()
