#!/usr/bin/env python3
"""
改善版ML Optimizer使用例とベストプラクティス
"""
import os
from datetime import datetime, timedelta
from improved_ml_optimizer import ImprovedFVGOptimizer  # 上記で作成したクラス

def run_standard_optimization():
    """標準的な最適化実行"""
    print("=== 標準的な最適化 ===")
    
    optimizer = ImprovedFVGOptimizer(
        n_trials=200,        # 十分な試行回数
        n_jobs=4,           # 並列数
        exploration_ratio=0.3  # 30%は探索に使用
    )
    
    # 多段階最適化
    best_params = optimizer.optimize_multi_stage(
        start_date='2022-01-01',
        end_date='2024-01-01'
    )
    
    # 結果の保存と可視化
    optimizer.save_detailed_results('standard_optimization.json')
    optimizer.plot_enhanced_results('standard_optimization.png')
    
    return best_params

def run_ensemble_optimization():
    """アンサンブル最適化（推奨）"""
    print("=== アンサンブル最適化（推奨） ===")
    
    optimizer = ImprovedFVGOptimizer(
        n_trials=300,        # 3つに分散されるので多めに
        n_jobs=6,
        exploration_ratio=0.4  # より探索的に
    )
    
    # アンサンブル実行
    best_params = optimizer.ensemble_optimize(
        start_date='2022-01-01',
        end_date='2024-01-01',
        n_optimizers=3  # 3つの独立した最適化
    )
    
    optimizer.save_detailed_results('ensemble_optimization.json')
    optimizer.plot_enhanced_results('ensemble_optimization.png')
    
    return best_params

def run_custom_optimization():
    """カスタマイズされた最適化"""
    print("=== カスタム最適化 ===")
    
    # 1. 短期間で素早く概要を掴む
    quick_optimizer = ImprovedFVGOptimizer(
        n_trials=50,
        n_jobs=8,
        exploration_ratio=0.5  # 半分は探索
    )
    
    # 最近1年のデータで素早く最適化
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    
    quick_params = quick_optimizer.optimize_multi_stage(
        start_date=one_year_ago.strftime('%Y-%m-%d'),
        end_date=today.strftime('%Y-%m-%d')
    )
    
    print(f"クイック最適化完了: {quick_params}")
    
    # 2. 有望なパラメータ周辺を詳細に探索
    detailed_optimizer = ImprovedFVGOptimizer(
        n_trials=100,
        n_jobs=4,
        exploration_ratio=0.2  # より収束的に
    )
    
    # 初期パラメータとして設定
    detailed_optimizer.best_params = quick_params
    detailed_optimizer._best_score = -float('inf')
    
    # 長期間データで詳細最適化
    best_params = detailed_optimizer.optimize_multi_stage(
        start_date='2021-01-01',
        end_date=today.strftime('%Y-%m-%d')
    )
    
    detailed_optimizer.save_detailed_results('custom_optimization.json')
    detailed_optimizer.plot_enhanced_results('custom_optimization.png')
    
    return best_params

def analyze_optimization_results(json_file):
    """最適化結果の分析"""
    import json
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n=== {json_file} の分析 ===")
    print(f"最良スコア: {results.get('best_score', 'N/A')}")
    print(f"総試行回数: {results['total_trials']}")
    
    # 探索タイプの分析
    exploration = results.get('exploration_summary', {})
    if exploration:
        wide = exploration.get('wide_exploration_count', 0)
        local = exploration.get('local_exploration_count', 0)
        total = wide + local
        if total > 0:
            print(f"探索タイプ: 広範囲 {wide/total*100:.1f}%, 局所 {local/total*100:.1f}%")
    
    # パラメータ統計
    print("\n推奨パラメータ範囲:")
    param_stats = results.get('param_statistics', {})
    for param, stats in param_stats.items():
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)
        best = stats.get('best', 0)
        print(f"  {param}: {mean:.3f} ± {std:.3f} (最良: {best})")
    
    # 上位結果の分析
    print("\n上位5つの結果:")
    top_results = results.get('top_results', [])[:5]
    for i, result in enumerate(top_results, 1):
        score = result.get('combined_score', 0)
        s2_rate = result.get('val_metrics', {}).get('conversion_rate', 0)
        print(f"  {i}. スコア: {score:.2f}, S2転換率: {s2_rate:.1f}%")
    
    return results

def compare_optimization_methods():
    """異なる最適化手法の比較"""
    import time
    
    methods = [
        ("標準的な多段階最適化", run_standard_optimization),
        ("アンサンブル最適化", run_ensemble_optimization),
        ("カスタム2段階最適化", run_custom_optimization)
    ]
    
    results_comparison = {}
    
    for method_name, method_func in methods:
        print(f"\n{'='*60}")
        print(f"実行中: {method_name}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            best_params = method_func()
            elapsed_time = time.time() - start_time
            
            results_comparison[method_name] = {
                'params': best_params,
                'time': elapsed_time,
                'success': True
            }
            
            print(f"\n完了！実行時間: {elapsed_time/60:.1f}分")
            
        except Exception as e:
            print(f"エラー: {e}")
            results_comparison[method_name] = {
                'error': str(e),
                'success': False
            }
    
    # 結果の比較
    print("\n" + "="*60)
    print("最適化手法の比較結果")
    print("="*60)
    
    for method_name, result in results_comparison.items():
        print(f"\n【{method_name}】")
        if result['success']:
            print(f"  実行時間: {result['time']/60:.1f}分")
            print(f"  最適パラメータ:")
            for param, value in result['params'].items():
                print(f"    {param}: {value}")
        else:
            print(f"  エラー: {result['error']}")

def production_ready_optimization():
    """本番環境向けの最適化設定"""
    print("=== 本番環境向け最適化 ===")
    
    # 環境変数から設定を読み込み
    n_trials = int(os.getenv('OPTIMIZATION_TRIALS', 500))
    n_jobs = int(os.getenv('OPTIMIZATION_JOBS', 8))
    n_optimizers = int(os.getenv('ENSEMBLE_SIZE', 5))
    
    # データ期間の設定
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3年間
    
    print(f"設定:")
    print(f"  試行回数: {n_trials}")
    print(f"  並列数: {n_jobs}")
    print(f"  アンサンブルサイズ: {n_optimizers}")
    print(f"  データ期間: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    
    # 本番用オプティマイザー
    optimizer = ImprovedFVGOptimizer(
        n_trials=n_trials,
        n_jobs=n_jobs,
        exploration_ratio=0.35
    )
    
    # ロバストなアンサンブル最適化
    best_params = optimizer.ensemble_optimize(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        n_optimizers=n_optimizers
    )
    
    # 詳細な結果を保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    optimizer.save_detailed_results(f'production_optimization_{timestamp}.json')
    optimizer.plot_enhanced_results(f'production_optimization_{timestamp}.png')
    
    # .envファイル用の出力を生成
    print("\n.envファイル用の設定:")
    print("```")
    for param, value in best_params.items():
        env_name = param.upper()
        print(f"{env_name}={value}")
    print("```")
    
    # バックテスト推奨
    print("\n次のステップ:")
    print("1. 上記のパラメータで直近6ヶ月のバックテストを実行")
    print("2. 異なる市場環境（上昇相場、下降相場、レンジ相場）でテスト")
    print("3. リスク管理パラメータの微調整")
    
    return best_params

if __name__ == "__main__":
    # 使用例を選択
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "standard":
            run_standard_optimization()
        elif mode == "ensemble":
            run_ensemble_optimization()
        elif mode == "custom":
            run_custom_optimization()
        elif mode == "compare":
            compare_optimization_methods()
        elif mode == "production":
            production_ready_optimization()
        elif mode == "analyze":
            if len(sys.argv) > 2:
                analyze_optimization_results(sys.argv[2])
            else:
                print("使用法: python script.py analyze <json_file>")
        else:
            print(f"不明なモード: {mode}")
    else:
        print("使用可能なモード:")
        print("  python script.py standard   - 標準的な最適化")
        print("  python script.py ensemble   - アンサンブル最適化（推奨）")
        print("  python script.py custom     - カスタム最適化")
        print("  python script.py compare    - 手法の比較")
        print("  python script.py production - 本番環境向け")
        print("  python script.py analyze <file> - 結果の分析")
        print("\nデフォルトでアンサンブル最適化を実行します...")
        run_ensemble_optimization()
