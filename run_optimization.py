#!/usr/bin/env python3
"""
ML最適化の実行スクリプト
カスタムパラメータで最適化を実行する例
"""
import os
from datetime import datetime, timedelta
from ml_optimizer import FVGParameterOptimizer

def main():
    """メイン実行関数"""
    
    # 最適化パラメータ
    n_trials = int(os.getenv('OPTUNA_N_TRIALS', 50))
    n_jobs = int(os.getenv('OPTUNA_N_JOBS', 6))
    max_symbols = int(os.getenv('MAX_SYMBOLS', 500))
    
    # 期間設定
    today = datetime.now()
    three_years_ago = today - timedelta(days=365*3)
    one_year_ago = today - timedelta(days=365*1)

    train_start = os.getenv('TRAIN_START_DATE', three_years_ago.strftime('%Y-%m-%d'))
    train_end = os.getenv('TRAIN_END_DATE', one_year_ago.strftime('%Y-%m-%d'))
    test_start = os.getenv('TEST_START_DATE', one_year_ago.strftime('%Y-%m-%d'))
    test_end = os.getenv('TEST_END_DATE', today.strftime('%Y-%m-%d'))
    
    print("=" * 60)
    print("FVG Break Alert Bot - ML Parameter Optimization")
    print("=" * 60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"試行回数: {n_trials}")
    print(f"並列ジョブ数: {n_jobs}")
    print(f"最大銘柄数: {max_symbols}")
    print(f"訓練期間: {train_start} - {train_end}")
    print(f"テスト期間: {test_start} - {test_end}")
    print("=" * 60)
    
    # 最適化インスタンスを作成
    optimizer = FVGParameterOptimizer(
        n_trials=n_trials,
        n_jobs=n_jobs
    )
    
    # 銘柄数を制限（デバッグ用）
    original_get_sp500 = optimizer.get_sp500_symbols
    def limited_get_sp500():
        symbols = original_get_sp500()
        return symbols[:max_symbols]
    optimizer.get_sp500_symbols = limited_get_sp500
    
    try:
        # 最適化実行
        print("\n📊 最適化を開始します...")
        best_params = optimizer.optimize(
            start_date=train_start,
            end_date=train_end
        )
        
        # テスト期間で検証
        print("\n🧪 テスト期間で検証中...")
        test_results = optimizer.validate_best_params(
            test_period_start=test_start,
            test_period_end=test_end
        )
        
        # 結果を保存
        print("\n💾 結果を保存中...")
        optimizer.save_results()
        
        # 可視化
        print("\n📈 結果を可視化中...")
        optimizer.plot_optimization_history()
        
        # サマリー表示
        print("\n" + "=" * 60)
        print("✅ 最適化完了！")
        print("=" * 60)
        print(f"最適パラメータ:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        print("\n" + "-" * 60)
        print("🧪 テスト期間での合算パフォーマンス")
        print("-" * 60)
        if test_results.get('error'):
            print(f"エラー: {test_results['error']}")
        else:
            s1 = test_results['s1_stats']
            s2 = test_results['s2_stats']
            print(f"対象銘柄数: {test_results['symbols_tested']} (うちエラー: {test_results['symbols_with_errors']})")
            print("\n--- 戦略1: FVG検出 ---")
            print(f"トレード数: {s1['count']} 回")
            print(f"勝率: {s1['win_rate']:.2f}%")
            print(f"平均リターン: {s1['avg_return']:.2f}%")

            print("\n--- 戦略2: FVG検出 → レジスタンス突破 ---")
            print(f"転換率 (S1→S2): {s2['conversion_rate']:.2f}%")
            print(f"トレード数: {s2['count']} 回")
            print(f"勝率: {s2['win_rate']:.2f}%")
            print(f"平均リターン: {s2['avg_return']:.2f}% (S1エントリーからの最終リターン)")

        print("\n結果ファイル:")
        print("  - optimized_params.json (最適化パラメータ)")
        print("  - optimization_results.png (最適化過程)")
        
        # サンプルチャート作成
        optimizer.create_example_chart(test_start, test_end)
        print("  - example_trade_chart.png (戦略2成功例)")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
