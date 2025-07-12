"""
ML Optimizerのテストスイート
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml_optimizer import FVGParameterOptimizer

class TestFVGParameterOptimizer:
    """FVGParameterOptimizerのテストクラス"""
    
    @pytest.fixture
    def optimizer(self):
        """テスト用のオプティマイザーインスタンス"""
        return FVGParameterOptimizer(n_trials=5, n_jobs=1)  # テスト用に小さく設定
    
    def test_initialization(self, optimizer):
        """初期化のテスト"""
        assert optimizer.n_trials == 5
        assert optimizer.n_jobs == 1
        assert optimizer.best_params is None
        assert optimizer.optimization_results == []
    
    def test_get_sp500_symbols(self, optimizer):
        """S&P500銘柄取得のテスト"""
        symbols = optimizer.get_sp500_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)
    
    def test_calculate_score_no_trades(self, optimizer):
        """トレードなしの場合のスコア計算テスト"""
        result = {
            'total_trades': 0,
            'avg_return': 0,
            'combined_win_rate': 0,
            'max_loss': 0
        }
        score = optimizer.calculate_score(result)
        assert score == -1000  # ペナルティスコア
    
    def test_calculate_score_with_trades(self, optimizer):
        """トレードありの場合のスコア計算テスト"""
        result = {
            'total_trades': 10,
            'avg_return': 5.0,
            'combined_win_rate': 60.0,
            'max_loss': -2.0
        }
        score = optimizer.calculate_score(result)
        assert score > 0  # 正のスコア
    
    def test_evaluate_parameters_invalid_symbols(self, optimizer):
        """無効な銘柄でのパラメータ評価テスト"""
        params = {
            'ma_period': 200,
            'fvg_min_gap': 0.5,
            'resistance_lookback': 20,
            'breakout_threshold': 1.005,
            'stop_loss_rate': 0.02,
            'target_profit_rate': 0.05
        }
        
        # 存在しない銘柄
        invalid_symbols = ['INVALID1', 'INVALID2']
        score = optimizer.evaluate_parameters(
            params, 
            invalid_symbols,
            '2023-01-01',
            '2023-12-31'
        )
        assert score == -1000  # エラー時のスコア
    
    @pytest.mark.slow
    def test_optimize_small_dataset(self, optimizer):
        """小規模データセットでの最適化テスト"""
        # テスト用に銘柄を限定
        optimizer.get_sp500_symbols = lambda: ['AAPL', 'MSFT', 'GOOGL']
        
        # 短期間で最適化
        best_params = optimizer.optimize(
            start_date='2023-01-01',
            end_date='2023-06-30'
        )
        
        assert isinstance(best_params, dict)
        assert 'ma_period' in best_params
        assert 'fvg_min_gap' in best_params
        assert 'resistance_lookback' in best_params
        assert 'breakout_threshold' in best_params
        assert 'stop_loss_rate' in best_params
        assert 'target_profit_rate' in best_params
    
    def test_save_results(self, optimizer, tmp_path):
        """結果保存のテスト"""
        # ダミーの最適化結果を設定
        optimizer.best_params = {
            'ma_period': 200,
            'fvg_min_gap': 0.5,
            'resistance_lookback': 20,
            'breakout_threshold': 1.005,
            'stop_loss_rate': 0.02,
            'target_profit_rate': 0.05
        }
        
        optimizer.optimization_results = [{
            'params': optimizer.best_params,
            'train_score': 50,
            'val_score': 45,
            'combined_score': 48
        }]
        
        # 一時ファイルに保存
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            optimizer.save_results('test_results.json')
            
            # ファイルが作成されたか確認
            assert os.path.exists('test_results.json')
            
            # 内容を確認
            import json
            with open('test_results.json', 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data['best_params'] == optimizer.best_params
            assert 'optimization_date' in saved_data
            assert len(saved_data['optimization_history']) == 1
            
        finally:
            os.chdir(original_dir)

# pytest設定
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
