import pprint
from backtest import FVGBreakBacktest

if __name__ == "__main__":
    # テスト用の非常に緩いパラメータ
    test_params = {
        'ma_period': 50,
        'fvg_min_gap': 0.01,  # 非常に小さなギャップでも検出
        'resistance_lookback': 10,
        'breakout_threshold': 1.0001,  # ほぼブレイクしていなくてもOK
        'stop_loss_rate': 0.05,
        'target_profit_rate': 0.1
    }

    # バックテスターを作成
    backtester = FVGBreakBacktest(**test_params)

    # 特定の銘柄（例：AAPL）と期間でテスト
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    print(f"--- 手動テスト開始: {symbol} ---")
    result = backtester.run_backtest(symbol, start_date, end_date)

    # 結果を詳細に出力
    pprint.pprint(result)

    if result['total_trades'] > 0:
        print("\n🎉 トレードが発生しました！ここから条件を調整していきましょう。")
    else:
        print("\n😢 やはりトレードが発生しませんでした。backtest.pyのロジック自体を見直す必要があります。")
