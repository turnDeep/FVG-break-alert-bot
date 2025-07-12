# FVG Break Alert Bot 設定ファイル - 機械学習最適化用

# ===== Discord設定（MLオプティマイザーでは不要） =====
# DISCORD_BOT_TOKEN=your_discord_bot_token_here

# ===== 初期パラメータ（MLで最適化される） =====

# Bot基本設定
BOT_CHANNEL_NAME=fvg-break-alerts
SCAN_INTERVAL=15

# テクニカル設定（これらが最適化対象）
MA_PERIOD=200                # 移動平均期間（50-300で最適化）
FVG_MIN_GAP_PERCENT=0.5      # 最小ギャップサイズ（0.1-2.0%で最適化）
RESISTANCE_LOOKBACK=20       # レジスタンス検出期間（10-40日で最適化）
BREAKOUT_THRESHOLD=1.005     # 突破判定の閾値（0.1-2.0%で最適化）

# リスク管理設定（これらも最適化対象）
STOP_LOSS_RATE=0.02          # ストップロス率（1-5%で最適化）
TARGET_PROFIT_RATE=0.05      # 利確目標率（2-10%で最適化）

# ===== ML最適化設定 =====

# Optuna設定
OPTUNA_N_TRIALS=100          # 最適化試行回数
OPTUNA_N_JOBS=4             # 並列実行数

# バックテスト期間
TRAIN_START_DATE=2022-01-01
TRAIN_END_DATE=2024-01-01
TEST_START_DATE=2024-01-01
TEST_END_DATE=2024-12-31

# 最適化対象銘柄数（デバッグ用）
MAX_SYMBOLS=100              # 全S&P500なら500に設定

# ===== その他の設定 =====

# デバッグモード
DEBUG_MODE=false

# ログレベル
LOG_LEVEL=INFO
