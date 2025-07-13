# FVG Break Alert Bot 📈

S&P500銘柄のFair Value Gap（FVG）とレジスタンス突破を検出し、Discord通知する自動売買支援ボット。機械学習によるパラメータ最適化機能付き。

## 🌟 主な機能

### 2段階アラートシステム
1. **戦略1（FVG検出）**: 週足200SMA以上 + 日足200MA付近でFVGを検出
2. **戦略2（レジスタンス突破）**: 戦略1の条件達成後、レジスタンスを突破

### 特徴
- S&P500全銘柄（約500銘柄）を自動監視
- 15分間隔でスキャン（市場開場時のみ）
- 美しいチャート付きDiscordアラート
- 機械学習によるパラメータ最適化
- バックテスト機能による戦略検証

## 🚀 クイックスタート

### 1. 前提条件
- Python 3.11以上
- Discord Bot Token（[Discord Developer Portal](https://discord.com/developers/applications)から取得）
- Docker Desktop（Dev Container使用時）

### 2. インストール

```bash
# リポジトリをクローン
git clone https://github.com/turnDeep/FVG-break-alert-bot.git
cd FVG-break-alert-bot

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt

# 環境変数を設定
cp .env.example .env
# .envファイルを編集してDISCORD_BOT_TOKENを設定
```

### 3. Botの起動

```bash
python bot.py
```

## 🎯 戦略の詳細

### 戦略1: FVG検出条件
- 価格が週足200SMA以上
- 価格が日足200MA付近（デフォルトは±5%以内、ML最適化により変動）
- FVG（Fair Value Gap）が発生
  - 3本目の安値 > 1本目の高値
  - ギャップサイズ: 0.5%以上

### 戦略2: レジスタンス突破条件
- 戦略1の全条件を満たす
- 直近レジスタンスを突破（×1.005倍以上）
- 前日終値がレジスタンス以下

## 📊 ML最適化

### 最適化の実行

```bash
# 基本的な実行（100回試行）
python ml_optimizer.py

# カスタム設定での実行
python run_optimization.py
```

### Dev Containerでの実行（推奨）

```bash
# VSCodeでプロジェクトを開く
code .

# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
# コンテナ内で実行
make optimize
```

### 最適化プロセス
1. S&P500銘柄を訓練用（80%）と検証用（20%）に分割
2. Optunaで以下のパラメータを最適化：
   - MA期間（20-200）
   - FVG最小ギャップ（0.1-1.0%）
   - レジスタンス検出期間（5-30日）
   - 突破閾値（1.0-1.01）
   - ストップロス率（1-5%）
   - 利確率（1-10%）
   - MA近接判定（5-20%）

## 🔧 設定項目（.env）

```env
# 必須設定
DISCORD_BOT_TOKEN=your_bot_token_here
BOT_CHANNEL_NAME=fvg-break-alerts
SCAN_INTERVAL=15

# テクニカル設定（ML最適化で自動調整可能）
MA_PERIOD=200
FVG_MIN_GAP_PERCENT=0.5
RESISTANCE_LOOKBACK=20
BREAKOUT_THRESHOLD=1.005
STOP_LOSS_RATE=0.02
TARGET_PROFIT_RATE=0.05
MA_PROXIMITY_PERCENT=0.05

# デバッグ
DEBUG_MODE=false
```

## 📱 Discord コマンド

| コマンド | 説明 |
|---------|------|
| `!status` | Botの稼働状態を表示 |
| `!check [SYMBOL]` | 特定銘柄の現在状態をチェック |
| `!fvg_list` | FVG発生銘柄リストを表示 |
| `!resistance_list` | レジスタンス突破銘柄リストを表示 |
| `!scan` | 手動スキャン実行（管理者のみ） |

## 🧪 バックテスト

```python
from backtest import FVGBreakBacktest

# バックテスト実行
backtester = FVGBreakBacktest()
result = backtester.run_backtest(
    symbol="NVDA",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# レポート表示
print(backtester.create_summary_report(result))
```

## 📂 プロジェクト構造

```
FVG-break-alert-bot/
├── bot.py                 # メインのDiscord Bot
├── backtest.py           # バックテストエンジン
├── ml_optimizer.py       # ML最適化モジュール
├── indicators.py         # 高度なテクニカル指標
├── requirements.txt      # Python依存関係
├── .env                  # 環境変数（要作成）
├── .devcontainer/        # Dev Container設定
│   ├── devcontainer.json
│   ├── Dockerfile
│   └── requirements-dev.txt
├── tests/               # テストスイート
│   └── test_ml_optimizer.py
└── Makefile            # 便利なコマンド集
```

## 🐳 Docker / Dev Container

### Dev Container の利点
- 環境構築の自動化
- 依存関係の完全な管理
- チーム間での環境統一

### 使用方法
1. Docker Desktop をインストール
2. VSCode の「Dev Containers」拡張機能をインストール
3. `Ctrl+Shift+P` → 「Dev Containers: Reopen in Container」

### Makefile コマンド
```bash
make help        # ヘルプ表示
make optimize    # ML最適化実行
make test        # テスト実行
make jupyter     # Jupyter Lab起動
make format      # コードフォーマット
make lint        # コード品質チェック
```

## ⚠️ 注意事項

### リスク管理
- このボットは投資助言ではありません
- 実際の取引は自己責任で行ってください
- 必ずストップロスを設定してください

### API制限
- yfinanceのAPI制限に注意
- Discord APIのレート制限を考慮
- 大量銘柄スキャン時は間隔を調整

### セキュリティ
- `.env`ファイルは絶対にGitにコミットしない
- Discord Bot Tokenは厳重に管理
- 本番環境では適切な権限設定を

## 🔍 トラブルシューティング

### よくある問題

**Q: トレードが全く検出されない**
```bash
# パラメータを緩くしてテスト
python step1.py
```

**Q: メモリ不足エラー**
```bash
# Docker Desktopのメモリ割り当てを増やす（推奨: 8GB以上）
```

**Q: ML最適化が遅い**
```bash
# 銘柄数や試行回数を減らす
MAX_SYMBOLS=50 OPTUNA_N_TRIALS=30 python run_optimization.py
```

## 📚 参考資料

- [Fair Value Gap（FVG）とは](https://www.investopedia.com/terms/f/fair-value.asp)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [discord.py Documentation](https://discordpy.readthedocs.io/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- S&P500データ: Wikipedia
- 株価データ: Yahoo Finance
- チャート: mplfinance
- ML最適化: Optuna

---

**免責事項**: このボットは教育目的で作成されています。実際の投資判断は自己責任で行ってください。過去のパフォーマンスは将来の結果を保証するものではありません。