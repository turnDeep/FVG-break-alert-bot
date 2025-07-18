# FVG Break Alert Bot 📈

S&P500銘柄のFair Value Gap（FVG）とレジスタンス突破を検出し、Discordで通知する自動売買支援ボット。機械学習によるパラメータ最適化機能付き。

## 🌟 主な機能

- **2段階アラート**: FVG検出（戦略1）とレジスタンス突破（戦略2）を段階的に通知。
- **日次スキャン**: 市場クローズ後の日足データに基づき、全S&P500銘柄からシグナルを検出。
- **MLによるパラメータ最適化**: ローカル環境でスクリプトを実行し、バックテストに基づいた最適な戦略パラメータを探索。
- **詳細なレポート**: 最適化結果をJSONとインタラクティブなHTMLダッシュボードで出力。
- **高度なアラート**: FVGの品質やMLからのコメントを含む、視覚的で情報豊富なアラート。
- **柔軟なテスト**: `--mock`フラグにより、Discordに接続せず主要機能の動作確認が可能。

## 🚀 クイックスタート

### 1. 前提条件
- Python 3.11以上
- Discord Bot Token（[Discord Developer Portal](https://discord.com/developers/applications)から取得）

### 2. インストール

```bash
# リポジトリをクローン
git clone https://github.com/turnDeep/FVG-break-alert-bot.git
cd FVG-break-alert-bot

# 仮想環境を作成して有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 必要なライブラリをインストール
pip install -r requirements.txt
pip install setuptools # pkg_resourcesのために必要
```

### 3. 環境変数の設定
`.env.example`をコピーして`.env`ファイルを作成し、エディタで開きます。

```bash
cp .env.example .env
# nano .env または code .env などで編集
```
最低限、`DISCORD_BOT_TOKEN`に取得したトークンを設定してください。その他のパラメータは、後述する`!optimize`コマンドで最適化された値に更新することを推奨します。


## 🤖 Botの実行

Botは日次スキャンモードで動作します。コマンドラインから以下を実行してください。

```bash
# Discordに実際に通知を送信
python bot.py

# Discordに接続せず、結果をコンソールに出力してテスト
python bot.py --mock
```
これにより、S&P500の全銘柄がスキャンされ、シグナルが検出されると設定されたDiscordチャンネルに通知が送信されます。

## 📱 Discord コマンド

| コマンド | 説明 | 使用例 |
|---|---|---|
| `!status` | Botの基本情報を表示します。 | `!status` |
| `!backtest <SYMBOL>` | 現在の設定パラメータを使い、指定した銘柄のバックテストを実行します。 | `!backtest TSLA` |


## 📊 ML最適化とバックテスト

このボットの戦略パラメータは、機械学習（Optuna）を使って最適化できます。最適化は計算負荷が高いため、ローカル環境で直接スクリプトを実行します。

### 最適化の実行
```bash
# パラメータ最適化を実行
python run_optimization.py
```
このスクリプトは、以下の処理を自動で行います。
1.  **パラメータ探索**: 過去のデータ（訓練期間）で最もパフォーマンスが良いパラメータの組み合わせを探します。
2.  **検証**: 見つかった最適パラメータを別の期間（テスト期間）で検証し、汎化性能を確認します。
3.  **結果の保存**:
    - 最適なパラメータセットを `optimized_params.json` に保存します。
    - 最適化の過程を可視化した `optimization_results.png` を生成します。

得られた最適パラメータは、`.env`ファイルに反映させることで、以降のスキャンやアラートの精度を向上させることができます。

### バックテスト
`!backtest`コマンドを使うことで、現在の設定が特定の銘柄でどの程度のパフォーマンスを発揮するかを手軽に確認できます。

## 📂 プロジェクト構造

```
FVG-break-alert-bot/
├── bot.py                 # メインのDiscord Botロジック
├── run_optimization.py    # パラメータ最適化実行スクリプト
├── ml_optimizer.py        # ML最適化のコアロジック
├── backtest.py            # バックテストエンジン
├── indicators.py          # 高度なテクニカル指標
├── notifier.py            # Discord通知・モック通知クラス
├── requirements.txt       # Python依存関係
├── .env.example           # 環境変数テンプレート
└── optimization_results/  # 最適化レポートが保存されるディレクトリ
```

## ⚠️ 注意事項
- このボットは投資助言を目的としたものではありません。全ての取引は自己責任で行ってください。
- `yfinance` APIは時々不安定になることがあります。データ取得エラーが発生した場合は、時間をおいて再試行してください。
- `.env`ファイル、特に`DISCORD_BOT_TOKEN`は絶対に公開しないでください。

---

**免責事項**: このボットは教育および情報提供のみを目的として作成されています。実際の投資判断は、ご自身の責任において、十分なリサーチとリスク評価の上で行ってください。過去のパフォーマンスは将来の結果を保証するものではありません。
