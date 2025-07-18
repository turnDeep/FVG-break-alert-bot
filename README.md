# FVG Break Alert Bot 📈

S&P500銘柄のFair Value Gap（FVG）とレジスタンス突破を検出し、Discordで通知する自動売買支援ボット。機械学習によるパラメータ最適化機能付き。

## 🌟 主な機能

- **2段階アラート**: FVG検出（戦略1）とレジスタンス突破（戦略2）を段階的に通知。
- **日足/リアルタイム分析**: 市場クローズ後の日足分析と、市場開催中のリアルタイム分析の両モードに対応。
- **MLによる最適化**: `!optimize`コマンド一つで、バックテストに基づいた最適な戦略パラメータを探索。
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


## 🤖 Botの起動と実行モード

本ボットには2つの主要な実行モードがあります。

### 1. 日足スキャンモード (Daily Scan)
市場クローズ後に1回だけ実行し、その日の日足データに基づいてシグナルを検出します。

```bash
# Discordに実際に通知を送信
python bot.py daily

# Discordに接続せず、結果をコンソールに出力してテスト
python bot.py daily --mock
```

### 2. リアルタイムモード (Real-time Bot)
Discordボットとして常駐し、市場開催中に設定された間隔（デフォルト15分）でリアルタイムに市場をスキャンし続けます。`!optimize`などのコマンドもこのモードで使用します。

```bash
python bot.py realtime
```

## 📱 Discord コマンド

| コマンド | 説明 | 使用例 |
|---|---|---|
| `!status` | Botの稼働状態を表示します。 | `!status` |
| `!check <SYMBOL>` | 指定した銘柄の現在のテクニカル状況を分析し、チャート付きで表示します。 | `!check NVDA` |
| `!backtest <SYMBOL>` | 現在の設定パラメータを使い、指定した銘柄のバックテストを実行します。 | `!backtest TSLA` |
| `!optimize [TRIALS]` | MLによるパラメータ最適化を開始します。完了後、結果のJSONとHTMLレポートを送信します。**（時間がかかります）** | `!optimize 50` |
| `!fvg_list` | 現在FVGが検出されている銘柄のリストを表示します。 | `!fvg_list` |
| `!resistance_list` | レジスタンスを突破した銘柄のリストを表示します。 | `!resistance_list` |


## 📊 ML最適化とバックテスト

### `!optimize` コマンド
このボットの中核機能です。Discordからこのコマンドを実行すると、以下のプロセスが自動的に実行されます。
1.  **パラメータ探索**: Optunaを使い、過去のデータで最もパフォーマンスが良いパラメータの組み合わせを探します。
2.  **詳細バックテスト**: 見つかった最適パラメータを使い、全S&P500銘柄で詳細なバックテストを実行します。
3.  **結果の保存とレポート**:
    - 全ての詳細データを含む`optimization_*.json`ファイル。
    - 人間が視覚的に確認するための`optimization_report_*.html`ファイル。
4.  **結果の通知**: 上記2つのファイルをDiscordにアップロードして通知します。

得られた最適パラメータは、`.env`ファイルに反映させることで、以降のスキャンやアラートの精度を向上させることができます。

### `!backtest` コマンド
`!optimize`が全体の戦略を改善するのに対し、`!backtest`は現在の設定を使って特定の銘柄のパフォーマンスを手軽に確認するためのコマンドです。

### モックテスト
`!optimize`コマンドの動作を、Discordに接続せずに確認することも可能です。
```bash
# 試行回数5回で最適化をテスト実行
python bot.py realtime --mock --command=optimize
```

## 📂 プロジェクト構造

```
FVG-break-alert-bot/
├── bot.py                 # メインのDiscord Botロジック
├── optimization_manager.py # ML最適化とバックテストの管理クラス
├── backtest.py            # バックテストエンジン
├── indicators.py          # 高度なテクニカル指標
├── notifier.py            # Discord通知・モック通知クラス
├── requirements.txt       # Python依存関係
├── .env.example           # 環境変数テンプレート
└── optimization_results/  # 最適化結果が保存されるディレクトリ
```

## ⚠️ 注意事項
- このボットは投資助言を目的としたものではありません。全ての取引は自己責任で行ってください。
- `yfinance` APIは時々不安定になることがあります。データ取得エラーが発生した場合は、時間をおいて再試行してください。
- `.env`ファイル、特に`DISCORD_BOT_TOKEN`は絶対に公開しないでください。

---

**免責事項**: このボットは教育および情報提供のみを目的として作成されています。実際の投資判断は、ご自身の責任において、十分なリサーチとリスク評価の上で行ってください。過去のパフォーマンスは将来の結果を保証するものではありません。
