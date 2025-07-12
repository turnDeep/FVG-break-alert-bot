# FVG ML Optimizer Dev Container

このDev Containerは、FVG Break Alert Botのパラメータを機械学習で最適化するための開発環境です。

## 🚀 クイックスタート

### 1. 前提条件

- Docker Desktop がインストールされていること
- Visual Studio Code がインストールされていること
- VSCode拡張機能「Dev Containers」がインストールされていること

### 2. Dev Containerの起動

1. VSCodeでプロジェクトフォルダを開く
2. `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`) でコマンドパレットを開く
3. 「Dev Containers: Reopen in Container」を選択
4. 初回起動時は数分かかります（Dockerイメージのビルドのため）

### 3. 環境設定

```bash
# .envファイルの作成
cp .env.example .env

# 必要に応じて.envファイルを編集
# 特にML最適化のパラメータを調整
```

## 🧪 ML最適化の実行

### 基本的な実行方法

```bash
# Dev Container内のターミナルで実行
python ml_optimizer.py
```

### Jupyter Notebookでの実行

```bash
# Jupyter Labを起動
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# ブラウザで http://localhost:8888 にアクセス
```

### カスタム設定での実行

```python
# Python内での実行例
from ml_optimizer import FVGParameterOptimizer

# 最適化インスタンスを作成
optimizer = FVGParameterOptimizer(
    n_trials=200,  # 試行回数を増やす
    n_jobs=8       # 並列数を増やす（CPUコア数に応じて）
)

# 最適化実行
best_params = optimizer.optimize(
    start_date='2021-01-01',
    end_date='2023-12-31'
)

# テスト期間で検証
test_results = optimizer.validate_best_params(
    test_period_start='2024-01-01',
    test_period_end='2024-12-31'
)

# 結果を保存
optimizer.save_results()
```

## 📊 結果の確認

最適化が完了すると、以下のファイルが生成されます：

- `optimized_params.json` - 最適化されたパラメータ
- `optimization_results.png` - 最適化過程の可視化

## 🔧 トラブルシューティング

### メモリ不足エラー

```bash
# Dockerのメモリ割り当てを増やす
# Docker Desktop > Settings > Resources > Memory
# 推奨: 8GB以上
```

### パッケージインストールエラー

```bash
# コンテナ内で手動インストール
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU使用時の設定（オプション）

GPUを使用する場合は、NVIDIA Container Toolkitが必要です：

```bash
# ホストマシンで実行
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

## 📈 パフォーマンスチューニング

### 並列処理の最適化

```python
# CPUコア数に応じて調整
import multiprocessing
n_cores = multiprocessing.cpu_count()
optimizer = FVGParameterOptimizer(n_jobs=n_cores - 1)
```

### メモリ効率の改善

```python
# バッチサイズを調整
symbols_batch = symbols[:50]  # 一度に処理する銘柄数を制限
```

## 🔐 セキュリティ注意事項

- `.env`ファイルに実際のAPIキーやトークンを含める場合は、絶対にGitにコミットしないこと
- Dev Containerは開発用途のみに使用し、本番環境では使用しないこと

## 📚 参考資料

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
