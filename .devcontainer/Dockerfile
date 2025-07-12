FROM python:3.11-slim-bullseye

# 作業ディレクトリ設定
WORKDIR /workspace

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    # 基本ツール
    build-essential \
    wget \
    curl \
    git \
    vim \
    sudo \
    # Python開発用
    python3-dev \
    # グラフ描画用
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 追加の依存関係
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# vscodeユーザーの作成
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Pythonパッケージのインストール
COPY requirements.txt /tmp/requirements.txt
COPY .devcontainer/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements-dev.txt

# 作業ディレクトリの所有権設定
RUN chown -R vscode:vscode /workspace

# ユーザーを切り替え
USER vscode

# Jupyter設定
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

# Pythonパスの設定
ENV PYTHONPATH=/workspace:$PYTHONPATH

# エントリーポイント
CMD ["/bin/bash"]
