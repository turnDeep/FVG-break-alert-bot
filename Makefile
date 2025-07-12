.PHONY: help build run optimize test clean jupyter format lint

# デフォルトターゲット
help:
	@echo "使用可能なコマンド:"
	@echo "  make build      - Dev Containerをビルド"
	@echo "  make run        - Dev Containerを起動"
	@echo "  make optimize   - ML最適化を実行"
	@echo "  make test       - テストを実行"
	@echo "  make jupyter    - Jupyter Labを起動"
	@echo "  make format     - コードフォーマット"
	@echo "  make lint       - コード品質チェック"
	@echo "  make clean      - 一時ファイルを削除"

# Dev Containerビルド
build:
	docker build -t fvg-ml-optimizer .devcontainer/

# Dev Container起動
run:
	docker run -it --rm \
		-v $(PWD):/workspace \
		-p 8888:8888 \
		-p 6006:6006 \
		--name fvg-ml-optimizer \
		fvg-ml-optimizer

# ML最適化実行
optimize:
	python ml_optimizer.py

# テスト実行
test:
	python -m pytest tests/ -v --cov=.

# Jupyter Lab起動
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# コードフォーマット
format:
	black . --line-length 100
	isort . --profile black

# Lintチェック
lint:
	flake8 . --max-line-length=100 --exclude=.venv,venv,__pycache__
	pylint *.py --max-line-length=100

# クリーンアップ
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -f optimization_results.png
	rm -f optimized_params.json

# 環境セットアップ
setup:
	cp .env.example .env
	pip install -r requirements.txt

# 結果の表示
show-results:
	@if [ -f optimized_params.json ]; then \
		echo "=== 最適化されたパラメータ ==="; \
		cat optimized_params.json | python -m json.tool; \
	else \
		echo "最適化結果が見つかりません。'make optimize'を実行してください。"; \
	fi
