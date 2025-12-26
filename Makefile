.PHONY: setup train infer ui test lint clean dev all

setup:
	uv sync --dev
	pre-commit install
	dvc init --force 2>/dev/null || true

train:
	uv run python -m rps.train_with_dvc

infer:
	uv run python -m rps.infer

ui:
	uv run mlflow ui --host 0.0.0.0 --port 8080

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf data/ plots/ tb_logs/ .dvc/ rps_model_*.pth

dev: setup lint train ui
