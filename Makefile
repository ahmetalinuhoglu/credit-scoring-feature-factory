.PHONY: help run run-sample test test-unit test-integ test-coverage clean-outputs lint format spark-up spark-down spark-logs spark-test benchmark clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ═══════════════════════════════════════════════════════════════
# MODEL DEVELOPMENT
# ═══════════════════════════════════════════════════════════════

run:  ## Run model development pipeline with config
	python scripts/run_model_development.py --config config/model_development.yaml

run-sample:  ## Run on sample data with default config
	python scripts/run_model_development.py --config config/model_development.yaml --input data/sample/sample_features.parquet --train-end-date 2024-06-30

# ═══════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════

test:  ## Run all tests
	pytest tests/ -v

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integ:  ## Run integration tests only
	pytest tests/integration/ -v --timeout=300

test-coverage:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ═══════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════

lint:  ## Run linters
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:  ## Format code
	black src/ tests/ scripts/ --line-length=100
	isort src/ tests/ scripts/

# ═══════════════════════════════════════════════════════════════
# SPARK COMMANDS
# ═══════════════════════════════════════════════════════════════

spark-up:  ## Start local Spark cluster (Docker)
	@echo "Starting local Spark cluster..."
	docker-compose up -d
	@echo "Spark UI available at http://localhost:8080"

spark-down:  ## Stop local Spark cluster
	@echo "Stopping Spark cluster..."
	docker-compose down

spark-logs:  ## View Spark master logs
	docker-compose logs -f spark-master

spark-test:  ## Run Spark compatibility tests
	@echo "Running Spark compatibility tests..."
	python scripts/spark_local_test.py

# ═══════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════

benchmark:  ## Run FeatureFactory benchmark
	@echo "Running FeatureFactory benchmark..."
	python -c "\
import time; \
import pandas as pd; \
from src.features.feature_factory import FeatureFactory; \
apps = pd.read_csv('data/sample/sample_applications.csv'); \
bureau = pd.read_csv('data/sample/sample_credit_bureau.csv'); \
print(f'Applications: {len(apps)}, Bureau: {len(bureau)}'); \
factory = FeatureFactory(); \
start = time.time(); \
result = factory.generate_all_features(apps, bureau); \
elapsed = time.time() - start; \
print(f'Time: {elapsed:.1f}s | Rows: {len(result)} | Cols: {len(result.columns)}'); \
print(f'Rate: {len(apps)/elapsed:.1f} applications/sec')"

# ═══════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════

clean:  ## Remove cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage 2>/dev/null || true
	@echo "Cleaned cache files"

clean-outputs:  ## Remove outputs older than 30 days
	find outputs/model_development -maxdepth 1 -mtime +30 -exec rm -rf {} \;
