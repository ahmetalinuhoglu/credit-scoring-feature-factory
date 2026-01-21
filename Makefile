.PHONY: help spark-up spark-down spark-logs spark-test test benchmark clean

# Default target
help:
	@echo "Credit Scoring ML Pipeline - Development Commands"
	@echo ""
	@echo "Spark Commands:"
	@echo "  make spark-up      Start local Spark cluster (Docker)"
	@echo "  make spark-down    Stop local Spark cluster"
	@echo "  make spark-logs    View Spark master logs"
	@echo "  make spark-test    Run Spark compatibility tests"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test          Run all unit tests"
	@echo "  make test-feature  Run FeatureFactory tests only"
	@echo "  make benchmark     Run feature generation benchmark"
	@echo ""
	@echo "Other:"
	@echo "  make clean         Remove cache files and outputs"

# ═══════════════════════════════════════════════════════════════
# SPARK COMMANDS
# ═══════════════════════════════════════════════════════════════

spark-up:
	@echo "Starting local Spark cluster..."
	docker-compose up -d
	@echo "Spark UI available at http://localhost:8080"

spark-down:
	@echo "Stopping Spark cluster..."
	docker-compose down

spark-logs:
	docker-compose logs -f spark-master

spark-test:
	@echo "Running Spark compatibility tests..."
	python scripts/spark_local_test.py

# ═══════════════════════════════════════════════════════════════
# TESTING COMMANDS
# ═══════════════════════════════════════════════════════════════

test:
	pytest tests/ -v --tb=short

test-feature:
	pytest tests/features/test_feature_factory.py -v --tb=short --no-cov

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html

benchmark:
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

benchmark-parallel:
	@echo "Running FeatureFactory benchmark (parallel mode)..."
	python -c "\
import time; \
import pandas as pd; \
from src.features.feature_factory import FeatureFactory; \
apps = pd.read_csv('data/sample/sample_applications.csv'); \
bureau = pd.read_csv('data/sample/sample_credit_bureau.csv'); \
print(f'Applications: {len(apps)}, Bureau: {len(bureau)}'); \
factory = FeatureFactory(); \
start = time.time(); \
result = factory.generate_all_features(apps, bureau, parallel=True, n_jobs=-1); \
elapsed = time.time() - start; \
print(f'Time: {elapsed:.1f}s | Rows: {len(result)} | Cols: {len(result.columns)}'); \
print(f'Rate: {len(apps)/elapsed:.1f} applications/sec')"

# ═══════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage 2>/dev/null || true
	@echo "Cleaned cache files"
