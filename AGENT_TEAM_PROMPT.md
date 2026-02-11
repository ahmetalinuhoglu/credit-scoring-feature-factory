# Agent Team Prompt: Credit Scoring Pipeline Overhaul

Create an agent team with 5 teammates to overhaul this credit scoring ML pipeline into a production-grade, modular, reproducible system. Each teammate owns a distinct domain with clear file boundaries. Use Opus for all teammates. Do not require plan approval for any change or plan.

## Context: Current State

This is a credit scoring pipeline with two phases:
1. **Feature Extraction** (Spark) — one-time job, already complete, outputs parquet. Do NOT modify `scripts/spark_feature_extraction.py` or `src/features/`. These are frozen.
2. **Model Development** (Pandas/XGBoost) — the focus of this overhaul. Currently lives in `src/model_development/` and `scripts/run_model_development.py`.

Current pipeline steps: Data Loading → Constant Elimination → Missing Elimination → IV Filtering → PSI Stability → Correlation Filtering → Forward Feature Selection → Model Evaluation → Excel Report.

The pipeline works but is monolithic, not fully reproducible, has no formal config system, no proper artifact management, and lacks validation rigor expected in credit risk.

## Architecture Principles (ALL teammates must follow)

### 1. Modularity — Plug-and-Play Components
Every pipeline step must be an independent, self-contained module that:
- Can be imported and used standalone: `from src.components.iv_filter import IVFilter`
- Has a consistent interface: each component implements `fit(X, y)` → returns results, and `transform(X)` → returns filtered X
- Can be composed into pipelines OR used individually in notebooks
- Has its own default config section but accepts overrides
- Returns structured results (dataclass or dict) with metadata, not just filtered data

### 2. Single Config, Dual Execution
One YAML config file drives everything. Both notebook and script consume the same config:
```
# From script:
python scripts/run_model_development.py --config config/model_development.yaml

# From notebook:
config = load_config("config/model_development.yaml")
pipeline = ModelDevelopmentPipeline(config)
# OR use components individually:
iv_filter = IVFilter(config.iv)
iv_results = iv_filter.fit(X_train, y_train)
X_filtered = iv_filter.transform(X_train)
```

Config overrides via CLI args should still work (they override YAML values). The notebook reads the same YAML — no separate notebook config.

### 3. Reproducibility — Everything Saved, Everything Seeded
- Global random seed propagated to ALL random operations (train/test split, XGBoost, numpy, python random)
- The EXACT config used for a run is saved as `config.yaml` in the run output directory
- Every run gets a unique run_id: `{YYYYMMDD}_{HHMMSS}_{short_hash}` (hash of config for quick comparison)

### 4. Output Filesystem Structure
```
outputs/
└── model_development/
    └── {run_id}/                          # e.g., 20260212_143052_a1b2c3
        ├── config.yaml                    # Frozen config snapshot (exact replica)
        ├── run_metadata.json              # Run info: git hash, python version, package versions, duration, status
        ├── data/
        │   ├── train_test_split.parquet   # Saved split indices (application_id + set assignment)
        │   └── feature_summary.parquet    # Feature-level stats before any filtering
        ├── steps/
        │   ├── 01_constant/
        │   │   └── results.parquet        # Constant elimination results
        │   ├── 02_missing/
        │   │   └── results.parquet        # Missing value analysis results
        │   ├── 03_iv/
        │   │   ├── results.parquet        # IV scores, categories, univariate metrics
        │   │   └── woe_bins.parquet       # WoE binning boundaries (for deployment)
        │   ├── 04_psi/
        │   │   └── results.parquet        # PSI per check type
        │   ├── 05_correlation/
        │   │   ├── results.parquet        # Correlation pairs and decisions
        │   │   └── correlation_matrix.parquet  # Full correlation matrix of surviving features
        │   ├── 06_selection/
        │   │   ├── results.parquet        # Forward selection steps
        │   │   └── model.json             # XGBoost model (JSON format for portability)
        │   └── 07_evaluation/
        │       ├── performance.parquet    # AUC/Gini/KS per period
        │       ├── lift_tables.parquet    # Decile lift tables
        │       └── feature_importance.parquet
        ├── reports/
        │   └── model_development_report.xlsx  # Final Excel report (all sheets)
        └── logs/
            └── pipeline.log              # Full debug log for this run
```

### 5. Seeds & Determinism
Set seeds in ONE place in config, propagate everywhere:
```yaml
reproducibility:
  global_seed: 42
  # These are derived automatically but can be overridden:
  split_seed: 42      # train/test split
  model_seed: 42      # XGBoost random_state
  numpy_seed: 42      # numpy operations
```
At pipeline start, set `np.random.seed()`, `random.seed()`, and pass seed to all sklearn/xgboost calls.

---

## Team Structure & Assignments

### Teammate 1: "architect" — Data Engineering & Architecture
**Role:** Build the foundational infrastructure — config system, output management, project structure, and the pipeline orchestrator that ties everything together.

**Owned files:**
- `src/config/` — Config loading, validation, schema (Pydantic models)
- `src/pipeline/` — Pipeline orchestrator, step registry, component base classes
- `src/io/` — Output manager, artifact saving/loading, run directory creation
- `config/model_development.yaml` — The master config file
- `scripts/run_model_development.py` — Refactored CLI entry point

**Tasks:**
1. Design and implement the Pydantic config schema (`src/config/schema.py`). The config YAML must have sections for: `data`, `splitting`, `steps` (each step's thresholds), `model` (XGBoost params), `evaluation`, `output`, `reproducibility`. Validate types, ranges, and cross-field constraints (e.g., iv_min < iv_max).

2. Create the config loader (`src/config/loader.py`). Load from YAML, allow CLI overrides, allow programmatic overrides (for notebooks). Freeze config after loading (immutable). Provide `save_config(path)` to dump exact config used.

3. Build the output manager (`src/io/output_manager.py`). Creates the run directory structure shown above. Generates run_id with timestamp + config hash. Saves config snapshot, run metadata (git commit hash via `subprocess`, python version, installed package versions, start/end time, status). Provides `save_step_results(step_name, results_dict)` and `save_artifact(name, obj, format)`.

4. Define the component base class (`src/pipeline/base.py`). Abstract base with `fit(X, y, **kwargs) -> StepResult` and `transform(X) -> pd.DataFrame`. StepResult dataclass holds: step_name, input_features, output_features, eliminated_features, results_df, metadata_dict, duration_seconds. Components must be usable standalone without the pipeline orchestrator.

5. Build the pipeline orchestrator (`src/pipeline/orchestrator.py`). Accepts config, registers steps in order, runs them sequentially, passes outputs forward, handles failures gracefully (save partial results), tracks timing per step, produces final summary. Must support `run_all()`, `run_step(step_name)`, and `run_from(step_name)` (resume from a specific step).

6. Refactor `scripts/run_model_development.py` to use new config system. CLI args override YAML values. Entry point creates config → output manager → pipeline → runs → saves everything.

7. Collect `run_metadata.json`: git commit hash (or "uncommitted"), python version, key package versions (pandas, xgboost, sklearn, numpy), OS, run duration, final status, input file hash (first 1000 rows md5 for quick comparison).

### Teammate 2: "scientist" — Data Science & Model Components
**Role:** Refactor each pipeline step into a modular, plug-and-play component following the base class interface. Improve statistical rigor where the current implementation is weak.

**Owned files:**
- `src/components/` — All pipeline step components (one file per step)
- `src/components/__init__.py` — Clean exports

**Tasks:**
1. `src/components/data_splitter.py` — DataSplitter component.
   - Accepts config for test_size, train_end_date, seed
   - Stratified train/test split within training period
   - OOT auto-split by quarter after train_end_date
   - Saves split indices (application_id → set_name mapping) for reproducibility
   - Returns DataSplitResult with train/test/oot DataFrames + metadata (counts, bad rates per set)
   - Must validate: enough samples per split, target balance not extreme, dates are valid

2. `src/components/constant_filter.py` — ConstantFilter(BaseComponent).
   - fit(): identify features with n_unique < threshold (default 2) on training data
   - transform(): drop those features from any DataFrame
   - Results: per-feature unique count, variance, kept/eliminated status

3. `src/components/missing_filter.py` — MissingFilter(BaseComponent).
   - fit(): identify features exceeding missing_threshold (default 0.70) on training data
   - transform(): drop those features
   - Improvement: also report missing rate on test/OOT for comparison (but decision based on train only)
   - Results: per-feature missing count, missing rate (train/test/oot), kept/eliminated status

4. `src/components/iv_filter.py` — IVFilter(BaseComponent).
   - fit(): calculate IV with configurable n_bins (default 10), iv_min (0.02), iv_max (0.50)
   - Also compute univariate AUC, Gini, KS per feature
   - Save WoE binning boundaries (bin edges + WoE values) — these are needed for deployment/scorecard
   - Handle edge cases: zero events in bin (use Laplace smoothing with configurable epsilon), single-value bins
   - transform(): drop features outside IV range
   - Results: IV score, category, univariate metrics, WoE table, elimination reason

5. `src/components/psi_filter.py` — PSIFilter(BaseComponent).
   - fit(): calculate PSI with pluggable check strategies (quarterly, yearly, consecutive, half-split, custom date)
   - PSI checks are configurable in YAML as a list
   - Handle edge cases: empty bins (add small constant), too few observations for meaningful PSI
   - transform(): drop unstable features
   - Results: per-feature PSI values for each check, max/mean PSI, status

6. `src/components/correlation_filter.py` — CorrelationFilter(BaseComponent).
   - fit(): greedy elimination keeping higher-IV feature, configurable threshold (default 0.90)
   - Improvement: support both Pearson and Spearman (configurable)
   - Improvement: also compute VIF (Variance Inflation Factor) for selected features as additional info
   - transform(): drop correlated features
   - Results: correlation pairs, decisions, VIF values for survivors

7. `src/components/feature_selector.py` — ForwardFeatureSelector(BaseComponent).
   - fit(): forward stepwise selection with XGBoost, features tried in IV-descending order
   - Use proper seed from config for XGBoost random_state
   - Configurable: max_features (optional cap), auc_threshold (min improvement), early_stopping_rounds
   - XGBoost params from config (not hardcoded)
   - Save the final trained XGBoost model in JSON format
   - transform(): select only the chosen features
   - Results: selection steps, per-step AUC, which features added/skipped, final model reference

8. `src/components/model_evaluator.py` — ModelEvaluator(BaseComponent).
   - fit(): evaluate the final model on train/test/all OOT quarters
   - Metrics: AUC, Gini (2*AUC-1), KS, Precision@k, Lift@k (k configurable, default [5, 10, 20])
   - Decile lift tables per period
   - Feature importance (gain, weight, cover from XGBoost)
   - Improvement: add score distribution plots data (histogram bins for each period)
   - Improvement: add PSI of scores between train and each OOT quarter (model stability)
   - Results: performance_df, lift_tables, importance_df, score_psi_df

### Teammate 3: "risk-expert" — Credit Risk Domain & Reporting
**Role:** Ensure the pipeline meets credit risk model development standards. Improve the Excel report. Add domain-specific validations and documentation.

**Owned files:**
- `src/reporting/` — Excel report generator, report templates
- `src/validation/` — Domain-specific validation checks
- `docs/` — Model development documentation

**Tasks:**
1. `src/validation/model_checks.py` — Automated model quality checks that run after evaluation.
   - **Discrimination checks:** AUC > minimum (configurable, default 0.65 for each period), Gini > minimum
   - **Stability checks:** AUC degradation from train to test < threshold (e.g., 5pp), OOT AUC within range of test AUC
   - **Overfit detection:** Train AUC - Test AUC > threshold (e.g., 0.05) triggers warning
   - **Population stability:** Score PSI between train and OOT < threshold
   - **Concentration check:** No single feature contributes > X% importance (configurable)
   - **Monotonicity check:** For each selected feature, check if WoE trend is monotonic (flag non-monotonic)
   - Output: ValidationReport with pass/fail/warning per check, severity levels, recommendations

2. `src/validation/data_checks.py` — Pre-pipeline data quality checks.
   - **Target validation:** Binary (0/1 only), no nulls in target, bad rate within expected range (1%-50%)
   - **Date validation:** Date column exists, is parseable, covers expected range
   - **Feature type checks:** All features numeric (after excluding ID/date columns)
   - **Duplicate check:** No duplicate application_ids
   - **Leakage detection:** Flag features with suspiciously high single-feature AUC (>0.95)
   - These run BEFORE the pipeline starts and block execution on critical failures

3. Overhaul `src/reporting/excel_reporter.py` — Improved Excel report.
   - Keep existing 11 sheets but improve them:
   - **00_Summary:** Add validation check results (pass/fail), config parameters used, data quality summary, warnings/flags
   - **Add new sheet: 08_Validation** — Full validation report with all checks, pass/fail, severity, recommendation
   - **Add new sheet: 09_Score_Distribution** — Score histogram data per period, score PSI
   - **Add new sheet: 10_Config** — Full config dump in readable format (not just YAML, formatted as key-value table)
   - Conditional formatting: Red for FAIL, Yellow for WARNING, Green for PASS in validation sheet
   - Each sheet should have a header comment explaining what it shows and how to interpret it

4. `docs/model_development_methodology.md` — Document the methodology.
   - Purpose and scope of the model
   - Data requirements (input format, required columns, date conventions)
   - Each pipeline step: what it does, why it exists, what thresholds mean, how to interpret results
   - Validation checks: what each check tests and what failures mean
   - How to interpret the output Excel report sheet by sheet
   - Glossary of credit risk terms (IV, WoE, PSI, KS, Gini, AUC, OOT)
   - Keep it practical and concise — this is for the model developer, not a regulatory submission

5. Add credit risk specific logging throughout the pipeline:
   - Log bad rate per OOT quarter (trend monitoring)
   - Log if any OOT quarter has < 30 bads (insufficient for reliable AUC)
   - Log IV distribution summary (how many weak/medium/strong/suspicious)
   - Log warnings when thresholds are set to unusual values (e.g., PSI > 0.5 is very lenient)

### Teammate 4: "tester" — Testing & Quality Assurance
**Role:** Write comprehensive tests for all new components. Ensure the refactored pipeline produces correct results. Test both notebook and script execution paths.

**Owned files:**
- `tests/` — All test files
- `tests/conftest.py` — Shared fixtures
- `tests/data/` — Test fixtures and sample data

**Tasks:**
1. `tests/conftest.py` — Create shared pytest fixtures.
   - `sample_config`: A minimal valid config dict/object
   - `sample_data`: Small synthetic DataFrame (~100 rows, ~20 features) with known properties:
     - 2 constant features (to test constant filter)
     - 2 high-missing features (to test missing filter)
     - 2 low-IV features, 2 high-IV features (to test IV filter)
     - 2 highly correlated features (to test correlation filter)
     - Known target with ~20% bad rate
     - Date column spanning 4 quarters
   - `sample_split_data`: Pre-split train/test/oot from sample_data
   - `tmp_output_dir`: Temporary directory for output testing

2. `tests/unit/test_components.py` — Unit tests for each component.
   - Test each component's fit() and transform() independently
   - Test that constant filter eliminates exactly the constant features
   - Test that missing filter eliminates features above threshold and not below
   - Test IV calculation against hand-calculated values for simple cases
   - Test PSI calculation against known values
   - Test correlation filter keeps the higher-IV feature
   - Test forward selection adds features that improve AUC
   - Test evaluator produces correct metric shapes and ranges

3. `tests/unit/test_config.py` — Config system tests.
   - Test YAML loading, CLI override merging
   - Test validation rejects invalid configs (negative thresholds, iv_min > iv_max, etc.)
   - Test config immutability after loading
   - Test config save/load round-trip (save to YAML, reload, compare)

4. `tests/unit/test_output_manager.py` — Output manager tests.
   - Test directory structure creation
   - Test run_id format
   - Test config snapshot is saved correctly
   - Test artifact save/load round-trip

5. `tests/integration/test_pipeline.py` — End-to-end pipeline test.
   - Run full pipeline on sample_data fixture with known config
   - Verify all output files are created in correct structure
   - Verify config.yaml in output matches input config
   - Verify each step's results are saved as parquet
   - Verify Excel report is generated with all expected sheets
   - Verify pipeline produces deterministic results (run twice with same seed, compare outputs)

6. `tests/integration/test_notebook_script_parity.py` — Verify notebook and script produce identical results.
   - Run pipeline via script entry point
   - Run pipeline via Python API (simulating notebook usage)
   - Both use same config
   - Compare outputs: same features selected, same AUC values (within floating point tolerance)

7. `tests/unit/test_validation.py` — Test validation checks.
   - Test overfit detection triggers when train AUC >> test AUC
   - Test stability check triggers when OOT AUC drops significantly
   - Test data quality checks catch: non-binary target, null targets, duplicate IDs
   - Test leakage detection flags features with AUC > 0.95

### Teammate 5: "mlops" — MLOps, Notebook & Deployment Readiness
**Role:** Create the notebook interface, ensure dual-mode execution works, add MLOps tooling for experiment tracking and deployment readiness.

**Owned files:**
- `notebooks/` — Model development notebooks
- `src/tracking/` — Experiment tracking utilities
- `Makefile` or `justfile` — Task runner commands
- `pyproject.toml` or `setup.py` — Package configuration
- `.gitignore` — Updated for new output structure

**Tasks:**
1. `notebooks/model_development.ipynb` — Comprehensive notebook for interactive model development.
   - **Cell 1: Setup** — Imports, config loading from YAML
   - **Cell 2: Config display** — Show current config, allow overrides in-cell
   - **Cell 3: Data loading & splitting** — Load data, show split summary, bad rates per period
   - **Cell 4-10: Pipeline steps** — Each step in its own cell, using components directly:
     ```python
     # Cell 4: Constant Filter
     const_filter = ConstantFilter(config.steps.constant)
     const_results = const_filter.fit(X_train, y_train)
     print(const_results.summary())
     X_train = const_filter.transform(X_train)
     X_test = const_filter.transform(X_test)
     ```
   - **Cell 11: Evaluation** — Run evaluation, display performance tables
   - **Cell 12: Validation** — Run validation checks, display pass/fail
   - **Cell 13: Save everything** — Save all artifacts to run directory, generate Excel report
   - **Cell 14: Full pipeline mode** — Alternative: run everything in one call
     ```python
     pipeline = ModelDevelopmentPipeline(config)
     results = pipeline.run_all(input_path="data/sample/sample_features.parquet")
     ```
   - Each cell should be self-contained and re-runnable
   - Add markdown cells explaining each step for the notebook user

2. `src/tracking/experiment_tracker.py` — Lightweight experiment tracking.
   - Track: config params, step-level metrics (feature counts), final metrics (AUC per period), duration
   - Storage: Append to a local CSV/JSON file (`outputs/experiment_log.csv`) for easy comparison across runs
   - Columns: run_id, timestamp, config_hash, input_file, train_end_date, n_features_selected, train_auc, test_auc, oot_mean_auc, duration_seconds, status, notes
   - Optional: MLflow integration if mlflow is installed (try/except import), but the CSV tracking must work without mlflow
   - Provide `compare_runs(run_ids)` to compare configs and metrics side by side

3. Create `Makefile` with common commands:
   ```makefile
   run:           python scripts/run_model_development.py --config config/model_development.yaml
   run-sample:    python scripts/run_model_development.py --config config/model_development.yaml --input data/sample/sample_features.parquet --train-end-date 2024-06-30
   test:          pytest tests/ -v
   test-unit:     pytest tests/unit/ -v
   test-integ:    pytest tests/integration/ -v
   clean-outputs: find outputs/model_development -maxdepth 1 -mtime +30 -exec rm -rf {} \;
   ```

4. Update `pyproject.toml` — Proper package configuration.
   - Package name, version, description
   - Dependencies with version pins (not ranges for reproducibility)
   - Optional dependencies: `[spark]` for pyspark, `[tracking]` for mlflow, `[dev]` for pytest/linting
   - Entry points: `model-dev = scripts.run_model_development:main`

5. Update `.gitignore` for new structure:
   - `outputs/model_development/*/` (ignore run outputs, not the directory itself)
   - `*.parquet` in outputs
   - `logs/`
   - Keep `config/` tracked
   - Keep `data/sample/` tracked (small sample data for testing)

6. `src/tracking/run_comparison.py` — Compare two or more runs.
   - Load configs from multiple run directories
   - Diff configs: highlight what changed between runs
   - Compare metrics side by side: which run performed better on which metric
   - Output as DataFrame for easy notebook display

---

## Config File Template

The architect teammate should create `config/model_development.yaml` with this structure:

```yaml
# Model Development Pipeline Configuration
# This file controls all pipeline behavior. Both notebook and script use this config.
# CLI arguments override values in this file.

data:
  input_path: "data/sample/sample_features.parquet"
  target_column: "target"
  date_column: "date"
  id_columns: ["application_id", "customer_id"]
  exclude_columns: ["applicant_type"]  # Additional columns to exclude from features

splitting:
  train_end_date: "2024-06-30"  # Data after this date becomes OOT
  test_size: 0.20               # Fraction of training period for test set
  stratify: true                # Stratify split by target

steps:
  constant:
    enabled: true
    min_unique_values: 2

  missing:
    enabled: true
    threshold: 0.70  # Max missing rate to keep feature

  iv:
    enabled: true
    min_iv: 0.02       # Below = "useless"
    max_iv: 0.50       # Above = "suspicious" (possible leakage)
    n_bins: 10         # Quantile bins for IV calculation
    min_samples_per_bin: 50  # Minimum observations per bin

  psi:
    enabled: true
    threshold: 0.25
    n_bins: 10
    checks:
      - type: "quarterly"
      - type: "yearly"
      - type: "consecutive"
      # - type: "date_split"
      #   date: "2024-04-01"
      #   label: "Pre/Post Apr 2024"

  correlation:
    enabled: true
    threshold: 0.90
    method: "pearson"  # pearson or spearman

  selection:
    enabled: true
    method: "forward"
    auc_threshold: 0.0001  # Min AUC improvement to add feature
    max_features: null      # Optional cap on selected features (null = no cap)

model:
  algorithm: "xgboost"
  params:
    objective: "binary:logistic"
    eval_metric: "auc"
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 300
    early_stopping_rounds: 30
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 1
    gamma: 0
    reg_alpha: 0
    reg_lambda: 1
    # scale_pos_weight: auto  # auto-calculated from data

evaluation:
  metrics: ["auc", "gini", "ks"]
  precision_at_k: [5, 10, 20]  # Percentiles for precision/lift
  n_deciles: 10
  calculate_score_psi: true  # PSI of model scores across periods

validation:
  enabled: true
  checks:
    min_auc: 0.65                    # Minimum acceptable AUC for any period
    max_overfit_gap: 0.05            # Max allowed Train AUC - Test AUC
    max_oot_degradation: 0.08        # Max allowed Test AUC - worst OOT AUC
    max_score_psi: 0.25              # Max PSI of score distribution
    max_feature_concentration: 0.50  # Max single feature importance share
    min_oot_samples: 30              # Min bad count per OOT quarter for reliable evaluation
    check_monotonicity: true         # Check WoE monotonicity of selected features

output:
  base_dir: "outputs/model_development"
  save_step_results: true      # Save intermediate parquet files
  save_model: true             # Save XGBoost model as JSON
  save_split_indices: true     # Save train/test/oot assignment
  generate_excel: true         # Generate final Excel report
  save_correlation_matrix: true  # Save full correlation matrix (can be large)

reproducibility:
  global_seed: 42
  save_config: true       # Always save config snapshot in run directory
  save_metadata: true     # Save run metadata (git hash, versions, etc.)
  log_level: "DEBUG"      # DEBUG, INFO, WARNING, ERROR
```

---

## Coordination Rules

1. **architect builds first**: Config system, base classes, output manager must be ready before other teammates can integrate. Architect should complete tasks 1-4 first, then signal readiness.
2. **scientist depends on architect**: Needs base class and config schema to implement components. Can start designing components while waiting but must use the final base class interface.
3. **risk-expert depends on scientist**: Needs component outputs to build validation checks and reporting. Can start on methodology docs and report template independently.
4. **tester depends on architect + scientist**: Needs components to exist before writing tests. Can start on conftest.py and test fixtures immediately.
5. **mlops depends on architect + scientist**: Needs config system and components for notebook. Can start on Makefile, pyproject.toml, .gitignore, and experiment tracker immediately.

## Quality Standards

- Type hints on all public functions
- Docstrings on all public classes and methods (Google style)
- No hardcoded magic numbers — everything in config
- All imports at file top level
- No circular imports between modules
- Consistent naming: snake_case for functions/variables, PascalCase for classes
- Parquet for data artifacts (not CSV) — faster, smaller, typed
- Log step-level timing and feature counts at INFO level
- Handle edge cases gracefully: empty DataFrames, single-feature scenarios, zero bad rate quarters

## What NOT to Change

- `src/features/` — Feature factory is frozen, not part of this overhaul
- `scripts/spark_feature_extraction.py` — Spark job is frozen
- `scripts/dataproc_feature_job.py` — GCP Dataproc job is frozen
- `data/` — Sample data stays as-is
- `config/base_config.yaml`, `config/data_config.yaml`, `config/feature_config.yaml` — These are for feature extraction, not model development

Wait for all teammates to complete their plans before approving any. Review plans to ensure no file conflicts and consistent interfaces across teammates.
