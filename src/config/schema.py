"""
Pydantic Configuration Schema

Defines all configuration models for the model development pipeline.
All fields have sensible defaults matching the existing pipeline behavior.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Data source configuration."""

    model_config = {"frozen": True}

    input_path: str = "data/sample/sample_features.parquet"
    target_column: str = "target"
    date_column: str = "date"
    id_columns: List[str] = Field(default_factory=lambda: ["application_id", "customer_id"])
    exclude_columns: List[str] = Field(default_factory=lambda: ["applicant_type"])


class SplittingConfig(BaseModel):
    """Train/test/OOT splitting configuration."""

    model_config = {"frozen": True}

    train_end_date: str = "2024-06-30"
    test_size: float = Field(default=0.20, ge=0.0, le=1.0)
    stratify: bool = True


class ConstantConfig(BaseModel):
    """Constant feature elimination configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    min_unique_values: int = Field(default=2, ge=1)


class MissingConfig(BaseModel):
    """Missing value elimination configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class IVConfig(BaseModel):
    """Information Value elimination configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    min_iv: float = Field(default=0.02, ge=0.0)
    max_iv: float = Field(default=0.50, ge=0.0)
    n_bins: int = Field(default=10, ge=2)
    min_samples_per_bin: int = Field(default=50, ge=1)

    @model_validator(mode="after")
    def iv_range_valid(self) -> "IVConfig":
        if self.min_iv >= self.max_iv:
            raise ValueError(
                f"min_iv ({self.min_iv}) must be less than max_iv ({self.max_iv})"
            )
        return self


class PSICheckConfig(BaseModel):
    """Single PSI check configuration."""

    model_config = {"frozen": True}

    type: str
    date: Optional[str] = None
    label: Optional[str] = None


class PSIConfig(BaseModel):
    """PSI stability elimination configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    threshold: float = Field(default=0.25, ge=0.0)
    n_bins: int = Field(default=10, ge=2)
    checks: List[PSICheckConfig] = Field(
        default_factory=lambda: [
            PSICheckConfig(type="quarterly"),
            PSICheckConfig(type="yearly"),
            PSICheckConfig(type="consecutive"),
        ]
    )


class CorrelationConfig(BaseModel):
    """Correlation elimination configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    method: Literal["pearson", "spearman", "kendall"] = "pearson"


class SelectionConfig(BaseModel):
    """Sequential feature selection configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    method: Literal["forward", "backward"] = "forward"
    cv: int = Field(default=5, ge=2)
    max_features: int = Field(default=20, ge=1)
    min_features: int = Field(default=1, ge=1)
    tolerance: float = Field(default=0.001, ge=0.0)
    patience: int = Field(default=3, ge=1)


class VIFConfig(BaseModel):
    """Variance Inflation Factor elimination configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    threshold: float = Field(default=5.0, ge=1.0)
    iv_aware: bool = True


class TemporalFilterConfig(BaseModel):
    """Time-dependent variable performance filtering configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    min_quarterly_auc: float = Field(default=0.52, ge=0.50, le=1.0)
    max_auc_degradation: float = Field(default=0.05, ge=0.0)
    min_trend_slope: float = Field(default=-0.02)


class StepsConfig(BaseModel):
    """All pipeline step configurations."""

    model_config = {"frozen": True}

    constant: ConstantConfig = Field(default_factory=ConstantConfig)
    missing: MissingConfig = Field(default_factory=MissingConfig)
    iv: IVConfig = Field(default_factory=IVConfig)
    psi: PSIConfig = Field(default_factory=PSIConfig)
    temporal_filter: TemporalFilterConfig = Field(default_factory=TemporalFilterConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    vif: VIFConfig = Field(default_factory=VIFConfig)


class TuningConfig(BaseModel):
    """Hyperparameter tuning configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    method: Literal["optuna"] = "optuna"
    n_trials: int = Field(default=100, ge=1)
    timeout: Optional[int] = None
    cv: int = Field(default=5, ge=2)
    stability_weight: float = Field(default=1.0, ge=0.0)


class ModelConfig(BaseModel):
    """ML model configuration."""

    model_config = {"frozen": True}

    algorithm: Literal["xgboost"] = "xgboost"
    params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "early_stopping_rounds": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
        }
    )
    tuning: TuningConfig = Field(default_factory=TuningConfig)


class CalibrationConfig(BaseModel):
    """Calibration configuration."""
    model_config = {"frozen": True}
    enabled: bool = True
    method: Literal["platt", "isotonic", "temperature"] = "platt"


class SHAPConfig(BaseModel):
    """SHAP interpretability configuration."""
    model_config = {"frozen": True}
    enabled: bool = True
    max_samples: int = Field(default=500, ge=10)


class BootstrapConfig(BaseModel):
    """Bootstrap confidence interval configuration."""
    model_config = {"frozen": True}
    enabled: bool = True
    n_iterations: int = Field(default=1000, ge=100)
    confidence_level: float = Field(default=0.95, ge=0.50, le=0.99)


class SubsegmentConfig(BaseModel):
    """Subsegment analysis configuration."""

    model_config = {"frozen": True}

    enabled: bool = False
    columns: List[str] = Field(default_factory=list)  # e.g., ["applicant_type"]


class ConfusionMatrixConfig(BaseModel):
    """Confusion matrix configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    thresholds: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])


class EvaluationConfig(BaseModel):
    """Model evaluation configuration."""

    model_config = {"frozen": True}

    metrics: List[str] = Field(default_factory=lambda: ["auc", "gini", "ks"])
    precision_at_k: List[int] = Field(default_factory=lambda: [5, 10, 20])
    n_deciles: int = Field(default=10, ge=2)
    calculate_score_psi: bool = True
    importance_type: Literal["gain", "weight", "cover", "total_gain", "total_cover"] = "gain"
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    shap: SHAPConfig = Field(default_factory=SHAPConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    subsegment: SubsegmentConfig = Field(default_factory=SubsegmentConfig)
    confusion_matrix: ConfusionMatrixConfig = Field(default_factory=ConfusionMatrixConfig)


class ValidationChecksConfig(BaseModel):
    """Validation check thresholds."""

    model_config = {"frozen": True}

    min_auc: float = Field(default=0.65, ge=0.0, le=1.0)
    max_overfit_gap: float = Field(default=0.05, ge=0.0)
    max_oot_degradation: float = Field(default=0.08, ge=0.0)
    max_score_psi: float = Field(default=0.25, ge=0.0)
    max_feature_concentration: float = Field(default=0.50, ge=0.0, le=1.0)
    min_oot_samples: int = Field(default=30, ge=1)
    check_monotonicity: bool = True


class ValidationConfig(BaseModel):
    """Validation configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    checks: ValidationChecksConfig = Field(default_factory=ValidationChecksConfig)


class OutputConfig(BaseModel):
    """Output configuration."""

    model_config = {"frozen": True}

    base_dir: str = "outputs/model_development"
    save_step_results: bool = True
    save_model: bool = True
    save_split_indices: bool = True
    generate_excel: bool = True
    save_correlation_matrix: bool = True
    generate_docs: bool = True


class ReproducibilityConfig(BaseModel):
    """Reproducibility and logging configuration."""

    model_config = {"frozen": True}

    global_seed: int = 42
    n_jobs: int = Field(default=-1, ge=-1)
    save_config: bool = True
    save_metadata: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration combining all sections."""

    model_config = {"frozen": True}

    data: DataConfig = Field(default_factory=DataConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    steps: StepsConfig = Field(default_factory=StepsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


# ===================================================================
# Classic Model Configuration (WoE + LogReg + Scorecard)
# ===================================================================

class WoEConfig(BaseModel):
    """Weight of Evidence binning configuration."""

    model_config = {"frozen": True}

    n_bins: int = Field(default=10, ge=2)
    min_bin_size: float = Field(default=0.05, ge=0.01, le=0.5)
    monotonic: bool = True
    missing_bin: bool = True
    min_iv: float = Field(default=0.02, ge=0.0)
    max_iv: float = Field(default=0.50, ge=0.0)


class LogisticConfig(BaseModel):
    """Logistic regression model configuration."""

    model_config = {"frozen": True}

    solver: Literal["lbfgs", "liblinear", "saga"] = "lbfgs"
    penalty: Literal["l1", "l2", "none"] = "l2"
    C: float = Field(default=1.0, gt=0.0)
    max_iter: int = Field(default=1000, ge=100)
    class_weight: Optional[str] = "balanced"


class ScorecardConfig(BaseModel):
    """Scorecard generation configuration."""

    model_config = {"frozen": True}

    target_score: int = Field(default=600, ge=0)
    target_odds: float = Field(default=20.0, gt=0.0)
    pdo: float = Field(default=50.0, gt=0.0)


class ClassicPipelineConfig(BaseModel):
    """Top-level configuration for the classic credit risk model pipeline."""

    model_config = {"frozen": True}

    data: DataConfig = Field(default_factory=DataConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    woe: WoEConfig = Field(default_factory=WoEConfig)
    logistic: LogisticConfig = Field(default_factory=LogisticConfig)
    scorecard: ScorecardConfig = Field(default_factory=ScorecardConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
