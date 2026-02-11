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
    threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    method: Literal["pearson", "spearman", "kendall"] = "pearson"


class SelectionConfig(BaseModel):
    """Forward feature selection configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    method: Literal["forward"] = "forward"
    auc_threshold: float = Field(default=0.0001, ge=0.0)
    max_features: Optional[int] = None


class StepsConfig(BaseModel):
    """All pipeline step configurations."""

    model_config = {"frozen": True}

    constant: ConstantConfig = Field(default_factory=ConstantConfig)
    missing: MissingConfig = Field(default_factory=MissingConfig)
    iv: IVConfig = Field(default_factory=IVConfig)
    psi: PSIConfig = Field(default_factory=PSIConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)


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


class EvaluationConfig(BaseModel):
    """Model evaluation configuration."""

    model_config = {"frozen": True}

    metrics: List[str] = Field(default_factory=lambda: ["auc", "gini", "ks"])
    precision_at_k: List[int] = Field(default_factory=lambda: [5, 10, 20])
    n_deciles: int = Field(default=10, ge=2)
    calculate_score_psi: bool = True


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


class ReproducibilityConfig(BaseModel):
    """Reproducibility and logging configuration."""

    model_config = {"frozen": True}

    global_seed: int = 42
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
