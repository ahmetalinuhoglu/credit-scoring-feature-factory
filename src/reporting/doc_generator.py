"""
Markdown Report Generator

Generates run-specific Markdown documentation from pipeline results.
Uses Jinja2 for template rendering when available, with a fallback
to simple string formatting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import Jinja2; set a flag for fallback mode
try:
    import jinja2

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.info("Jinja2 not available; using fallback string formatting.")


# ──────────────────────────────────────────────────────────────────
# Helper dataclasses for template context
# ──────────────────────────────────────────────────────────────────

@dataclass
class FeatureInfo:
    """A selected feature with optional IV score."""

    name: str
    iv: Optional[float] = None


@dataclass
class PerformanceRow:
    """A single row in the performance table."""

    period: str
    auc: Optional[float] = None
    gini: Optional[float] = None
    ks: Optional[float] = None
    samples: Optional[int] = None
    bads: Optional[int] = None


@dataclass
class QuarterlyTrendRow:
    """A single row in the quarterly trend table."""

    period: str
    auc: Optional[float] = None
    auc_change: Optional[float] = None
    gini: Optional[float] = None
    gini_change: Optional[float] = None


@dataclass
class ValidationCheck:
    """A single validation check result."""

    name: str
    severity: str = ""
    result: str = ""
    details: str = ""


@dataclass
class ConfigItem:
    """A key-value configuration setting for display."""

    section: str
    param: str
    value: str


@dataclass
class OOTMetric:
    """AUC/Gini for a single OOT period."""

    auc: Optional[float] = None
    gini: Optional[float] = None


# ──────────────────────────────────────────────────────────────────
# Main generator class
# ──────────────────────────────────────────────────────────────────

class MarkdownReportGenerator:
    """Generate Markdown documentation from pipeline run results.

    Uses a Jinja2 template located in ``template_dir`` when Jinja2 is
    installed.  Falls back to a simple string-formatted report otherwise.

    Args:
        template_dir: Path to the directory containing Jinja2 templates.
            Defaults to ``docs/templates`` relative to the project root.
    """

    def __init__(self, template_dir: str = "docs/templates"):
        self.template_dir = Path(template_dir)

    # ── Public API ────────────────────────────────────────────────

    def generate(
        self,
        results: dict,
        output_dir: str,
        config: Any = None,
    ) -> str:
        """Generate a Markdown report from pipeline results.

        Args:
            results: Dict returned by ``ModelDevelopmentPipeline.run()``.
                Expected keys (all optional, missing keys handled gracefully):
                    run_id, input_path, train_end_date, total_features,
                    after_constant, after_missing, after_iv, after_psi,
                    after_temporal, after_correlation, after_selection,
                    after_vif, selected_features, tuning_best_params,
                    tuning_n_trials, AUC_Train, AUC_Test, Gini_Train,
                    Gini_Test, excel_path, log_file, status,
                    quarterly_trend_df (optional),
                    has_critical_failures (optional).
            output_dir: Directory to write the generated report to.
            config: Optional ``PipelineConfig`` instance for settings info.

        Returns:
            Absolute path to the generated Markdown file.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        context = self._build_context(results, config)
        run_id = results.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_filename = f"run_report_{run_id}.md"
        report_path = out_path / report_filename

        if JINJA2_AVAILABLE and self._template_exists():
            content = self._render_jinja2(context)
        else:
            content = self._render_fallback(context)

        report_path.write_text(content, encoding="utf-8")
        logger.info("Markdown report generated: %s", report_path)
        return str(report_path)

    # ── Context building ──────────────────────────────────────────

    def _build_context(self, results: dict, config: Any = None) -> dict:
        """Build the full template context dict from raw results.

        All keys are safe-accessed with ``.get()`` so that missing data
        produces ``None`` rather than raising ``KeyError``.
        """
        ctx: Dict[str, Any] = {}

        # -- Section 1: Executive summary ---------------------
        ctx["run_id"] = results.get("run_id", "unknown")
        ctx["run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ctx["status"] = results.get("status", "unknown")
        ctx["input_path"] = results.get("input_path", "N/A")
        ctx["total_features"] = _safe_int(results.get("total_features"))
        ctx["after_selection"] = _safe_int(results.get("after_selection"))
        ctx["after_vif"] = _safe_int(results.get("after_vif"))
        ctx["auc_train"] = results.get("AUC_Train")
        ctx["auc_test"] = results.get("AUC_Test")
        ctx["gini_train"] = results.get("Gini_Train")
        ctx["gini_test"] = results.get("Gini_Test")
        ctx["has_critical_failures"] = results.get("has_critical_failures")

        # Tuning
        ctx["tuning_enabled"] = results.get("tuning_best_params") is not None
        ctx["tuning_n_trials"] = results.get("tuning_n_trials", 0)

        # -- Section 2: Data description ----------------------
        ctx["train_end_date"] = results.get("train_end_date", "N/A")
        ctx["train_period"] = results.get("train_period")
        ctx["train_rows"] = results.get("train_rows")
        ctx["test_rows"] = results.get("test_rows")
        ctx["train_bad_rate"] = results.get("train_bad_rate")
        ctx["test_bad_rate"] = results.get("test_bad_rate")
        ctx["oot_periods"] = results.get("oot_periods")

        # -- Section 3: Variable funnel -----------------------
        ctx["after_constant"] = _safe_int(results.get("after_constant"))
        ctx["after_missing"] = _safe_int(results.get("after_missing"))
        ctx["after_iv"] = _safe_int(results.get("after_iv"))
        ctx["after_psi"] = _safe_int(results.get("after_psi"))
        ctx["after_temporal"] = _safe_int(results.get("after_temporal"))
        ctx["after_correlation"] = _safe_int(results.get("after_correlation"))

        # -- Section 4: Selected variables --------------------
        ctx["selected_features"] = self._build_feature_list(results)

        # -- Section 5: Model performance ---------------------
        ctx["performance_rows"] = self._build_performance_rows(results)
        ctx["oot_metrics"] = self._build_oot_metrics(results)

        # -- Section 6: Quarterly trend -----------------------
        ctx["quarterly_trend_rows"] = self._build_quarterly_trend(results)

        # -- Section 7: Validation ----------------------------
        ctx["validation_checks"] = self._build_validation_checks(results)

        # -- Section 8: Configuration -------------------------
        ctx["config_settings"] = self._build_config_settings(results, config)

        # -- Artifacts ----------------------------------------
        ctx["excel_path"] = results.get("excel_path")
        ctx["log_file"] = results.get("log_file")
        ctx["chart_path"] = results.get("chart_path")
        ctx["model_path"] = results.get("model_path")

        return ctx

    # ── Selected features ─────────────────────────────────────────

    def _build_feature_list(self, results: dict) -> List[FeatureInfo]:
        """Build the list of selected features with IV scores."""
        raw_features = results.get("selected_features", [])
        if not raw_features:
            return []

        # If the features are simple strings
        if isinstance(raw_features, list) and all(
            isinstance(f, str) for f in raw_features
        ):
            # Try to extract IV scores from iv_scores or shap_summary
            iv_scores = results.get("iv_scores", {})
            return [
                FeatureInfo(name=f, iv=iv_scores.get(f))
                for f in raw_features
            ]

        # If already FeatureInfo-like dicts
        return [
            FeatureInfo(
                name=f.get("name", str(f)),
                iv=f.get("iv"),
            )
            if isinstance(f, dict)
            else FeatureInfo(name=str(f))
            for f in raw_features
        ]

    # ── Performance rows ──────────────────────────────────────────

    def _build_performance_rows(self, results: dict) -> List[PerformanceRow]:
        """Build performance rows from results.

        Attempts to use a ``performance_df`` DataFrame if present,
        otherwise falls back to extracting AUC_*/Gini_* keys.
        """
        perf_df = results.get("performance_df")
        if perf_df is not None:
            try:
                rows = []
                for _, row in perf_df.iterrows():
                    rows.append(
                        PerformanceRow(
                            period=str(row.get("Period", "")),
                            auc=_safe_float(row.get("AUC")),
                            gini=_safe_float(row.get("Gini")),
                            ks=_safe_float(row.get("KS")),
                            samples=_safe_int(row.get("N")),
                            bads=_safe_int(row.get("Bads")),
                        )
                    )
                return rows
            except Exception:
                pass
        return []

    # ── OOT metrics fallback ──────────────────────────────────────

    def _build_oot_metrics(self, results: dict) -> Dict[str, OOTMetric]:
        """Extract OOT period metrics from AUC_OOT_* / Gini_OOT_* keys."""
        oot: Dict[str, OOTMetric] = {}
        for key, value in results.items():
            if key.startswith("AUC_OOT_"):
                label = key.replace("AUC_", "")
                if label not in oot:
                    oot[label] = OOTMetric()
                oot[label].auc = _safe_float(value)
            elif key.startswith("Gini_OOT_"):
                label = key.replace("Gini_", "")
                if label not in oot:
                    oot[label] = OOTMetric()
                oot[label].gini = _safe_float(value)
        return oot

    # ── Quarterly trend ───────────────────────────────────────────

    def _build_quarterly_trend(self, results: dict) -> List[QuarterlyTrendRow]:
        """Build quarterly trend rows from quarterly_trend_df."""
        trend_df = results.get("quarterly_trend_df")
        if trend_df is None:
            return []

        try:
            rows = []
            for _, row in trend_df.iterrows():
                rows.append(
                    QuarterlyTrendRow(
                        period=str(row.get("Period", "")),
                        auc=_safe_float(row.get("AUC")),
                        auc_change=_safe_float(row.get("AUC_Change")),
                        gini=_safe_float(row.get("Gini")),
                        gini_change=_safe_float(row.get("Gini_Change")),
                    )
                )
            return rows
        except Exception:
            return []

    # ── Validation checks ─────────────────────────────────────────

    def _build_validation_checks(self, results: dict) -> List[ValidationCheck]:
        """Build validation check rows from validation_report DataFrame."""
        val_df = results.get("validation_report")
        if val_df is None:
            return []

        try:
            checks = []
            for _, row in val_df.iterrows():
                checks.append(
                    ValidationCheck(
                        name=str(row.get("Check", "")),
                        severity=str(row.get("Severity", "")),
                        result=str(row.get("Result", "")),
                        details=str(row.get("Details", "")),
                    )
                )
            return checks
        except Exception:
            return []

    # ── Configuration settings ────────────────────────────────────

    def _build_config_settings(
        self, results: dict, config: Any = None
    ) -> List[ConfigItem]:
        """Extract key configuration settings for display.

        Reads from the ``PipelineConfig`` object when available, otherwise
        falls back to extracting known keys from results.
        """
        items: List[ConfigItem] = []

        if config is not None:
            try:
                items.extend(self._config_from_pydantic(config))
                return items
            except Exception:
                pass

        # Fallback: extract what we can from results dict
        _add = items.append
        if results.get("input_path"):
            _add(ConfigItem("data", "input_path", str(results["input_path"])))
        if results.get("train_end_date"):
            _add(ConfigItem("splitting", "train_end_date", str(results["train_end_date"])))
        if results.get("tuning_n_trials"):
            _add(ConfigItem("model.tuning", "n_trials", str(results["tuning_n_trials"])))

        return items

    def _config_from_pydantic(self, config: Any) -> List[ConfigItem]:
        """Extract settings from a PipelineConfig Pydantic model."""
        items: List[ConfigItem] = []

        # Data
        items.append(ConfigItem("data", "input_path", str(config.data.input_path)))
        items.append(ConfigItem("data", "target_column", str(config.data.target_column)))
        items.append(ConfigItem("data", "date_column", str(config.data.date_column)))

        # Splitting
        items.append(
            ConfigItem("splitting", "train_end_date", str(config.splitting.train_end_date))
        )
        items.append(ConfigItem("splitting", "test_size", str(config.splitting.test_size)))

        # Steps
        items.append(ConfigItem("steps.iv", "min_iv", str(config.steps.iv.min_iv)))
        items.append(ConfigItem("steps.iv", "max_iv", str(config.steps.iv.max_iv)))
        items.append(
            ConfigItem("steps.missing", "threshold", str(config.steps.missing.threshold))
        )
        items.append(
            ConfigItem("steps.psi", "threshold", str(config.steps.psi.threshold))
        )
        items.append(
            ConfigItem(
                "steps.correlation", "threshold", str(config.steps.correlation.threshold)
            )
        )
        items.append(
            ConfigItem("steps.selection", "method", str(config.steps.selection.method))
        )
        items.append(
            ConfigItem(
                "steps.selection", "max_features", str(config.steps.selection.max_features)
            )
        )
        items.append(
            ConfigItem("steps.vif", "enabled", str(config.steps.vif.enabled))
        )
        items.append(
            ConfigItem("steps.vif", "threshold", str(config.steps.vif.threshold))
        )

        # Tuning
        items.append(
            ConfigItem("model.tuning", "enabled", str(config.model.tuning.enabled))
        )
        items.append(
            ConfigItem("model.tuning", "n_trials", str(config.model.tuning.n_trials))
        )

        # Reproducibility
        items.append(
            ConfigItem(
                "reproducibility", "global_seed", str(config.reproducibility.global_seed)
            )
        )
        items.append(
            ConfigItem("reproducibility", "n_jobs", str(config.reproducibility.n_jobs))
        )

        return items

    # ── Rendering ─────────────────────────────────────────────────

    def _template_exists(self) -> bool:
        """Check whether the Jinja2 template file exists."""
        template_path = self.template_dir / "run_report.md.j2"
        return template_path.is_file()

    def _render_jinja2(self, context: dict) -> str:
        """Render the report using Jinja2."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            undefined=jinja2.Undefined,
            keep_trailing_newline=True,
        )
        template = env.get_template("run_report.md.j2")
        return template.render(**context)

    def _render_fallback(self, context: dict) -> str:
        """Render the report using plain string formatting (no Jinja2)."""
        lines: List[str] = []
        _a = lines.append

        _a("# Model Development Run Report")
        _a("")
        _a(f"**Run ID:** {context.get('run_id', 'unknown')}")
        _a(f"**Date:** {context.get('run_date', 'N/A')}")
        _a(f"**Status:** {context.get('status', 'unknown')}")
        _a("")
        _a("---")
        _a("")

        # Section 1: Executive Summary
        _a("## 1. Executive Summary")
        _a("")
        _a("| Metric | Value |")
        _a("|--------|-------|")
        _a(f"| Status | {context.get('status', 'unknown')} |")
        _a(f"| Run Date | {context.get('run_date', 'N/A')} |")
        _a(f"| Input File | {context.get('input_path', 'N/A')} |")
        _a(f"| Total Features | {_fmt_int(context.get('total_features'))} |")
        _a(f"| Selected Features | {_fmt_int(context.get('after_selection'))} |")
        _a(f"| Final Features (after VIF) | {_fmt_int(context.get('after_vif'))} |")
        if context.get("auc_train") is not None:
            _a(f"| Train AUC | {context['auc_train']:.4f} |")
        if context.get("auc_test") is not None:
            _a(f"| Test AUC | {context['auc_test']:.4f} |")
        if context.get("gini_train") is not None:
            _a(f"| Train Gini | {context['gini_train']:.4f} |")
        if context.get("gini_test") is not None:
            _a(f"| Test Gini | {context['gini_test']:.4f} |")
        _a("")

        # Section 2: Data Description
        _a("---")
        _a("")
        _a("## 2. Data Description")
        _a("")
        _a("| Property | Value |")
        _a("|----------|-------|")
        _a(f"| Input File | `{context.get('input_path', 'N/A')}` |")
        _a(f"| Train End Date | {context.get('train_end_date', 'N/A')} |")
        if context.get("train_rows") is not None:
            _a(f"| Train Rows | {context['train_rows']} |")
        if context.get("test_rows") is not None:
            _a(f"| Test Rows | {context['test_rows']} |")
        _a("")

        # Section 3: Variable Funnel
        _a("---")
        _a("")
        _a("## 3. Variable Elimination Funnel")
        _a("")
        _a("| Step | Features Remaining | Eliminated |")
        _a("|------|--------------------|------------|")
        total = context.get("total_features")
        _a(f"| Initial | {_fmt_int(total)} | -- |")
        prev = total
        for step_key, step_label in [
            ("after_constant", "Constant Elimination"),
            ("after_missing", "Missing Elimination"),
            ("after_iv", "IV Elimination"),
            ("after_psi", "PSI Elimination"),
            ("after_temporal", "Temporal Filter"),
            ("after_correlation", "Correlation Elimination"),
            ("after_selection", "Sequential Selection"),
            ("after_vif", "VIF Check"),
        ]:
            val = context.get(step_key)
            if val is not None and prev is not None:
                _a(f"| {step_label} | {val} | {prev - val} |")
                prev = val
        _a("")

        # Section 4: Selected Variables
        _a("---")
        _a("")
        _a("## 4. Selected Variables")
        _a("")
        features = context.get("selected_features", [])
        if features:
            _a("| # | Feature | IV Score |")
            _a("|---|---------|----------|")
            for i, feat in enumerate(features, 1):
                iv_str = f"{feat.iv:.4f}" if feat.iv is not None else "N/A"
                _a(f"| {i} | `{feat.name}` | {iv_str} |")
        else:
            _a("No feature details available.")
        _a("")

        # Section 5: Model Performance
        _a("---")
        _a("")
        _a("## 5. Model Performance")
        _a("")
        perf_rows = context.get("performance_rows", [])
        if perf_rows:
            _a("| Period | AUC | Gini | KS | Samples | Bads |")
            _a("|--------|-----|------|----|---------|------|")
            for row in perf_rows:
                auc_s = f"{row.auc:.4f}" if row.auc is not None else "N/A"
                gini_s = f"{row.gini:.4f}" if row.gini is not None else "N/A"
                ks_s = f"{row.ks:.4f}" if row.ks is not None else "N/A"
                samples_s = str(row.samples) if row.samples is not None else "N/A"
                bads_s = str(row.bads) if row.bads is not None else "N/A"
                _a(f"| {row.period} | {auc_s} | {gini_s} | {ks_s} | {samples_s} | {bads_s} |")
        else:
            _a("| Period | AUC | Gini |")
            _a("|--------|-----|------|")
            if context.get("auc_train") is not None:
                gini_s = f"{context['gini_train']:.4f}" if context.get("gini_train") else "N/A"
                _a(f"| Train | {context['auc_train']:.4f} | {gini_s} |")
            if context.get("auc_test") is not None:
                gini_s = f"{context['gini_test']:.4f}" if context.get("gini_test") else "N/A"
                _a(f"| Test | {context['auc_test']:.4f} | {gini_s} |")
        _a("")

        # Section 6: Quarterly Trend
        _a("---")
        _a("")
        _a("## 6. Quarterly Trend")
        _a("")
        trend_rows = context.get("quarterly_trend_rows", [])
        if trend_rows:
            _a("| Period | AUC | AUC Change | Gini | Gini Change |")
            _a("|--------|-----|------------|------|-------------|")
            for row in trend_rows:
                auc_s = f"{row.auc:.4f}" if row.auc is not None else "N/A"
                auc_c = f"{row.auc_change:+.4f}" if row.auc_change is not None else "--"
                gini_s = f"{row.gini:.4f}" if row.gini is not None else "N/A"
                gini_c = f"{row.gini_change:+.4f}" if row.gini_change is not None else "--"
                _a(f"| {row.period} | {auc_s} | {auc_c} | {gini_s} | {gini_c} |")
        else:
            _a("No quarterly trend data available.")
        _a("")

        # Section 7: Validation
        _a("---")
        _a("")
        _a("## 7. Validation Results")
        _a("")
        if context.get("has_critical_failures") is not None:
            if context["has_critical_failures"]:
                _a("**Overall Status:** FAILED -- Critical issues detected")
            else:
                _a("**Overall Status:** PASSED -- No critical failures")
            _a("")
            checks = context.get("validation_checks", [])
            if checks:
                _a("| Check | Severity | Result | Details |")
                _a("|-------|----------|--------|---------|")
                for check in checks:
                    _a(
                        f"| {check.name} | {check.severity} "
                        f"| {check.result} | {check.details} |"
                    )
        else:
            _a("Validation was not executed for this run.")
        _a("")

        # Section 8: Configuration
        _a("---")
        _a("")
        _a("## 8. Configuration Summary")
        _a("")
        cfg_items = context.get("config_settings", [])
        if cfg_items:
            _a("| Section | Parameter | Value |")
            _a("|---------|-----------|-------|")
            for item in cfg_items:
                _a(f"| {item.section} | {item.param} | {item.value} |")
        else:
            _a("No configuration details available.")
        _a("")

        # Artifacts
        _a("---")
        _a("")
        _a("## Artifacts")
        _a("")
        _a("| Artifact | Path |")
        _a("|----------|------|")
        if context.get("excel_path"):
            _a(f"| Excel Report | `{context['excel_path']}` |")
        if context.get("log_file"):
            _a(f"| Log File | `{context['log_file']}` |")
        if context.get("chart_path"):
            _a(f"| Selection Chart | `{context['chart_path']}` |")
        if context.get("model_path"):
            _a(f"| Model Artifact | `{context['model_path']}` |")
        _a("")
        _a("---")
        _a("")
        _a("*Generated by MarkdownReportGenerator*")
        _a("")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert a value to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_int(value: Any) -> str:
    """Format an int for display, returning 'N/A' for None."""
    if value is None:
        return "N/A"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "N/A"
