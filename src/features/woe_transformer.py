"""
Weight of Evidence (WoE) Transformer

Implements binning and WoE encoding for credit scoring.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.core.base import PandasComponent
from src.core.exceptions import FeatureEngineeringError


@dataclass
class WoEBin:
    """Single bin statistics."""
    bin_id: int
    lower_bound: float
    upper_bound: float
    count: int
    good_count: int
    bad_count: int
    woe: float
    iv_contribution: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bin_id': self.bin_id,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'count': self.count,
            'good_count': self.good_count,
            'bad_count': self.bad_count,
            'woe': self.woe,
            'iv_contribution': self.iv_contribution
        }


@dataclass
class WoEFeatureResult:
    """WoE binning result for a single feature."""
    feature_name: str
    bins: List[WoEBin]
    iv_score: float
    is_monotonic: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'bins': [b.to_dict() for b in self.bins],
            'iv_score': self.iv_score,
            'is_monotonic': self.is_monotonic
        }


class WoETransformer(PandasComponent):
    """
    Weight of Evidence transformer for credit scoring.
    
    Features:
    - Automatic optimal binning (monotonic WoE)
    - Manual bin specification
    - WoE encoding of features
    - IV calculation per feature
    - Bin statistics export
    
    WoE Formula:
        WoE_i = ln(Distribution of Goods_i / Distribution of Bads_i)
        
    IV Formula:
        IV = Σ (% Goods_i - % Bads_i) * WoE_i
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None
    ):
        """
        Initialize WoE transformer.
        
        Args:
            config: Configuration dictionary
            name: Optional transformer name
        """
        super().__init__(config, name or "WoETransformer")
        
        # WoE binning settings
        woe_config = self.get_config('model.logistic_regression.woe_binning', {})
        self.n_bins = woe_config.get('n_bins', 10)
        self.min_bin_size = woe_config.get('min_bin_size', 0.05)
        self.monotonic = woe_config.get('monotonic', True)
        self.missing_bin = woe_config.get('missing_bin', True)
        
        # Small constant to avoid log(0)
        self.epsilon = 1e-6
        
        # Fitted binning results
        self._binning_results: Dict[str, WoEFeatureResult] = {}
        self._is_fitted = False
    
    def validate(self) -> bool:
        """Validate transformer configuration."""
        if self.n_bins < 2:
            self.logger.error("n_bins must be at least 2")
            return False
        if not 0 < self.min_bin_size < 0.5:
            self.logger.error("min_bin_size must be between 0 and 0.5")
            return False
        return True
    
    def run(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """
        Run the WoE transformation (satisfies abstract method requirement).
        
        This is a wrapper around fit_transform() to comply with PipelineComponent interface.
        
        Args:
            df: DataFrame with features and target
            features: List of feature columns to bin
            target_column: Name of binary target column
            
        Returns:
            DataFrame with WoE-encoded features
        """
        return self.fit_transform(df, features, target_column)
    
    def fit(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_column: str = 'target'
    ) -> 'WoETransformer':
        """
        Fit WoE binning on training data.
        
        Args:
            df: Training DataFrame
            features: List of feature columns to bin
            target_column: Name of binary target column
            
        Returns:
            Self
        """
        self._start_execution()
        
        try:
            # Validate target
            if target_column not in df.columns:
                raise FeatureEngineeringError(
                    f"Target column '{target_column}' not found",
                    feature_name=target_column
                )
            
            # Calculate total goods and bads
            total_goods = (df[target_column] == 0).sum()
            total_bads = (df[target_column] == 1).sum()
            
            if total_goods == 0 or total_bads == 0:
                raise FeatureEngineeringError(
                    "Target must have both classes (0 and 1)",
                    feature_name=target_column
                )
            
            self.logger.info(f"Fitting WoE on {len(features)} features")
            self.logger.info(f"Total: {len(df)} rows, {total_goods} goods, {total_bads} bads")
            
            # Fit each feature
            for feature in features:
                if feature not in df.columns:
                    self.logger.warning(f"Feature '{feature}' not found, skipping")
                    continue
                
                try:
                    result = self._fit_feature(
                        df, feature, target_column,
                        total_goods, total_bads
                    )
                    self._binning_results[feature] = result
                    self.logger.debug(
                        f"Feature '{feature}': IV={result.iv_score:.4f}, "
                        f"monotonic={result.is_monotonic}"
                    )
                except Exception as e:
                    self.logger.warning(f"Could not fit feature '{feature}': {e}")
            
            self._is_fitted = True
            self.logger.info(f"Fitted WoE for {len(self._binning_results)} features")
            
            self._end_execution()
            return self
            
        except Exception as e:
            self._end_execution()
            raise FeatureEngineeringError(
                f"WoE fitting failed: {e}",
                cause=e
            )
    
    def _fit_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        target_column: str,
        total_goods: int,
        total_bads: int
    ) -> WoEFeatureResult:
        """Fit WoE binning for a single feature."""
        
        # Handle missing values
        data = df[[feature, target_column]].copy()
        has_missing = data[feature].isna().any()
        
        # Separate missing if configured
        if has_missing and self.missing_bin:
            missing_mask = data[feature].isna()
            data_missing = data[missing_mask]
            data = data[~missing_mask]
        else:
            data_missing = None
            data = data.dropna(subset=[feature])
        
        # Initial quantile binning
        try:
            data['bin'] = pd.qcut(
                data[feature], 
                q=self.n_bins, 
                labels=False, 
                duplicates='drop'
            )
        except ValueError:
            # Fall back to fewer bins if not enough unique values
            n_unique = data[feature].nunique()
            if n_unique < self.n_bins:
                data['bin'] = pd.qcut(
                    data[feature], 
                    q=min(n_unique, self.n_bins), 
                    labels=False, 
                    duplicates='drop'
                )
            else:
                raise
        
        # Get bin edges
        bins = []
        grouped = data.groupby('bin')
        
        for bin_id, group in sorted(grouped, key=lambda x: x[0]):
            count = len(group)
            good_count = (group[target_column] == 0).sum()
            bad_count = (group[target_column] == 1).sum()
            
            # Calculate WoE
            pct_goods = (good_count / total_goods) if total_goods > 0 else 0
            pct_bads = (bad_count / total_bads) if total_bads > 0 else 0
            
            # Add epsilon to avoid division by zero
            pct_goods = max(pct_goods, self.epsilon)
            pct_bads = max(pct_bads, self.epsilon)
            
            woe = np.log(pct_goods / pct_bads)
            iv_contribution = (pct_goods - pct_bads) * woe
            
            bin_obj = WoEBin(
                bin_id=int(bin_id),
                lower_bound=float(group[feature].min()),
                upper_bound=float(group[feature].max()),
                count=int(count),
                good_count=int(good_count),
                bad_count=int(bad_count),
                woe=float(woe),
                iv_contribution=float(iv_contribution)
            )
            bins.append(bin_obj)
        
        # Add missing bin if exists
        if data_missing is not None and len(data_missing) > 0:
            good_count = (data_missing[target_column] == 0).sum()
            bad_count = (data_missing[target_column] == 1).sum()
            
            pct_goods = max(good_count / total_goods, self.epsilon)
            pct_bads = max(bad_count / total_bads, self.epsilon)
            
            woe = np.log(pct_goods / pct_bads)
            iv_contribution = (pct_goods - pct_bads) * woe
            
            bins.append(WoEBin(
                bin_id=-1,  # Special bin for missing
                lower_bound=np.nan,
                upper_bound=np.nan,
                count=len(data_missing),
                good_count=int(good_count),
                bad_count=int(bad_count),
                woe=float(woe),
                iv_contribution=float(iv_contribution)
            ))
        
        # Merge small bins if needed
        bins = self._merge_small_bins(bins, total_goods, total_bads)
        
        # Make monotonic if configured
        if self.monotonic:
            bins = self._make_monotonic(bins, total_goods, total_bads)
        
        # Calculate total IV
        iv_score = sum(b.iv_contribution for b in bins)
        
        # Check monotonicity
        woe_values = [b.woe for b in bins if b.bin_id >= 0]
        is_monotonic = self._check_monotonic(woe_values)
        
        return WoEFeatureResult(
            feature_name=feature,
            bins=bins,
            iv_score=iv_score,
            is_monotonic=is_monotonic
        )
    
    def _merge_small_bins(
        self,
        bins: List[WoEBin],
        total_goods: int,
        total_bads: int
    ) -> List[WoEBin]:
        """Merge bins that are too small."""
        if len(bins) <= 2:
            return bins
        
        total_count = sum(b.count for b in bins if b.bin_id >= 0)
        min_count = int(total_count * self.min_bin_size)
        
        # Simple merge: combine with adjacent bin if too small
        merged = []
        for i, bin_obj in enumerate(bins):
            if bin_obj.bin_id < 0:  # Keep missing bin separate
                merged.append(bin_obj)
                continue
                
            if bin_obj.count < min_count and len(merged) > 0:
                # Merge with previous bin
                prev = merged[-1]
                if prev.bin_id >= 0:
                    merged[-1] = self._merge_two_bins(
                        prev, bin_obj, total_goods, total_bads
                    )
                    continue
            
            merged.append(bin_obj)
        
        return merged
    
    def _merge_two_bins(
        self,
        bin1: WoEBin,
        bin2: WoEBin,
        total_goods: int,
        total_bads: int
    ) -> WoEBin:
        """Merge two adjacent bins."""
        count = bin1.count + bin2.count
        good_count = bin1.good_count + bin2.good_count
        bad_count = bin1.bad_count + bin2.bad_count
        
        pct_goods = max(good_count / total_goods, self.epsilon)
        pct_bads = max(bad_count / total_bads, self.epsilon)
        
        woe = np.log(pct_goods / pct_bads)
        iv_contribution = (pct_goods - pct_bads) * woe
        
        return WoEBin(
            bin_id=bin1.bin_id,
            lower_bound=min(bin1.lower_bound, bin2.lower_bound),
            upper_bound=max(bin1.upper_bound, bin2.upper_bound),
            count=count,
            good_count=good_count,
            bad_count=bad_count,
            woe=woe,
            iv_contribution=iv_contribution
        )
    
    def _make_monotonic(
        self,
        bins: List[WoEBin],
        total_goods: int,
        total_bads: int
    ) -> List[WoEBin]:
        """Ensure WoE values are monotonic by merging adjacent non-monotonic bins."""
        # Separate missing bin
        regular_bins = [b for b in bins if b.bin_id >= 0]
        missing_bin = [b for b in bins if b.bin_id < 0]
        
        if len(regular_bins) <= 2:
            return bins
        
        # Sort by lower bound
        regular_bins.sort(key=lambda x: x.lower_bound)
        
        # Iteratively merge non-monotonic bins
        max_iterations = 100
        for _ in range(max_iterations):
            if self._check_monotonic([b.woe for b in regular_bins]):
                break
            
            # Find first non-monotonic pair and merge
            for i in range(len(regular_bins) - 1):
                if not self._is_monotonic_pair(
                    regular_bins[i].woe, 
                    regular_bins[i + 1].woe,
                    [b.woe for b in regular_bins]
                ):
                    merged = self._merge_two_bins(
                        regular_bins[i], 
                        regular_bins[i + 1],
                        total_goods, 
                        total_bads
                    )
                    regular_bins = regular_bins[:i] + [merged] + regular_bins[i+2:]
                    break
            
            if len(regular_bins) <= 2:
                break
        
        return regular_bins + missing_bin
    
    def _check_monotonic(self, values: List[float]) -> bool:
        """Check if values are monotonically increasing or decreasing."""
        if len(values) <= 1:
            return True
        
        increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        return increasing or decreasing
    
    def _is_monotonic_pair(self, woe1: float, woe2: float, all_woe: List[float]) -> bool:
        """Check if a pair maintains monotonicity direction."""
        if len(all_woe) <= 2:
            return True
        
        # Determine overall direction from first and last
        direction = all_woe[-1] - all_woe[0]
        
        if direction >= 0:  # Increasing
            return woe2 >= woe1
        else:  # Decreasing
            return woe2 <= woe1
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to WoE values.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with WoE-encoded features
        """
        if not self._is_fitted:
            raise FeatureEngineeringError(
                "Transformer not fitted. Call fit() first."
            )
        
        result = df.copy()
        
        for feature, binning in self._binning_results.items():
            if feature not in df.columns:
                continue
            
            woe_column = f"{feature}_woe"
            result[woe_column] = self._apply_woe(df[feature], binning)
        
        return result
    
    def _apply_woe(self, series: pd.Series, binning: WoEFeatureResult) -> pd.Series:
        """Apply WoE encoding to a series."""
        # Default to 0.0 (neutral WoE) for values outside known bins
        result = pd.Series(0.0, index=series.index, dtype=float)

        # Get sorted bins (excluding missing)
        regular_bins = sorted(
            [b for b in binning.bins if b.bin_id >= 0],
            key=lambda x: x.lower_bound
        )
        missing_bin = next((b for b in binning.bins if b.bin_id < 0), None)

        # Handle missing values
        if missing_bin:
            result[series.isna()] = missing_bin.woe
        else:
            result[series.isna()] = 0.0

        # Apply bins to non-missing values
        for bin_obj in regular_bins:
            mask = (series >= bin_obj.lower_bound) & (series <= bin_obj.upper_bound)
            result[mask] = bin_obj.woe

        # Values below min bin get the lowest bin's WoE
        if regular_bins:
            below_mask = (~series.isna()) & (series < regular_bins[0].lower_bound)
            result[below_mask] = regular_bins[0].woe
            # Values above max bin get the highest bin's WoE
            above_mask = (~series.isna()) & (series > regular_bins[-1].upper_bound)
            result[above_mask] = regular_bins[-1].woe

        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, features, target_column)
        return self.transform(df)
    
    def get_woe_table(self, feature: str) -> Optional[List[WoEBin]]:
        """Get WoE table for a specific feature."""
        if feature in self._binning_results:
            return self._binning_results[feature].bins
        return None
    
    def get_iv_summary(self) -> Dict[str, float]:
        """Get IV scores for all features."""
        return {
            name: result.iv_score 
            for name, result in self._binning_results.items()
        }
    
    def get_iv_category(self, iv: float) -> str:
        """Categorize IV score."""
        if iv < 0.02:
            return "useless"
        elif iv < 0.1:
            return "weak"
        elif iv < 0.3:
            return "medium"
        elif iv < 0.5:
            return "strong"
        else:
            return "suspicious"
    
    def export_binning(self, path: Union[str, Path]) -> None:
        """Export binning configuration to JSON file."""
        if not self._is_fitted:
            raise FeatureEngineeringError("Transformer not fitted")
        
        export_data = {
            'config': {
                'n_bins': self.n_bins,
                'min_bin_size': self.min_bin_size,
                'monotonic': self.monotonic,
                'missing_bin': self.missing_bin
            },
            'features': {
                name: result.to_dict()
                for name, result in self._binning_results.items()
            }
        }
        
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported binning to {path}")
    
    def load_binning(self, path: Union[str, Path]) -> None:
        """Load binning configuration from JSON file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Restore configuration
        config = data.get('config', {})
        self.n_bins = config.get('n_bins', self.n_bins)
        self.min_bin_size = config.get('min_bin_size', self.min_bin_size)
        self.monotonic = config.get('monotonic', self.monotonic)
        self.missing_bin = config.get('missing_bin', self.missing_bin)
        
        # Restore binning results
        self._binning_results = {}
        for feature_name, feature_data in data.get('features', {}).items():
            bins = [
                WoEBin(**bin_data)
                for bin_data in feature_data.get('bins', [])
            ]
            self._binning_results[feature_name] = WoEFeatureResult(
                feature_name=feature_name,
                bins=bins,
                iv_score=feature_data.get('iv_score', 0),
                is_monotonic=feature_data.get('is_monotonic', True)
            )
        
        self._is_fitted = True
        self.logger.info(f"Loaded binning from {path}")
    
    @property
    def is_fitted(self) -> bool:
        """Check if transformer is fitted."""
        return self._is_fitted
    
    @property
    def fitted_features(self) -> List[str]:
        """Get list of fitted features."""
        return list(self._binning_results.keys())
