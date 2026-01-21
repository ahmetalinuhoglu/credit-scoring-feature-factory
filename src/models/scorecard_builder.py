"""
Scorecard Builder

Generates credit scorecard from Logistic Regression model and WoE binning.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.core.exceptions import ModelTrainingError


@dataclass
class ScorecardBin:
    """Scorecard bin with point assignment."""
    feature: str
    bin_id: int
    bin_range: str
    woe: float
    coefficient: float
    points: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature': self.feature,
            'bin_id': self.bin_id,
            'bin_range': self.bin_range,
            'woe': self.woe,
            'coefficient': self.coefficient,
            'points': self.points
        }


class ScorecardBuilder:
    """
    Build credit scorecard from Logistic Regression model.
    
    Scoring Formula:
        Score = Base_Score - (PDO / ln(2)) * Σ(WoE_i * β_i)
        
    Where:
        - PDO: Points to Double Odds
        - β_i: Coefficient for feature i
        - WoE_i: Weight of Evidence for feature i bin
        
    Higher score = lower risk (good)
    Lower score = higher risk (bad)
    """
    
    def __init__(
        self,
        lr_model: Any,
        woe_transformer: Any,
        pdo: float = 20,
        base_score: float = 600,
        base_odds: float = 50,
        min_score: int = 300,
        max_score: int = 850
    ):
        """
        Initialize scorecard builder.
        
        Args:
            lr_model: Fitted LogisticRegressionModel with coefficients
            woe_transformer: Fitted WoETransformer with binning
            pdo: Points to Double Odds (typically 20)
            base_score: Base score at base odds (typically 600)
            base_odds: Base odds (good:bad ratio, typically 50:1)
            min_score: Minimum possible score
            max_score: Maximum possible score
        """
        self.lr_model = lr_model
        self.woe_transformer = woe_transformer
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.min_score = min_score
        self.max_score = max_score
        
        # Scaling factor
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        
        # Built scorecard
        self._scorecard: Dict[str, List[ScorecardBin]] = {}
        self._is_built = False
    
    def build(self) -> Dict[str, List[ScorecardBin]]:
        """
        Build the scorecard from model and WoE binning.
        
        Returns:
            Dictionary of feature name to list of ScorecardBin
        """
        # Validate inputs
        if not hasattr(self.lr_model, 'is_fitted') or not self.lr_model.is_fitted:
            raise ModelTrainingError(
                "Logistic Regression model not fitted",
                model_name="ScorecardBuilder"
            )
        
        if not self.woe_transformer.is_fitted:
            raise ModelTrainingError(
                "WoE transformer not fitted",
                model_name="ScorecardBuilder"
            )
        
        # Get coefficients from LR model
        coefficients = self.lr_model.get_coefficients()
        
        # Get intercept
        if hasattr(self.lr_model.model, 'intercept_'):
            intercept = self.lr_model.model.intercept_[0]
        else:
            intercept = 0
        
        # Calculate base points from intercept
        n_features = len(coefficients)
        base_points_per_feature = (self.offset - self.factor * intercept) / n_features
        
        # Build scorecard for each feature
        for feature_name in self.woe_transformer.fitted_features:
            # Get WoE binning for this feature
            woe_table = self.woe_transformer.get_woe_table(feature_name)
            if woe_table is None:
                continue
            
            # Get coefficient for this feature (might be WoE-encoded name)
            coef_name = f"{feature_name}_woe"
            if coef_name in coefficients:
                coefficient = coefficients[coef_name]
            elif feature_name in coefficients:
                coefficient = coefficients[feature_name]
            else:
                continue
            
            # Create scorecard bins
            feature_bins = []
            for woe_bin in woe_table:
                # Calculate points for this bin
                # Points = Base_points - Factor * Coefficient * WoE
                points = base_points_per_feature - self.factor * coefficient * woe_bin.woe
                points = int(round(points))
                
                # Create bin range string
                if woe_bin.bin_id < 0:
                    bin_range = "Missing"
                else:
                    if np.isinf(woe_bin.lower_bound):
                        bin_range = f"< {woe_bin.upper_bound:.2f}"
                    elif np.isinf(woe_bin.upper_bound):
                        bin_range = f">= {woe_bin.lower_bound:.2f}"
                    else:
                        bin_range = f"[{woe_bin.lower_bound:.2f}, {woe_bin.upper_bound:.2f}]"
                
                scorecard_bin = ScorecardBin(
                    feature=feature_name,
                    bin_id=woe_bin.bin_id,
                    bin_range=bin_range,
                    woe=woe_bin.woe,
                    coefficient=coefficient,
                    points=points
                )
                feature_bins.append(scorecard_bin)
            
            self._scorecard[feature_name] = feature_bins
        
        self._is_built = True
        return self._scorecard
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate final scores for a DataFrame.
        
        Args:
            df: DataFrame with original (non-WoE) features
            
        Returns:
            Series of final scores
        """
        if not self._is_built:
            raise ModelTrainingError(
                "Scorecard not built. Call build() first.",
                model_name="ScorecardBuilder"
            )
        
        # Initialize with zeros
        scores = pd.Series(0.0, index=df.index)
        
        for feature_name, bins in self._scorecard.items():
            if feature_name not in df.columns:
                continue
            
            feature_scores = self._score_feature(df[feature_name], bins)
            scores += feature_scores
        
        # Clip to score range
        scores = scores.clip(self.min_score, self.max_score)
        
        return scores.astype(int)
    
    def _score_feature(
        self,
        series: pd.Series,
        bins: List[ScorecardBin]
    ) -> pd.Series:
        """Score a single feature."""
        result = pd.Series(0.0, index=series.index)
        
        # Separate regular and missing bins
        regular_bins = [b for b in bins if b.bin_id >= 0]
        missing_bin = next((b for b in bins if b.bin_id < 0), None)
        
        # Handle missing values
        if missing_bin:
            result[series.isna()] = missing_bin.points
        
        # Sort bins by lower bound
        regular_bins.sort(key=lambda b: float('-inf') if '[' not in b.bin_range else 
                         float(b.bin_range.split('[')[1].split(',')[0].strip()))
        
        # Apply bins based on value ranges
        for scorecard_bin in regular_bins:
            # Parse bin range
            try:
                lower, upper = self._parse_bin_range(scorecard_bin.bin_range)
                mask = (series >= lower) & (series <= upper) & (~series.isna())
                result[mask] = scorecard_bin.points
            except Exception:
                continue
        
        return result
    
    def _parse_bin_range(self, bin_range: str) -> tuple:
        """Parse bin range string to lower and upper bounds."""
        if bin_range == "Missing":
            return (np.nan, np.nan)
        
        if bin_range.startswith('<'):
            upper = float(bin_range.replace('<', '').strip())
            return (float('-inf'), upper)
        
        if bin_range.startswith('>='):
            lower = float(bin_range.replace('>=', '').strip())
            return (lower, float('inf'))
        
        if bin_range.startswith('[') and ']' in bin_range:
            # Format: [lower, upper]
            inner = bin_range.strip('[]')
            parts = inner.split(',')
            lower = float(parts[0].strip())
            upper = float(parts[1].strip())
            return (lower, upper)
        
        raise ValueError(f"Cannot parse bin range: {bin_range}")
    
    def get_scorecard_table(self) -> pd.DataFrame:
        """
        Get scorecard as a DataFrame.
        
        Returns:
            DataFrame with all scorecard bins
        """
        if not self._is_built:
            raise ModelTrainingError(
                "Scorecard not built",
                model_name="ScorecardBuilder"
            )
        
        rows = []
        for feature, bins in self._scorecard.items():
            for bin_obj in bins:
                rows.append(bin_obj.to_dict())
        
        return pd.DataFrame(rows)
    
    def get_score_distribution(self, scores: pd.Series) -> Dict[str, Any]:
        """
        Get score distribution statistics.
        
        Args:
            scores: Series of scores
            
        Returns:
            Distribution statistics
        """
        return {
            'count': int(len(scores)),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': int(scores.min()),
            'max': int(scores.max()),
            'p10': int(scores.quantile(0.10)),
            'p25': int(scores.quantile(0.25)),
            'p50': int(scores.quantile(0.50)),
            'p75': int(scores.quantile(0.75)),
            'p90': int(scores.quantile(0.90))
        }
    
    def export_scorecard(
        self,
        path: Union[str, Path],
        format: str = 'excel'
    ) -> None:
        """
        Export scorecard to file.
        
        Args:
            path: Output path
            format: 'excel', 'json', or 'csv'
        """
        if not self._is_built:
            raise ModelTrainingError(
                "Scorecard not built",
                model_name="ScorecardBuilder"
            )
        
        path = Path(path)
        df = self.get_scorecard_table()
        
        if format == 'excel':
            df.to_excel(path, index=False, sheet_name='Scorecard')
        elif format == 'json':
            export_data = {
                'metadata': {
                    'pdo': self.pdo,
                    'base_score': self.base_score,
                    'base_odds': self.base_odds,
                    'min_score': self.min_score,
                    'max_score': self.max_score,
                    'factor': self.factor,
                    'offset': self.offset
                },
                'features': {
                    feature: [b.to_dict() for b in bins]
                    for feature, bins in self._scorecard.items()
                }
            }
            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'csv':
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load_scorecard(self, path: Union[str, Path]) -> None:
        """
        Load scorecard from JSON file.
        
        Args:
            path: Path to JSON scorecard file
        """
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Restore metadata
        metadata = data.get('metadata', {})
        self.pdo = metadata.get('pdo', self.pdo)
        self.base_score = metadata.get('base_score', self.base_score)
        self.base_odds = metadata.get('base_odds', self.base_odds)
        self.min_score = metadata.get('min_score', self.min_score)
        self.max_score = metadata.get('max_score', self.max_score)
        self.factor = metadata.get('factor', self.pdo / np.log(2))
        self.offset = metadata.get('offset', self.base_score - self.factor * np.log(self.base_odds))
        
        # Restore scorecard
        self._scorecard = {}
        for feature, bins_data in data.get('features', {}).items():
            bins = [ScorecardBin(**b) for b in bins_data]
            self._scorecard[feature] = bins
        
        self._is_built = True
    
    @property
    def is_built(self) -> bool:
        """Check if scorecard is built."""
        return self._is_built
    
    @property
    def features(self) -> List[str]:
        """Get list of scorecard features."""
        return list(self._scorecard.keys())
