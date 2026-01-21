"""
Features Module

Provides feature engineering functionality using Spark.
"""

from src.features.base_transformer import BaseTransformer
from src.features.feature_extractor import FeatureExtractor
from src.features.data_dictionary import DataDictionaryGenerator

__all__ = [
    "BaseTransformer",
    "FeatureExtractor",
    "DataDictionaryGenerator",
]
