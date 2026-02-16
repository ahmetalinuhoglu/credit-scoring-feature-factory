"""
Features Module

Provides feature engineering functionality using Spark.
"""

# Lazy imports to avoid requiring PySpark when only WoE is needed
def __getattr__(name):
    if name == "BaseTransformer":
        from src.features.base_transformer import BaseTransformer
        return BaseTransformer
    elif name == "FeatureExtractor":
        from src.features.feature_extractor import FeatureExtractor
        return FeatureExtractor
    elif name == "DataDictionaryGenerator":
        from src.features.data_dictionary import DataDictionaryGenerator
        return DataDictionaryGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseTransformer",
    "FeatureExtractor",
    "DataDictionaryGenerator",
]
