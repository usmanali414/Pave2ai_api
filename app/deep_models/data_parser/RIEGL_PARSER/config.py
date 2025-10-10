"""
Configuration for RIEGL_PARSER data processing.
"""
from typing import Dict, Any

RIEGL_PARSER_CONFIG = {
    "parser_name": "RIEGL_PARSER",
    "supported_formats": [".las", ".laz", ".ply", ".pcd"],
    "data_processing": {
        "normalize": True,
        "remove_outliers": True,
        "outlier_threshold": 3.0,
        "voxel_size": 0.1,
        "max_points_per_cloud": 100000
    },
    "preprocessing": {
        "feature_extraction": True,
        "augmentation": True,
        "train_test_split": 0.8,
        "validation_split": 0.1
    },
    "output_format": {
        "features": "numpy_array",
        "labels": "numpy_array",
        "metadata": "dict"
    }
}
