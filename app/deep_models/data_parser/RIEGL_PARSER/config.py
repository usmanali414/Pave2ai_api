"""
Configuration for RIEGL_PARSER data processing.
"""
from typing import Dict, Any

RIEGL_PARSER_CONFIG = {
    "parser_name": "RIEGL_PARSER",
    "supported_formats": [".jpg", ".png", ".json"],
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
    "mask_generation": {
        "combining_classes": [
            ["crack", "cracks"],  # Example classes that should be combined
            ["background", "bg"]
        ],
        "classes_rgb": [
            [0, 0, 255],    # Red for cracks
            [255, 255, 255] # White for background
        ],
        "default_thickness": 2,
        "background_color": [255, 255, 255]
    },
    "local_storage": {
        "base_path": "static",
        "dataset_dir": "dataset",
        "images_dir": "orig_images",
        "jsons_dir": "jsons",
        "masks_dir": "masks",
        "patches_dir": "split_patches_data"
    },
    "patch_generation": {
        "image_size": (1308, 2473),
        "base_patch_size": 50,
        "tile_sizes": [50, 350, 500, 1000, 2500],
        "same_size_images": True,
        "mask_extension": ".png",
        "datatype": "new",
        "class_to_code": {
            "0": "Background",
            "1": "Slabs", 
            "2": "Cracks",
            "3": "Mark",
            "4": "Patch"
        },
        "color_code": {
            "0": (255, 255, 255),
            "1": (54, 244, 67),
            "2": (143, 206, 0),
            "3": (201, 0, 118),
            "4": (244, 67, 54)
        }
    },
    "output_format": {
        "features": "numpy_array",
        "labels": "numpy_array",
        "metadata": "dict"
    }
}