"""
Configuration for AMCNN v1 model.
"""
from typing import Dict, Any

AMCNN_V1_CONFIG = {
    "model_name": "AMCNN",
    "version": "v1",
    "architecture": {
        "input_shape": (100000, 3),  # (num_points, features)
        "num_classes": 10,
        "feature_dim": 256,
        "hidden_layers": [512, 256, 128],
        "dropout_rate": 0.3,
        "activation": "relu"
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "metrics": ["accuracy", "precision", "recall"],
        "validation_split": 0.2,
        "early_stopping": {
            "patience": 10,
            "monitor": "val_loss",
            "restore_best_weights": True
        },
        "reduce_lr": {
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-7
        }
    },
    "data_augmentation": {
        "rotation": True,
        "scaling": True,
        "noise": True,
        "jitter": 0.01
    },
    "model_output": {
        "weights_format": "h5",
        "save_best_only": True,
        "save_frequency": 10
    }
}
