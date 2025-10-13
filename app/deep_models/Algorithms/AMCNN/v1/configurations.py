"""
Configuration class for AMCNN training pipeline.
"""
from typing import Dict, Any
import os

class configurations:
    
    def __init__(self):
        # Model configuration
        self.modelConfg = {
            'class_num': 5,  # Number of classes
            'batch_size': 32,
            'learning_rate': 0.001,
            'N_epoch': 100
        }
        
        # Base patch size
        self.base_patch_size = 50
        
        # Class folder mapping
        self.class_folder_dict = {
            '0': 0,  # Background
            '1': 1,  # Slabs
            '2': 2,  # Cracks
            '3': 3,  # Mark
            '4': 4   # Patch
        }
        
        # Labels dictionary for class weights
        self.labels_dict = {
            0: 1000,  # Background - more samples
            1: 500,   # Slabs
            2: 200,   # Cracks - fewer samples
            3: 100,   # Mark - fewer samples
            4: 150    # Patch
        }
        
        # Training data path (relative to dataset)
        self.train_data_path = 'split_patches_data'
        
        # Model output paths
        self.model_path = 'models'
        self.modelname = 'amcnn_model'
        
        # Tile sizes for multi-scale processing
        self.tile_sizes = [50, 350, 500, 1000, 2500]
        
        # Image size configuration
        self.image_size = (1308, 2473)
        self.same_size_images = False
        
        # Color code mapping
        self.color_code = {
            '0': (255, 255, 255),  # Background - White
            '1': (54, 244, 67),    # Slabs - Green
            '2': (143, 206, 0),    # Cracks - Yellow
            '3': (201, 0, 118),    # Mark - Magenta
            '4': (244, 67, 54)     # Patch - Red
        }
        
        # Data name for saving patches
        self.dataName = 'patch_'
