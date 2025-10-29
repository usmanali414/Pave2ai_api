
#!/usr/bin/env python3
"""
train_instance.py
Clean training script for Detectron2 Mask R-CNN instance segmentation.
Driven entirely by config_instance.py for easy API integration.

Usage:
    From command line: python train_instance.py
    From API: from train_instance import train_model; train_model()
"""
import os
from pathlib import Path
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from app.services.s3.s3_operations import S3Operations
from app.utils.logger_utils import logger
import asyncio
import requests

from app.deep_models.Algorithms.DETECTRON2.v1.config import (
    get_dataset_root, get_images_dir, get_annotations_dir, get_output_dir,
    MODEL_ZOO_CONFIG, CATEGORIES,
    IMS_PER_BATCH, BASE_LR, MAX_ITER, STEPS, WARMUP_ITERS,
    NUM_WORKERS, BATCH_SIZE_PER_IMAGE, ROI_HEADS_NUM_CLASSES,
    CHECKPOINT_PERIOD, EVAL_PERIOD, DEVICE, CONF_THRESH_TEST, NMS_THRESH_TEST
)

class DETECTRON2:
    def __init__(self):
        self.s3_operations = S3Operations()
        self.base_path = get_dataset_root()
        self.local_weights_path = os.path.join(get_output_dir(), "detectron2_weights.pth")

    def setup_datasets(self):
        """Register COCO datasets for training and validation."""
        ann_dir = get_annotations_dir()  # Use getter function for runtime paths
        ann_train = ann_dir / "instances_train.json"
        ann_val = ann_dir / "instances_val.json"
        img_root = get_images_dir()  # Use getter function for runtime paths
        
        # Check if annotations exist
        if not ann_train.exists():
            raise FileNotFoundError(
                f"Training annotations not found: {ann_train}\n"
                f"Please run json_to_coco.py first to generate COCO annotations."
            )
        
        if not ann_val.exists():
            raise FileNotFoundError(
                f"Validation annotations not found: {ann_val}\n"
                f"Please run json_to_coco.py first to generate COCO annotations."
            )
        
        # Unregister if already registered (for re-runs)
        for dataset_name in ["road_train_inst", "road_val_inst"]:
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.remove(dataset_name)
        
        # Register datasets
        register_coco_instances("road_train_inst", {}, str(ann_train), str(img_root))
        register_coco_instances("road_val_inst", {}, str(ann_val), str(img_root))
        
        # Set metadata
        thing_classes = [cat["name"] for cat in CATEGORIES]
        MetadataCatalog.get("road_train_inst").thing_classes = thing_classes
        MetadataCatalog.get("road_val_inst").thing_classes = thing_classes
        
        print(f"✓ Registered training dataset: {ann_train}")
        print(f"✓ Registered validation dataset: {ann_val}")
        print(f"✓ Image root: {img_root}")
        print(f"✓ Classes: {thing_classes}")


    async def get_training_config(self, initial_weights_flag: bool, initial_weights_s3: str):
        """Build Detectron2 config for training."""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_CONFIG))
        
        # Datasets
        cfg.DATASETS.TRAIN = ("road_train_inst",)
        cfg.DATASETS.TEST = ("road_val_inst",)
        cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
        
        # Prepare output directory early (needed if we download weights here)
        out_dir = get_output_dir()  # Use getter function for runtime paths
        os.makedirs(out_dir, exist_ok=True)
        cfg.OUTPUT_DIR = str(out_dir)
        
        # Model weights with fallback logic
        if initial_weights_s3 and initial_weights_flag and isinstance(initial_weights_s3, str):
            # Try to download from S3 first
            dl = await asyncio.to_thread(
                self.s3_operations.download_file,
                initial_weights_s3,
                local_path=self.local_weights_path,
                return_content=False
            )
            
            if not dl.get("success"):
                # Fallback: S3 file not found, download Model Zoo weights and save locally
                logger.info(f"S3 weights not found at {initial_weights_s3}, downloading Model Zoo weights as fallback")
                
                def _download(url, dest):
                    """Download Model Zoo weights to local path."""
                    with requests.get(url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    return dest
                
                try:
                    # Download Model Zoo weights to local path
                    local_path = await asyncio.to_thread(_download, model_zoo.get_checkpoint_url(MODEL_ZOO_CONFIG), self.local_weights_path)
                    logger.info(f"Downloaded Model Zoo weights to {local_path}")
                    cfg.MODEL.WEIGHTS = local_path
                    
                    # Upload to S3 for future use (optional, don't fail if it errors)
                    try:
                        up = await asyncio.to_thread(self.s3_operations.upload_file, local_path, initial_weights_s3)
                        if up.get("success"):
                            logger.info(f"Uploaded default weights to S3: {initial_weights_s3}")
                        else:
                            logger.warning(f"Failed to upload default weights to S3: {up.get('error')}")
                    except Exception as upload_error:
                        logger.warning(f"Failed to upload default weights to S3: {upload_error}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download Model Zoo weights as fallback: {e}")
            else:
                # Successfully downloaded from S3
                logger.info(f"Downloaded weights from S3: {initial_weights_s3}")
                cfg.MODEL.WEIGHTS = self.local_weights_path
        else:
            # No initial weights flag set, use Model Zoo weights
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ZOO_CONFIG)
        
        # Solver (optimizer)
        cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
        cfg.SOLVER.BASE_LR = BASE_LR
        cfg.SOLVER.MAX_ITER = MAX_ITER
        cfg.SOLVER.STEPS = STEPS
        cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
        cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
        
        # Model architecture
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = ROI_HEADS_NUM_CLASSES
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESH_TEST
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = NMS_THRESH_TEST
        
        # Input format
        cfg.INPUT.MASK_FORMAT = "polygon"  # We use polygon format from COCO
        
        # Evaluation
        cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
        
        # Output and device
        cfg.MODEL.DEVICE = DEVICE
        
        return cfg


    async def train(self, initial_weights_flag: bool, initial_weights_s3: str, final_weights_s3: str, logs_s3: str, resume=False):
        """
        Train the Mask R-CNN model.
        
        Args:
            initial_weights_flag: If True, use initial weights. If False, start fresh.
            initial_weights_s3: S3 path to initial weights.
            final_weights_s3: S3 path to final weights.
            logs_s3: S3 path to logs.
            resume (bool): If True, resume from last checkpoint. If False, start fresh.
        
        Returns:
            trainer: The trained DefaultTrainer object
        """
        setup_logger()
        
        print("=" * 80)
        print("DETECTRON2 MASK R-CNN TRAINING")
        print("=" * 80)
        
        # Setup datasets
        self.setup_datasets()
        
        # Get config
        cfg = await self.get_training_config(initial_weights_flag, initial_weights_s3)
        
        # Setup default configuration
        default_setup(cfg, {})
        
        print("\n" + "=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Model: {MODEL_ZOO_CONFIG}")
        print(f"Output directory: {cfg.OUTPUT_DIR}")
        print(f"Device: {DEVICE}")
        print(f"Number of classes: {ROI_HEADS_NUM_CLASSES}")
        print(f"Base learning rate: {BASE_LR}")
        print(f"Max iterations: {MAX_ITER}")
        print(f"Images per batch: {IMS_PER_BATCH}")
        print(f"Resume from checkpoint: {resume}")
        print("=" * 80 + "\n")
        
        # Create trainer and start training
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=resume)
        
        print("Starting training...")
        await asyncio.to_thread(trainer.train)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Model saved to: {cfg.OUTPUT_DIR}")
        print(f"Final model: {cfg.OUTPUT_DIR}/{self.weights_name}")

        try:
            await asyncio.to_thread(self.s3_operations.upload_file, self.local_weights_path, final_weights_s3)
        except Exception as e:
            print(f"Failed to upload weights to S3: {e}")
        
        
        return trainer


if __name__ == "__main__":
    detectron2 = DETECTRON2()
    asyncio.run(detectron2.train(initial_weights_flag=False, initial_weights_s3="", final_weights_s3="", logs_s3="", resume=False))
