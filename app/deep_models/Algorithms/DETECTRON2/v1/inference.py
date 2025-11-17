
#!/usr/bin/env python3
"""
infer_instance.py
Inference script for Detectron2 Mask R-CNN instance segmentation.

WHAT THIS SCRIPT DOES:
1. Loads your trained model (model_final.pth)
2. Processes ALL images in the specified directory (default: sample_data/orig_images/)
3. Generates TWO types of outputs:
   
   A. QUALITATIVE (Visual):
      - PNG images with colored masks overlaid
      - Each instance gets a colored overlay
      - Bounding boxes and labels shown
      - Saved to: viz_instances/*_pred.png
   
   B. QUANTITATIVE (Numbers):
      - JSON file with all detection data
      - Bounding boxes, scores, class IDs
      - Saved to: viz_instances/predictions.json

HOW IT KNOWS WHICH IMAGES TO USE:
- By default: Uses images from sample_data/orig_images/
- You can specify custom directory (see below)
- Processes ALL .jpg, .jpeg, .png files in the directory

Usage:
    # Simple command line:
    python infer_instance.py
    
    # From Python API (custom directory):
    from infer_instance import run_inference
    results = run_inference(
        weights="model_final.pth",
        images_dir="/path/to/your/test/images",
        out_dir="my_results"
    )

For detailed explanation, see INFERENCE_GUIDE.md
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import os

from app.deep_models.Algorithms.DETECTRON2.v1.config import (
    get_dataset_root, get_images_dir, get_output_dir, MODEL_ZOO_CONFIG, DEVICE,
    ROI_HEADS_NUM_CLASSES, CONF_THRESH_TEST, TEST_INPUT_MIN_SIZE, 
    TEST_INPUT_MAX_SIZE, CATEGORIES
)

def get_cfg_for_infer(weights_path: str):
    """Build config for inference."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_CONFIG))
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = ROI_HEADS_NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESH_TEST
    cfg.INPUT.MIN_SIZE_TEST = TEST_INPUT_MIN_SIZE
    cfg.INPUT.MAX_SIZE_TEST = TEST_INPUT_MAX_SIZE
    cfg.MODEL.DEVICE = DEVICE
    return cfg


def run_inference(weights="model_final.pth", images_dir=None, out_dir="viz_instances"):
    """
    Run inference on images and save predictions.
    
    Args:
        weights (str): Path to weights file. Can be:
            - Absolute path (e.g., "/tmp/model_final.pth")
            - Relative filename in OUTPUT_DIR (e.g., "model_final.pth")
        images_dir (str/Path): Directory containing images to process. If None, uses DATASET_ROOT/IMAGES_SUBDIR
        out_dir (str/Path): Output directory for visualizations and predictions
    
    Returns:
        results (list): List of prediction dictionaries
    """
    if images_dir is None:
        images_dir = get_images_dir()
    else:
        images_dir = Path(images_dir)
    
    out_dir = get_output_dir() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if weights exist - handle both absolute and relative paths
    if os.path.isabs(weights):
        weights_path = Path(weights)
    else:
        weights_path = get_output_dir() / weights
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Please train the model first or provide valid weights path"
        )
    
    print("="*80)
    print("DETECTRON2 INFERENCE")
    print("="*80)
    print(f"Model weights: {weights_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Device: {DEVICE}")
    print(f"Confidence threshold: {CONF_THRESH_TEST}")
    print("="*80 + "\n")
    
    # Load model
    cfg = get_cfg_for_infer(str(weights_path))
    predictor = DefaultPredictor(cfg)
    
    # Setup metadata - create custom metadata for visualization
    from detectron2.data import Metadata
    meta = Metadata()
    meta.thing_classes = [cat["name"] for cat in CATEGORIES]
    meta.thing_colors = [
        (54, 244, 67),   # Slabs - green
        (143, 206, 0),   # Cracks - yellow-green
        (201, 0, 118),   # Mark - pink
        (244, 67, 54),   # Patch - red
    ]
    
    # Process images
    results = []
    image_files = sorted([p for p in images_dir.glob("*") 
                         if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    
    if not image_files:
        print(f"Warning: No images found in {images_dir}")
        return results
    
    print(f"Processing {len(image_files)} images...\n")
    
    for i, p in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {p.name}...", end=" ")
        
        # Load image
        im = np.array(Image.open(p).convert("RGB"))
        
        # Run inference
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        
        print(f"Found {len(instances)} instances")
        
        # Save visualization
        v = Visualizer(im, metadata=meta, instance_mode=ColorMode.IMAGE)
        vis = v.draw_instance_predictions(instances).get_image()
        Image.fromarray(vis).save(out_dir / f"{p.stem}_pred.png")
        
        # Collect predictions
        pred_dict = {
            "file_name": p.name,
            "num_instances": len(instances),
            "boxes_xyxy": instances.pred_boxes.tensor.numpy().tolist() if instances.has("pred_boxes") else [],
            "scores": instances.scores.numpy().tolist() if instances.has("scores") else [],
            "classes": instances.pred_classes.numpy().tolist() if instances.has("pred_classes") else [],
        }
        
        # Add class names
        if instances.has("pred_classes"):
            class_names = [meta.thing_classes[cls] for cls in instances.pred_classes.numpy()]
            pred_dict["class_names"] = class_names
        
        results.append(pred_dict)
    
    # Save predictions JSON
    predictions_file = out_dir / "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED")
    print("="*80)
    print(f"✓ Processed {len(results)} images")
    print(f"✓ Visualizations: {out_dir}/*.png")
    print(f"✓ Predictions: {predictions_file}")
    
    # Summary statistics
    total_instances = sum(r["num_instances"] for r in results)
    print(f"\nTotal instances detected: {total_instances}")
    
    # Count by class
    class_counts = {}
    for r in results:
        for cls_name in r.get("class_names", []):
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    if class_counts:
        print("\nInstances by class:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  {cls_name}: {count}")
    
    print("="*80 + "\n")
    
    return results


def run_folder_inference(weights: str = "model_final.pth", out_dir: str = "viz_instances"):
    print(f"Running inference of det2")
    """Main function for command-line usage."""
    return run_inference(weights=weights, out_dir=out_dir)


if __name__ == "__main__":
    run_folder_inference(weights="model_final.pth", out_dir="viz_instances")
