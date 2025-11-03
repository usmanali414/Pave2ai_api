
# config_instance.py
# ═══════════════════════════════════════════════════════════════════════════
# CENTRALIZED CONFIGURATION FOR DETECTRON2 INSTANCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════
#
# ALL PIPELINE PARAMETERS ARE CONTROLLED HERE
# Edit this file to configure paths, training, and inference settings
#
# See PIPELINE_GUIDE.md for detailed explanations of each parameter
#
# NOTE: Category IDs are 1..N to be COCO-compliant. Background is implicit (0).

# from pathlib import Path

# # ═══════════════════════════════════════════════════════════════════════════
# # DATA PATHS - Where your data is located
# # ═══════════════════════════════════════════════════════════════════════════

# DATASET_ROOT = Path(__file__).parent / "Dataset"  # Your dataset root folder
# # To use a different dataset:
# # DATASET_ROOT = Path("/path/to/your/dataset")

# IMAGES_SUBDIR = "orig_images"    # Subdirectory containing RGB images
# MASKS_SUBDIR  = "masks"          # (Optional) Subdirectory for PNG masks
# JSON_SUBDIR = "jsons"             # Subdirectory containing JSON label files

from pathlib import Path
from typing import Optional, Dict

# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME DATA PATHS - Set by orchestrator after RIEGL_PARSER.load_data()
# ═══════════════════════════════════════════════════════════════════════════

# Add near the top
from pathlib import Path
from typing import Optional, Dict

_STATIC_ROOT: Optional[Path] = None  # E:/Pave2ai_api/static

def set_static_root(root: Path):
    global _STATIC_ROOT
    _STATIC_ROOT = root

def _default_static_root() -> Path:
    here = Path(__file__).resolve()
    # robust default: repo_root/static
    repo_root = here.parents[5] if len(here.parents) >= 6 else here.parents[-1]
    return repo_root / "static"

def get_static_root() -> Path:
    return _STATIC_ROOT or _default_static_root()

# Hardcode the model layout relative to static root (no 'dataset' level)
MODEL_NS = Path("DETECTRON2") / "v1"
IMAGES_SUBDIR = "orig_images"
MASKS_SUBDIR  = "masks"
JSON_SUBDIR   = "jsons"
ANNOTATIONS_DIR = "annotations"
OUTPUT_DIR = "output_instances"

def get_dataset_root() -> Path:
    return get_static_root() / MODEL_NS

def get_images_dir() -> Path:
    return get_dataset_root() / IMAGES_SUBDIR

def get_jsons_dir() -> Path:
    return get_dataset_root() / JSON_SUBDIR

def get_masks_dir() -> Path:
    return get_dataset_root() / MASKS_SUBDIR

def get_annotations_dir() -> Path:
    return get_dataset_root() / ANNOTATIONS_DIR

def get_output_dir() -> Path:
    return get_dataset_root() / OUTPUT_DIR
# TRAIN/VAL SPLIT - How to divide your data
# ═══════════════════════════════════════════════════════════════════════════

# Option 1: Automatic split (uses SPLIT_TRAIN ratio)
TRAIN_LIST = None    # Leave as None for automatic split
VAL_LIST   = None    # Leave as None for automatic split

# Option 2: Explicit lists (if you have predefined splits)
# TRAIN_LIST = DATASET_ROOT / "splits/train.txt"  # File with training image stems
# VAL_LIST   = DATASET_ROOT / "splits/val.txt"    # File with validation image stems

# Automatic split settings (used when TRAIN_LIST/VAL_LIST are None)
SPLIT_TRAIN = 0.75   # 75% training, 25% validation
                     # Adjust: 0.8 for 80/20, 0.9 for 90/10
SPLIT_SEED  = 1337   # Random seed for reproducibility

# ═══════════════════════════════════════════════════════════════════════════
# CLASS MAPPING - How JSON labels map to final classes
# ═══════════════════════════════════════════════════════════════════════════
# This defines which original JSON labels get merged into each final class
# Add/remove labels as needed for your dataset

COMBINING_CLASSES_MAP = [
    ['conc-slb', 'bridge'],                                    # → Class 1: Slabs
    ['conc-crk', 'conc-crk-seal', 'conc-cut', 'conc-spl'],    # → Class 2: Cracks
    ['conc-mrk'],                                              # → Class 3: Mark
    ['conc-pat'],                                              # → Class 4: Patch
]
# To add a label: Add it to the appropriate sublist above
# To add a class: Add a new sublist and update CATEGORIES below

# Colors and classes of your masks (for PNG mask generation if needed)
COLOR_CODE = {'0': (255,255,255), '1': (54,244,67), '2': (143,206,0), '3': (201,0,118), '4': (244,67,54)}
CLASS_TO_CODE = {'0': 'Background', '1': 'Slabs', '2': 'Cracks', '3': 'Mark', '4': 'Patch'}

# COCO categories (ids must start at 1 and be contiguous)
CATEGORIES = [
    {"id": 1, "name": "Slabs"},
    {"id": 2, "name": "Cracks"},
    {"id": 3, "name": "Mark"},
    {"id": 4, "name": "Patch"},
]

# Minimum instance area to keep (in pixels)
MIN_INSTANCE_AREA = 50

# Output directory for COCO annotations (created by preprocessing)
ANNOTATIONS_DIR = "annotations"  # Subdirectory inside DATASET_ROOT

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING PARAMETERS - Control model training
# ═══════════════════════════════════════════════════════════════════════════

# Output directory for trained model and logs
OUTPUT_DIR = "output_instances"

# Model architecture
MODEL_ZOO_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# PRETRAINED_WEIGHTS = None  # None = use model_zoo pretrained weights
                           # Or set to path of your .pth file

# Training hyperparameters
# OPTIMIZED FOR YOUR DATASET: 1900 train + 700 val images
MAX_ITER = 15000         # Total training iterations (~16 epochs)
# MAX_ITER = 5         # Total training iterations (~16 epochs)
                         # Your dataset: 1900 images ÷ 2 batch = 950 iter/epoch
                         # 16 epochs = 15,200 iterations (good for your dataset size)

BASE_LR = 0.00025        # Learning rate (good starting point)
                         # Can reduce to 0.0001 if training is unstable

IMS_PER_BATCH = 2        # Batch size (images processed together)
# IMS_PER_BATCH = 1        # Batch size (images processed together)
                         # Your images are large (2150×3406), so keep at 2
                         # Reduce to 1 if GPU memory issues

WARMUP_ITERS = 500       # Learning rate warmup iterations (~0.5 epochs)
# WARMUP_ITERS = 50       # Learning rate warmup iterations (~0.5 epochs)
STEPS = [10000, 13000]   # Learning rate decay steps (at 10.5 and 13.7 epochs)
# STEPS = []   # Learning rate decay steps (at 10.5 and 13.7 epochs)

# Data loadingD
NUM_WORKERS = 2          # Parallel data loading workers
# NUM_WORKERS = 0        # Parallel data loading workers

# ROI settings
BATCH_SIZE_PER_IMAGE = 256   # RPN/ROI sampler batch size
                             # Reduce to 128 if GPU memory limited

ROI_HEADS_NUM_CLASSES = len(CATEGORIES)  # Number of classes (auto-set)

# Checkpointing and evaluation
CHECKPOINT_PERIOD = 1000   # Save checkpoint every N iterations (~1 epoch)
EVAL_PERIOD = 1000         # Evaluate on validation set every N iterations (~1 epoch)
# CHECKPOINT_PERIOD = 200   # Save checkpoint every N iterations (~1 epoch)
# EVAL_PERIOD = 200         # Evaluate on validation set every N iterations (~1 epoch)

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE PARAMETERS - Control prediction behavior
# ═══════════════════════════════════════════════════════════════════════════

CONF_THRESH_TEST = 0.5     # Confidence threshold (0.0 to 1.0)
                           # Lower (0.3) = more detections (may include false positives)
                           # Higher (0.7) = fewer detections (only high confidence)

NMS_THRESH_TEST = 0.5      # Non-maximum suppression threshold
                           # Controls overlapping detection removal

# Image resizing for inference
# OPTIMIZED FOR YOUR IMAGE SIZE: 2150×3406
TEST_INPUT_MIN_SIZE = 1200   # Resize shorter side to this (was 800)
                             # Your images are large, so use higher resolution
TEST_INPUT_MAX_SIZE = 2000   # Maximum size for longer side (was 1333)
                             # Keeps aspect ratio while fitting in GPU memory

# ═══════════════════════════════════════════════════════════════════════════
# DEVICE SELECTION - GPU or CPU
# ═══════════════════════════════════════════════════════════════════════════

DEVICE = "cuda"   # "cuda" for GPU (fast), "cpu" for CPU (slow but works everywhere)
# DEVICE = "cpu"   # "cuda" for GPU (fast), "cpu" for CPU (slow but works everywhere)
                  # Training on CPU is VERY slow - reduce MAX_ITER if using CPU
