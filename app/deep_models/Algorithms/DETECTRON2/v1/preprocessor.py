#!/usr/bin/env python3
"""
json_to_coco.py
Convert JSON label files (with polygons and lines) directly to COCO-format instance annotations.
This script mimics the class merging logic from mask_generator.py but outputs COCO JSON instead of PNG masks.

Each polygon/line in the JSON becomes a separate instance in COCO format.
"""
import json
import random
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Dict

from app.deep_models.Algorithms.DETECTRON2.v1.config import (
    get_dataset_root, get_annotations_dir, TRAIN_LIST, VAL_LIST, 
    SPLIT_TRAIN, SPLIT_SEED, CATEGORIES, COMBINING_CLASSES_MAP,IMAGES_SUBDIR,JSON_SUBDIR
)


def build_label_to_class_map(combining_classes: List[List[str]]) -> Dict[str, int]:
    """
    Build a mapping from original label names to COCO category IDs (1..N).
    
    Args:
        combining_classes: List of lists, where each sublist contains labels that map to one class.
                          The order corresponds to category IDs: [class_1_labels, class_2_labels, ...]
    
    Returns:
        Dictionary mapping label string -> category_id
    """
    label_map = {}
    for cat_idx, label_group in enumerate(combining_classes):
        cat_id = cat_idx + 1  # COCO category IDs start at 1
        for label in label_group:
            label_map[label] = cat_id
    return label_map


def linestring_to_polygon(coordinates: List[List[int]], thickness: float) -> List[List[float]]:
    """
    Convert a LineString with thickness to a polygon.
    
    Args:
        coordinates: List of [x, y] points defining the line
        thickness: Line thickness in pixels
    
    Returns:
        Polygon coordinates as a flat list [x1, y1, x2, y2, ...]
    """
    if len(coordinates) < 2:
        return []
    
    try:
        # Create a blank image to draw the line with thickness
        points = np.array(coordinates, dtype=np.int32)
        
        # Validate coordinates
        if len(points) == 0 or points.shape[1] != 2:
            return []
        
        # Get bounding box with padding
        min_x, min_y = points.min(axis=0) - int(thickness) - 10
        max_x, max_y = points.max(axis=0) + int(thickness) + 10
        
        # Ensure non-negative and valid dimensions
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = max(min_x + 1, max_x), max(min_y + 1, max_y)
        
        # Validate canvas dimensions
        canvas_width = max_x - min_x + 1
        canvas_height = max_y - min_y + 1
        
        if canvas_width <= 0 or canvas_height <= 0:
            print(f"Warning: Invalid canvas dimensions: {canvas_width}x{canvas_height}")
            return []
        
        # Limit canvas size to prevent memory issues
        if canvas_width > 10000 or canvas_height > 10000:
            print(f"Warning: Canvas too large: {canvas_width}x{canvas_height}, skipping")
            return []
        
        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        
        # Shift coordinates to canvas space
        shifted_coords = points - np.array([min_x, min_y])
        
        # Draw line with thickness
        thickness_int = max(1, int(thickness))
        for i in range(len(shifted_coords) - 1):
            pt1 = tuple(shifted_coords[i])
            pt2 = tuple(shifted_coords[i + 1])
            cv2.line(canvas, pt1, pt2, 255, thickness=thickness_int)
        
        # Find contours to get polygon
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 3:
            return []
        
        # Shift back to original coordinate space
        polygon = largest_contour.reshape(-1, 2) + np.array([min_x, min_y])
        
        return polygon.astype(float).ravel().tolist()
        
    except Exception as e:
        print(f"Warning: Error processing LineString: {e}")
        return []


def polygon_to_coco_format(coordinates: List[List[int]]) -> List[float]:
    """
    Convert polygon coordinates to COCO format (flat list of x,y values).
    
    Args:
        coordinates: List of [x, y] points
    
    Returns:
        Flat list [x1, y1, x2, y2, ...]
    """
    if len(coordinates) < 3:
        return []
    
    flat_coords = []
    for x, y in coordinates:
        flat_coords.extend([float(x), float(y)])
    
    return flat_coords


def compute_bbox_from_polygon(polygon: List[float]) -> List[float]:
    """
    Compute bounding box [x, y, width, height] from polygon.
    
    Args:
        polygon: Flat list [x1, y1, x2, y2, ...]
    
    Returns:
        [x, y, width, height]
    """
    if len(polygon) < 6:  # Need at least 3 points
        return [0, 0, 0, 0]
    
    xs = polygon[0::2]
    ys = polygon[1::2]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def compute_polygon_area(polygon: List[float]) -> float:
    """
    Compute area of polygon using shoelace formula.
    
    Args:
        polygon: Flat list [x1, y1, x2, y2, ...]
    
    Returns:
        Area in pixels
    """
    if len(polygon) < 6:
        return 0.0
    
    xs = polygon[0::2]
    ys = polygon[1::2]
    
    # Shoelace formula
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    
    return abs(area) / 2.0


def parse_json_to_instances(json_path: Path, label_map: Dict[str, int], 
                           image_width: int, image_height: int) -> List[Tuple[int, List[float]]]:
    """
    Parse a JSON file and extract instances (category_id, polygon_coords).
    
    Args:
        json_path: Path to JSON file
        label_map: Mapping from label string to category ID
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        List of (category_id, polygon) tuples
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    objects = data.get('lsfs', {}).get('99', {}).get('objects', [])
    
    instances = []
    
    for obj in objects:
        # Skip deleted objects
        if obj.get('deleted', False):
            continue
        
        label = obj.get('label', '')
        
        # Map label to category ID
        cat_id = label_map.get(label, None)
        
        if cat_id is None:
            # Skip unknown labels (they would be background or masked areas)
            continue
        
        obj_type = obj.get('type', '')
        coordinates = obj.get('coordinates', [])
        
        if not coordinates:
            continue
        
        # Validate coordinates
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            print(f"Warning: Invalid coordinates for {label}, skipping")
            continue
        
        # Check if coordinates are valid numbers
        try:
            for coord in coordinates:
                if not isinstance(coord, list) or len(coord) != 2:
                    raise ValueError("Invalid coordinate format")
                float(coord[0]), float(coord[1])  # Test if convertible to float
        except (ValueError, TypeError, IndexError):
            print(f"Warning: Invalid coordinate values for {label}, skipping")
            continue
        
        polygon = []
        
        if obj_type == 'Polygon':
            polygon = polygon_to_coco_format(coordinates)
        
        elif obj_type == 'LineString':
            thickness = obj.get('thickness', 1.0)
            # Validate thickness
            if thickness <= 0 or thickness > 1000:  # Reasonable limits
                print(f"Warning: Invalid thickness {thickness}, using default 1.0")
                thickness = 1.0
            polygon = linestring_to_polygon(coordinates, thickness)
        
        else:
            # Unknown type, skip
            continue
        
        if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
            continue
        
        # Validate polygon is within image bounds (clip if necessary)
        # This is important because some coordinates might be negative or outside bounds
        clipped_polygon = []
        for i in range(0, len(polygon), 2):
            x = max(0, min(polygon[i], image_width))
            y = max(0, min(polygon[i+1], image_height))
            clipped_polygon.extend([x, y])
        
        instances.append((cat_id, clipped_polygon))
    
    return instances


def list_json_stems(json_dir: Path) -> List[str]:
    """List all JSON file stems (without _mask.json suffix)."""
    stems = []
    for p in sorted(json_dir.glob("*.json")):
        stem = p.stem
        # Remove _mask suffix if present
        if stem.endswith('_mask'):
            stem = stem[:-5]
        stems.append(stem)
    return list(set(stems))  # Remove duplicates


def make_splits(json_dir: Path) -> Tuple[List[str], List[str]]:
    """Create train/val splits."""
    if TRAIN_LIST and VAL_LIST:
        train_stems = [l.strip() for l in Path(TRAIN_LIST).read_text().splitlines() if l.strip()]
        val_stems = [l.strip() for l in Path(VAL_LIST).read_text().splitlines() if l.strip()]
    else:
        rng = random.Random(SPLIT_SEED)
        stems = list_json_stems(json_dir)
        rng.shuffle(stems)
        n_tr = int(len(stems) * SPLIT_TRAIN)
        train_stems, val_stems = stems[:n_tr], stems[n_tr:]
    
    return train_stems, val_stems


def find_image(images_dir: Path, stem: str) -> Tuple[Path, int, int]:
    """Find image file and return path, width, height."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            # Read image to get dimensions
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                return p, w, h
    return None, 0, 0


def build_coco(images_dir: Path, json_dir: Path, stems: List[str], 
               label_map: Dict[str, int], 
               start_ann_id: int = 1, start_img_id: int = 1) -> Dict:
    """
    Build COCO format annotations from JSON files.
    
    Args:
        images_dir: Directory containing original images
        json_dir: Directory containing JSON label files
        stems: List of file stems to process
        label_map: Mapping from label to category ID
        start_ann_id: Starting annotation ID
        start_img_id: Starting image ID
    
    Returns:
        COCO format dictionary
    """
    images = []
    annotations = []
    ann_id = start_ann_id
    img_id = start_img_id
    
    for stem in stems:
        # Find corresponding image
        img_path, width, height = find_image(images_dir, stem)
        
        if img_path is None:
            print(f"Warning: Could not find image for {stem}, skipping...")
            continue
        
        # Find corresponding JSON
        json_path = json_dir / f"{stem}_mask.json"
        if not json_path.exists():
            json_path = json_dir / f"{stem}.json"
        
        if not json_path.exists():
            print(f"Warning: Could not find JSON for {stem}, skipping...")
            continue
        
        # Add image entry
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "height": height,
            "width": width
        })
        
        # Parse instances from JSON
        instances = parse_json_to_instances(json_path, label_map, width, height)
        
        # Create annotation for each instance
        for cat_id, polygon in instances:
            bbox = compute_bbox_from_polygon(polygon)
            area = compute_polygon_area(polygon)
            
            if area < 10:  # Skip very small instances
                continue
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "iscrowd": 0,
                "segmentation": [polygon],  # COCO expects list of polygons
                "bbox": bbox,
                "area": area
            })
            ann_id += 1
        
        img_id += 1
    
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c["id"], "name": c["name"], "supercategory": "none"} 
                      for c in CATEGORIES],
    }
    
    return coco


def main():
    """Main function to convert JSON labels to COCO format."""
    root = get_dataset_root()
    json_dir =  root / JSON_SUBDIR
    images_dir = root / IMAGES_SUBDIR
    ann_dir = get_annotations_dir()  # Use getter function for runtime paths
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    # Build label mapping
    label_map = build_label_to_class_map(COMBINING_CLASSES_MAP)
    
    print(f"Label mapping: {label_map}")
    print(f"Processing JSONs from: {json_dir}")
    print(f"Processing images from: {images_dir}")
    
    # Create train/val splits
    train_stems, val_stems = make_splits(json_dir)
    
    print(f"\nTrain samples: {len(train_stems)}")
    print(f"Val samples: {len(val_stems)}")
    
    # Build COCO annotations
    print("\nBuilding train annotations...")
    train_coco = build_coco(images_dir, json_dir, train_stems, label_map, 
                           start_ann_id=1, start_img_id=1)
    
    print("Building validation annotations...")
    val_coco = build_coco(images_dir, json_dir, val_stems, label_map,
                         start_ann_id=1_000_000, start_img_id=1_000_000)
    
    # Save COCO JSON files
    train_path = ann_dir / "instances_train.json"
    val_path = ann_dir / "instances_val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"\n✓ Wrote {len(train_coco['images'])} train images, {len(train_coco['annotations'])} annotations -> {train_path}")
    print(f"✓ Wrote {len(val_coco['images'])} val images, {len(val_coco['annotations'])} annotations -> {val_path}")
    
    # Print class distribution
    print("\nClass distribution (train):")
    class_counts = {}
    for ann in train_coco['annotations']:
        cat_id = ann['category_id']
        class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
    
    for cat in CATEGORIES:
        count = class_counts.get(cat['id'], 0)
        print(f"  {cat['name']} (id={cat['id']}): {count} instances")


if __name__ == "__main__":
    main()


