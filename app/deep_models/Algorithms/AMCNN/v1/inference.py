import os
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from app.utils.logger_utils import logger

from keras.optimizers import Adam
from app.deep_models.Algorithms.AMCNN.v1 import config as amcnn_config_mod
from app.deep_models.Algorithms.AMCNN.v1.AMCNN import AMCNN
from app.deep_models.data_parser.RIEGL_PARSER.config import RIEGL_PARSER_CONFIG
from app.deep_models.Algorithms.AMCNN.v1.preprocessor import RIEGLPreprocessor


def _get_dataset_base_path() -> str:
    current_file = Path(__file__)
    project_root = current_file.parents[5]  # .../app/deep_models/Algorithms/AMCNN/v1 → project root
    base_path = RIEGL_PARSER_CONFIG["local_storage"]["base_path"]  # e.g. "static"
    return str(project_root / base_path)


def _get_images_dir() -> str:
    base = _get_dataset_base_path()
    rel = RIEGL_PARSER_CONFIG["local_storage"]["images_dir"]  # e.g. "orig_images"
    return os.path.join(base, rel)


def _get_output_dirs() -> Tuple[str, str]:
    # root/static/amcnn_inference/{masks,overlays}
    base_static = _get_dataset_base_path()
    inf_base = RIEGL_PARSER_CONFIG["local_storage"]["inference_base_dir"]
    masks_dir = os.path.join(base_static, inf_base, RIEGL_PARSER_CONFIG["local_storage"]["inference_masks_dir"])
    overlays_dir = os.path.join(base_static, inf_base, RIEGL_PARSER_CONFIG["local_storage"]["inference_overlays_dir"])
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    return masks_dir, overlays_dir


def _load_inference_components(weights_local_path: Optional[str] = None):
    # AMCNN config
    # Use AMCNN class' internal config for consistency
    amcnn = AMCNN()
    configs = amcnn.training_configs

    # Build model in inference mode
    learning_rate = configs.modelConfg['learning_rate']
    optim = Adam(learning_rate)
    # Build model in inference mode using AMCNN architecture
    model = amcnn.classifier(optim, is_training=False)

    # Load weights: prefer provided local override
    if weights_local_path and os.path.exists(weights_local_path):
        model.load_weights(weights_local_path)
    else:
        raise Exception("No weights provided for inference")

    # Use preprocessor utilities for tiling during inference
    preprocessor = RIEGLPreprocessor()

    return model, preprocessor, configs


def infer_image(img: np.ndarray, model, preprocessor: RIEGLPreprocessor, configs) -> np.ndarray:
    image_size = img.shape[:2]
    base_patch_size = configs.base_patch_size

    # Compute grid and padded images using preprocessor utilities
    no_of_rows, no_of_cols = preprocessor.get_no_rows_cols(image_size, base_patch_size)
    padded_imgs_list, padding_tuples_list = preprocessor.get_padded_imgs(img, image_size)

    # Create output mask with same channel shape as padded image
    mask_shape = padded_imgs_list[0].shape
    mask = np.zeros(mask_shape, dtype=np.uint8)

    # Iterate columns and rows similar to training tiling
    pred_calls = 0
    for col_index1 in range(no_of_cols):
        for row_index1 in range(no_of_rows):
            tiles_nested = preprocessor.get_tiles_of_all_sizes(row_index1, col_index1, padded_imgs_list, padding_tuples_list)
            curr_tiles = tiles_nested
            if isinstance(curr_tiles, list) and curr_tiles and isinstance(curr_tiles[0], list):
                curr_tiles = curr_tiles[0]

            if not curr_tiles:
                logger.info(f"inference: skip empty tiles r={row_index1} c={col_index1}")
                continue
            if len(curr_tiles) < 5:
                logger.info(f"inference: skip <5 tiles r={row_index1} c={col_index1} n={len(curr_tiles)}")
                continue
            # Validate tile shapes/dtypes once in a while
            t0 = curr_tiles[0]
            if not isinstance(t0, np.ndarray):
                logger.info(f"inference: non-ndarray tile r={row_index1} c={col_index1} type={type(t0)}")
                continue
            if t0.ndim != 3:
                logger.info(f"inference: bad tile ndim={t0.ndim} shape={t0.shape} r={row_index1} c={col_index1}")
                continue

            # Model expects (1,H,W,3), tiles are already resized in preprocessor
            tile_0 = np.expand_dims(curr_tiles[0], axis=0)
            tile_1 = np.expand_dims(curr_tiles[1], axis=0)
            tile_2 = np.expand_dims(curr_tiles[2], axis=0)
            tile_3 = np.expand_dims(curr_tiles[3], axis=0)
            tile_4 = np.expand_dims(curr_tiles[4], axis=0)

            result = model.predict([tile_0, tile_1, tile_2, tile_3, tile_4])
            pred = int(np.argmax(result, -1)[0])
            pred_calls += 1
            if pred_calls <= 10:
                logger.info(f"inference: pred r={row_index1} c={col_index1} class={pred}")

            start_y, end_y = (col_index1 * base_patch_size), (col_index1 * base_patch_size) + base_patch_size
            start_x, end_x = (row_index1 * base_patch_size), (row_index1 * base_patch_size) + base_patch_size
            mask[start_y:end_y, start_x:end_x, :] = configs.color_code[str(pred)]

    return mask


# def run_image_inference(image_path: str, save_overlay: bool = True) -> Dict[str, str]:
#     """
#     Runs inference for a single image and saves results under static/amcnn_inference.

#     Returns:
#       { "mask_path": ".../masks/<name>.png", "overlay_path": ".../overlays/<name>_overlay.png" (if saved) }
#     """
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     model, patcher, configs = _load_inference_components()
#     masks_dir, overlays_dir = _get_output_dirs()

#     # Load and normalize image to [0,1]
#     img = cv2.imread(image_path)
#     if img is None:
#         raise RuntimeError(f"Failed to load image: {image_path}")
#     img_norm = img / 255.0

#     # Predict mask
#     mask = infer_image(img_norm, model, patcher, configs)
#     mask_cropped = mask[:img.shape[0], :img.shape[1], :].astype('uint8')

#     # Build output paths
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     mask_path = os.path.join(masks_dir, f"{base_name}.png")
#     cv2.imwrite(mask_path, mask_cropped)

#     overlay_path = ""
#     if save_overlay:
#         overlay_path = os.path.join(overlays_dir, f"{base_name}_overlay.png")
#         overlay = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 5))
#         plt.axis('off')
#         plt.imshow(overlay)
#         plt.imshow(mask_cropped, alpha=0.4)
#         plt.tight_layout()
#         plt.savefig(overlay_path, dpi=300, bbox_inches='tight', pad_inches=0)
#         plt.close()

#     return {"mask_path": mask_path, "overlay_path": overlay_path}


def run_folder_inference(images_dir: Optional[str] = None, save_overlay: bool = True, weights_local_path: Optional[str] = None) -> Dict[str, list]:
    """
    Runs inference for all images in a folder. If images_dir is None, uses dynamic path:
      static/orig_images from RIEGL_PARSER_CONFIG.
    """
    if images_dir is None:
        images_dir = _get_images_dir()

    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"Images directory not found: {images_dir}")

    # Preload components once for all images (allowing weights override)
    model, preprocessor, configs = _load_inference_components(weights_local_path)
    masks_dir, overlays_dir = _get_output_dirs()

    results = []
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_norm = img / 255.0

        mask = infer_image(img_norm, model, preprocessor, configs)
        mask_cropped = mask[:img.shape[0], :img.shape[1], :].astype('uint8')

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{base_name}.png")
        cv2.imwrite(mask_path, mask_cropped)

        overlay_path = ""
        if save_overlay:
            overlay_path = os.path.join(overlays_dir, f"{base_name}_overlay.png")
            overlay = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.axis('off')
            plt.imshow(overlay)
            plt.imshow(mask_cropped, alpha=0.4)
            plt.tight_layout()
            plt.savefig(overlay_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        results.append({"image": img_path, "mask_path": mask_path, "overlay_path": overlay_path})

    return {"results": results, "masks_dir": masks_dir, "overlays_dir": overlays_dir}