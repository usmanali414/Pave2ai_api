# RIEGL_PARSER_CONFIG = {
#     "local_storage": {
#         "base_path": "static",
#         "dataset_dir": "dataset",
#         "images_dir": "orig_images",
#         "jsons_dir": "jsons",
#         "masks_dir": "masks",
#         "patches_dir": "split_patches_data",
#         "inference_base_dir": "amcnn_inference",
#         "inference_masks_dir": "masks",
#         "inference_overlays_dir": "overlays"
#     }
# }

RIEGL_PARSER_CONFIG = {
    "local_storage": {
        "base_path": "static",
        "dataset_dir": "dataset",
        "images_dir": "orig_images",
        "jsons_dir": "jsons",
        "masks_dir": "masks",
        "patches_dir": "split_patches_data",
        "inference_base_dir": "inference",  # Changed: generic name instead of "amcnn_inference"
        "inference_masks_dir": "masks",
        "inference_overlays_dir": "overlays"
    }
}