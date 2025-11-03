import os
import asyncio
import tempfile
import importlib
from bson import ObjectId
from app.database.conn import mongo_client
from config import database_config
from app.services.s3.s3_operations import S3Operations
from app.utils.logger_utils import logger

def normalize_component_name(name: str) -> str:
    if not name:
        return ""
    return name.replace(" ", "_")

async def get_training_run(train_run_id: str):
    try:
        if not train_run_id and not isinstance(train_run_id, str):
            raise Exception("train_run_id is required and must be a string")
        train_run = await mongo_client.database[database_config["TRAIN_RUN_COLLECTION"]].find_one({"_id": ObjectId(train_run_id)})
        if not train_run or not train_run.get("status") == "completed":
            raise Exception("train_run not found or not completed")
        else:
            return train_run
    except Exception as e:
        raise Exception(str(e))

async def get_train_config(train_config_id: str):
    try:
        if not train_config_id and not isinstance(train_config_id, str):
            raise Exception("train_config_id is required and must be a string")
        train_config = await mongo_client.database[database_config["TRAIN_CONFIG_COLLECTION"]].find_one({"_id": ObjectId(train_config_id)})
        if not train_config:
            raise Exception("train_config not found")
        else:
            return train_config
    except Exception as e:
        raise Exception(str(e))

async def load_bucket_config(project_id: str) -> dict:
    try:
        db = mongo_client.database
        bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
        config = await bucket_configs.find_one({"project_id": project_id})
        if not config:
            raise Exception(f"Bucket config not found for project: {project_id}")
        else:
            return config
    except Exception as e:
        raise Exception(str(e))

async def trigger_inference(train_run_id: str):
    """
    Non-blocking: spawns background task and returns immediately.
    """
    if not train_run_id or not isinstance(train_run_id, str):
        raise Exception("train_run_id is required and must be a string")

    # Fire-and-forget background job
    logger.info(f"inference: schedule run_id={train_run_id}")
    asyncio.create_task(_run_inference_background(train_run_id))

    return {
        "status": "started",
        "train_run_id": train_run_id,
        "message": "Inference started in background"
    }

async def _run_inference_background(train_run_id: str):
    """
    Background pipeline:
      1) Load train_run from DB (contains train_config_id)
      2) Load config/run/bucket config
      3) Download trained weights (output_weights_s3_url) → local temp
      4) Upload same weights → folder_structure['inference_input_model']
      5) Run model folder inference with local weights (non-blocking via to_thread)
      6) Upload:
         - weights → folder_structure['inference_output_labels']
         - masks → folder_structure['inference_output_metadata']/masks/<original>.png
         - labelized_images → .../labelized_images/<original>.png
    """
    s3 = S3Operations()
    tmp_dir = tempfile.mkdtemp(prefix="inference_")

    try:
        # Load DB docs
        logger.info(f"inference: start run_id={train_run_id}")
        train_run = await get_training_run(train_run_id)
        
        # Extract train_config_id from train_run document
        train_config_id = train_run.get("train_config_id")
        if not train_config_id:
            raise Exception("train_config_id not found in train_run document")
        
        train_config = await get_train_config(train_config_id)
        bucket_config = await load_bucket_config(train_config.get("project_id"))
        folder_structure = (bucket_config.get("folder_structure") or {})

        # Required S3 roots
        inference_input_model = folder_structure.get("inference_input_model")
        inference_output_labels = folder_structure.get("inference_output_labels")
        inference_output_metadata = folder_structure.get("inference_output_metadata")
        if not inference_input_model or not inference_output_labels or not inference_output_metadata:
            raise Exception("Bucket config missing required inference paths (inference_input_model, inference_output_labels, inference_output_metadata)")

        # Import model-specific inference module
        model_name = normalize_component_name(train_config["metadata"].get("model_name"))
        model_version = normalize_component_name(train_config.get("model_version"))
        inference_module_path = f"app.deep_models.Algorithms.{model_name}.{model_version}.inference"
        inference_module = importlib.import_module(inference_module_path)
        run_folder_inference = getattr(inference_module, "run_folder_inference")
        logger.info(f"inference: module loaded model={model_name} version={model_version}")

        # 1) Download trained weights from S3 (full URL in run)
        weights_s3_url = train_run.get("output_weights_s3_url")
        if not weights_s3_url:
            raise Exception("No trained model weights found for this training run")
        local_weights_path = os.path.join(tmp_dir, os.path.basename(weights_s3_url))
        dl = await asyncio.to_thread(s3.download_file, weights_s3_url, local_weights_path)
        if not dl or not dl.get("success"):
            raise Exception(f"Failed to download weights: {weights_s3_url}")
        logger.info("inference: weights downloaded")

        # 2) Upload same weights to inference_input_model (for audit/traceability)
        inp_dest = f"{inference_input_model.rstrip('/')}/{train_run_id}/{os.path.basename(local_weights_path)}"
        up_inp = await asyncio.to_thread(s3.upload_file, local_weights_path, inp_dest)
        if not up_inp or not up_inp.get("success"):
            raise Exception(f"Failed to upload weights to inference_input_model: {inp_dest}")
        logger.info("inference: input weights uploaded")

        # 3) Run folder inference using the local weights (blocking work offloaded)
        #    This writes local masks/overlays into static/amcnn_inference/...
        #    Returns: {"results":[{"image", "mask_path", "overlay_path"}, ...], "masks_dir", "overlays_dir"}
        logger.info("inference: folder inference begin")
        # inf_result = await asyncio.to_thread(run_folder_inference, None, True, local_weights_path)
        def _run_any_inference(inference_module, local_weights_path, model_name_str, images_dir=None, out_dir="viz_instances"):
            # For Detectron2: use run_inference directly (it now accepts absolute paths)
            if model_name_str.upper() == "DETECTRON2" and hasattr(inference_module, "run_inference"):
                from pathlib import Path
                from app.deep_models.Algorithms.DETECTRON2.v1.config import get_output_dir
                
                # Pass absolute path directly - Detectron2's run_inference now handles it
                result_list = inference_module.run_inference(
                    weights=str(local_weights_path),  # absolute path
                    images_dir=images_dir,
                    out_dir=out_dir,
                )
                
                # Transform Detectron2 results to match expected format
                # Detectron2 saves visualizations as {image_stem}_pred.png in out_dir
                det2_out_dir = Path(get_output_dir()) / out_dir
                transformed_results = []
                
                if isinstance(result_list, list):
                    for pred_dict in result_list:
                        # Detectron2 returns dicts with "file_name" key
                        file_name = pred_dict.get("file_name", "")
                        if not file_name:
                            continue
                        
                        # Build paths
                        image_path = str(Path(images_dir) / file_name) if images_dir else None
                        overlay_path = str(det2_out_dir / f"{Path(file_name).stem}_pred.png")
                        
                        # Detectron2 doesn't generate separate masks, use overlay as both
                        mask_path = overlay_path  # Same as overlay for Detectron2
                        
                        transformed_results.append({
                            "image": image_path,
                            "mask_path": mask_path if Path(overlay_path).exists() else None,
                            "overlay_path": overlay_path if Path(overlay_path).exists() else None
                        })
                
                return {
                    "results": transformed_results
                }
            # For AMCNN: use run_folder_inference with correct signature
            if model_name_str.upper() == "AMCNN" and hasattr(inference_module, "run_folder_inference"):
                return inference_module.run_folder_inference(
                    images_dir=images_dir,
                    save_overlay=True,
                    weights_local_path=str(local_weights_path),  # absolute path
                )
            # For other models: use run_inference if available
            if hasattr(inference_module, "run_inference"):
                return inference_module.run_inference(
                    weights=str(local_weights_path),  # absolute path
                    images_dir=images_dir,
                    out_dir=out_dir,
                )
            # Fallback to run_folder_inference for other models
            if hasattr(inference_module, "run_folder_inference"):
                return inference_module.run_folder_inference(
                    weights=str(local_weights_path),  # absolute path
                    out_dir=out_dir,
                )
            # Last resort generic 'infer(weights_path)'
            if hasattr(inference_module, "infer"):
                return inference_module.infer(str(local_weights_path))
            raise RuntimeError("No supported inference entry found in module")

        # Resolve images_dir from model config when available
        images_dir = None
        try:
            if model_name.upper() == "DETECTRON2":
                from app.deep_models.Algorithms.DETECTRON2.v1.config import get_images_dir as d2_get_images_dir
                images_dir = str(d2_get_images_dir())
            elif model_name.upper() == "AMCNN":
                from app.deep_models.Algorithms.AMCNN.v1.config import AMCNN_V1_CONFIG
                images_dir = str(AMCNN_V1_CONFIG.get_images_dir())
        except Exception:
            images_dir = None

        # Use the adapter (works for current and future models)
        inf_result = await asyncio.to_thread(
            _run_any_inference, inference_module, local_weights_path, model_name, images_dir, "viz_instances"
        )
        logger.info("inference: folder inference done")

        # 4) Upload final weights snapshot to inference_output_labels
        out_dest = f"{inference_output_labels.rstrip('/')}/{train_run_id}/{os.path.basename(local_weights_path)}"
        up_out = await asyncio.to_thread(s3.upload_file, local_weights_path, out_dest)
        if not up_out or not up_out.get("success"):
            raise Exception(f"Failed to upload weights to inference_output_labels: {out_dest}")
        logger.info("inference: output weights uploaded")

        # 5) Upload masks and labelized images (overlay) to inference_output_metadata
        results = inf_result.get("results", []) if isinstance(inf_result, dict) else []
        uploaded_masks = 0
        uploaded_overlays = 0
        for item in results:
            img_path = item.get("image")
            mask_path = item.get("mask_path")
            overlay_path = item.get("overlay_path")

            if not img_path:
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]

            # {run_id}/masks/<original>.png
            if mask_path and os.path.exists(mask_path):
                s3_mask = f"{inference_output_metadata.rstrip('/')}/{train_run_id}/masks/{base}.png"
                await asyncio.to_thread(s3.upload_file, mask_path, s3_mask)
                uploaded_masks += 1

            # {run_id}/labelized_images/<original>.png
            if overlay_path and os.path.exists(overlay_path):
                s3_overlay = f"{inference_output_metadata.rstrip('/')}/{train_run_id}/labelized_images/{base}.png"
                await asyncio.to_thread(s3.upload_file, overlay_path, s3_overlay)
                uploaded_overlays += 1
        logger.info(f"inference: uploaded masks={uploaded_masks} overlays={uploaded_overlays}")
        logger.info(f"inference: complete run_id={train_run_id}")
        
    except Exception as e:
        logger.error(f"inference: failed run_id={train_run_id} error={str(e)}")
        pass
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir)
        except Exception:
            pass