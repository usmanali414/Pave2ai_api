import os
import asyncio
from datetime import datetime
from typing import Dict, Any

import importlib
from bson import ObjectId
from app.database.conn import mongo_client
from config import database_config
from app.utils.logger_utils import logger
from app.services.s3.s3_operations import S3Operations
from app.deep_models.Algorithms.DETECTRON2.v1.config import DATASET_ROOT
from app.deep_models.Algorithms.DETECTRON2.v1.DETECTRON2 import DETECTRON2

# Data loader
from app.deep_models.data_parser.RIEGL_PARSER.RIEGL_PARSER import RIEGL_PARSER

class DETECTRON2Orchestrator:
    def __init__(self):
        self.model = None  # Will be lazily initialized after paths are set
    
    def normalize_component_name(self, name: str) -> str:
        if not name:
            return ""
        return name.replace(" ", "_")

    async def _update_step_status(self, train_run_id: str, step: str, status: str, extra: Dict[str, Any] = None):
        """
        Update a single step status field atomically.
        """
        db = mongo_client.database
        train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
        field = f"step_status.{step}"
        value = status if status in ["pending", "running", "completed", "failed"] else status
        update_doc = {
            "$set": {
                field: value,
                "updated_at": datetime.utcnow()
            }
        }
        if extra:
            update_doc["$set"].update(extra)
        await train_runs.update_one({"_id": ObjectId(train_run_id)}, update_doc)


    async def _update_s3_urls(self, train_run_id: str, urls: Dict[str, Any]):
        """
        Update S3 URL fields on the training run document.
        """
        if not urls:
            return
        db = mongo_client.database
        train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
        urls["updated_at"] = datetime.utcnow()
        await train_runs.update_one({"_id": ObjectId(train_run_id)}, {"$set": urls})


    async def _load_train_config(self, train_config_id: str) -> Dict[str, Any]:
        db = mongo_client.database
        train_configs = db[database_config["TRAIN_CONFIG_COLLECTION"]]
        cfg = await train_configs.find_one({"_id": ObjectId(train_config_id)})
        if not cfg:
            raise ValueError(f"train_config not found: {train_config_id}")
        return cfg


    async def _load_bucket_config(self, project_id: str) -> Dict[str, Any]:
        db = mongo_client.database
        bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
        bcfg = await bucket_configs.find_one({"project_id": project_id})
        if not bcfg:
            raise ValueError(f"bucket_config not found for project: {project_id}")
        return bcfg


    def _fs_paths_from_bucket(self, bucket_cfg: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract folder_structure paths used across steps. Keeps parity with AMCNN structure.
        """
        fs = (bucket_cfg.get("folder_structure") or {})
        return {
            "preprocessed_data": fs.get("preprocessed_data", ""),
            "annotate_label": fs.get("annotate_label", ""),
            "train_input_model": fs.get("train_input_model", ""),
            "train_output_model": fs.get("train_output_model", ""),
            "train_output_logs": fs.get("train_output_logs", ""),            # optional
            "train_output_metadata": fs.get("train_output_metadata", ""),  # optional
            "evaluation_output_metadata": fs.get("evaluation_output_metadata", ""),  # optional
        }


    # async def _step_loading_data(train_config: Dict[str, Any], bucket_cfg: Dict[str, Any], train_run_id: str) -> Dict[str, Any]:
    #     """
    #     Use RIEGL_PARSER to materialize local data (same flow as AMCNN).
    #     """
    #     await _update_step_status(train_run_id, "loading_data", "running")

    #     parser = RIEGL_PARSER()
    #     # This is a blocking step; run in thread
    #     transferred = await asyncio.to_thread(
    #         parser.load_data,
    #         project_id=train_config["project_id"],
    #         tenant_id=train_config["tenant_id"]
    #     )
    #     # Validate shape/keys as needed; parser already validates structure
    #     await _update_step_status(train_run_id, "loading_data", "completed")
    #     return transferred

    async def _step_loading_data(self, train_config: Dict[str, Any], bucket_cfg: Dict[str, Any], train_run_id: str) -> Dict[str, Any]:
        """
        Use RIEGL_PARSER to materialize local data in model-scoped directories.
        Also injects paths into DETECTRON2 config for runtime use.
        """
        await self._update_step_status(train_run_id, "loading_data", "running")

        parser = RIEGL_PARSER()
        # load_data is async and expects train_config (not project_id/tenant_id)
        transferred = await parser.load_data(train_config)
        
        # Validate
        if not parser.validate_transferred_data(transferred):
            raise ValueError("Data validation failed after transfer")
        
        # Inject paths into DETECTRON2 config for runtime use
        from app.deep_models.Algorithms.DETECTRON2.v1 import config as det2_config
        det2_config.set_runtime_paths(transferred["local_directories"])
        
        await self._update_step_status(train_run_id, "loading_data", "completed")
        return transferred


    async def _step_preprocessing(self, train_config: Dict[str, Any], bucket_cfg: Dict[str, Any], local_data: Dict[str, Any], train_run_id: str):
        """
        Optional: If DETECTRON2 needs conversions (COCO/label mappings/tiling), do it here.
        For now, mark as completed to keep parity. Wrap any blocking logic in to_thread.
        """
        await self._update_step_status(train_run_id, "preprocessing", "in_progress")
        try:
            model_version = self.normalize_component_name(train_config.get("model_version"))
            # Dynamically import the appropriate preprocessor based on model name and version
            model_name = self.normalize_component_name(train_config["metadata"].get("model_name"))
            preprocessor_module_path = f"app.deep_models.Algorithms.{model_name}.{model_version}.preprocessor"
            preprocessor_module = importlib.import_module(preprocessor_module_path)
            # Example placeholder:
            await asyncio.to_thread(preprocessor_module.main)
            await self._update_step_status(train_run_id, "preprocessing", "completed")
        except Exception as e:
            logger.error(f"Error preprocessing: {str(e)}")
            await self._update_step_status(train_run_id, "preprocessing", "failed", {"error": str(e)})
            raise

    async def _step_training(self, train_config: Dict[str, Any], bucket_cfg: Dict[str, Any], local_data: Dict[str, Any], train_run_id: str) -> Dict[str, Any]:
        """
        Execute DETECTRON2 training and upload artifacts to S3.
        Return dict with keys matching AMCNN’s result shape:
        {
            "initial_weights_path": "... optional ...",
            "output_weights_path": "s3://.../model.h5 or .pth",
            "output_logs_path": "s3://.../logs.csv or logs dir"
        }
        """
        await self._update_step_status(train_run_id, "training", "running")

        fs = self._fs_paths_from_bucket(bucket_cfg)
        train_config = train_config.get("metadata", {})
        initial_weights_flag = train_config.get("initial_weights", False)
        
        # Build S3 paths with proper handling of empty strings
        train_input_base = (fs.get('train_input_model') or '').rstrip('/')
        train_output_base = (fs.get('train_output_model') or '').rstrip('/')
        logs_base = (fs.get('train_output_metadata') or '').rstrip('/')
        
        initial_weights_s3 = f"{train_input_base}/{train_run_id}/detectron2_weights.pth" if train_input_base else ""
        final_weights_s3 = f"{train_output_base}/{train_run_id}/detectron2_weights.pth" if train_output_base else ""
        logs_s3 = f"{logs_base}/{train_run_id}" if logs_base else ""

        # Initialize model lazily (after paths are set in _step_loading_data)
        if self.model is None:
            self.model = DETECTRON2()

        await self.model.train(initial_weights_flag, initial_weights_s3, final_weights_s3, logs_s3)

        await self._update_step_status(train_run_id, "training", "completed")

        return {
            "initial_weights_path": initial_weights_s3,
            "output_weights_path": final_weights_s3,
            "output_logs_path": logs_s3
        }


    async def _step_saving_model(self, train_config: Dict[str, Any], bucket_cfg: Dict[str, Any], train_run_id: str, training_result: Dict[str, Any]):
        """
        If there is any finalization or snapshotting (e.g. copy best ckpt), handle here.
        """
        await self._update_step_status(train_run_id, "saving_model", "running")

        urls = {}
        if training_result:
            if training_result.get("output_weights_path"):
                urls["output_weights_s3_url"] = training_result["output_weights_path"]
            if training_result.get("output_logs_path"):
                urls["output_logs_s3_url"] = training_result["output_logs_path"]
            if training_result.get("initial_weights_path"):
                urls["input_weights_s3_url"] = training_result["initial_weights_path"]

        if urls:
            await self._update_s3_urls(train_run_id, urls)

        await self._update_step_status(train_run_id, "saving_model", "completed")


async def run_detectron2_training(train_config_id: str, train_run_id: str) -> Dict[str, Any]:
    """
    Entry point for DETECTRON2 training (non-resume).
    Mirrors AMCNN orchestrator signature expected by app/services/train/train.py
    """
    orchestrator = DETECTRON2Orchestrator()
    logger.info(f"det2: start train config={train_config_id} run={train_run_id}")
    train_config = await orchestrator._load_train_config(train_config_id)
    bucket_cfg = await orchestrator._load_bucket_config(train_config["project_id"])

    try:
        # loading_data
        local_data = await orchestrator._step_loading_data(train_config, bucket_cfg, train_run_id)

        # preprocessing
        await orchestrator._step_preprocessing(train_config, bucket_cfg, local_data, train_run_id)

        # training
        training_result = await orchestrator._step_training(train_config, bucket_cfg, local_data, train_run_id)

        # # saving_model
        # await orchestrator._step_saving_model(train_config, bucket_cfg, train_run_id, training_result)

        logger.info(f"det2: completed run={train_run_id}")
        return {
            "status": "completed",
            "train_run_id": train_run_id
        }
    except Exception as e:
        logger.error(f"det2: failed run={train_run_id} error={str(e)}")
        # Best effort mark step failed
        await orchestrator._update_step_status(train_run_id, "saving_model", "failed", {"error": str(e)})
        raise


async def run_detectron2_training_with_resume(self, train_config_id: str, train_run_id: str, resume_from: str) -> Dict[str, Any]:
    """
    Entry point for DETECTRON2 resume training. Resume from any of:
      loading_data → preprocessing → training → saving_model
    """
    orchestrator = DETECTRON2Orchestrator()
    logger.info(f"det2: resume from={resume_from} config={train_config_id} run={train_run_id}")
    train_config = await orchestrator._load_train_config(train_config_id)
    bucket_cfg = await orchestrator._load_bucket_config(train_config["project_id"])

    try:
        local_data = None

        if resume_from in ["loading_data", None]:
            local_data = await orchestrator._step_loading_data(train_config, bucket_cfg, train_run_id)
            resume_from = "preprocessing"
        if resume_from == "preprocessing":
            if local_data is None:
                # If resuming mid-pipeline, load or reconstruct local_data as needed (optional)
                local_data = {}
            await orchestrator._step_preprocessing(train_config, bucket_cfg, local_data, train_run_id)
            resume_from = "training"
        if resume_from == "training":
            training_result = await orchestrator._step_training(train_config, bucket_cfg, local_data or {}, train_run_id)
            resume_from = "saving_model"
        if resume_from == "saving_model":
            await orchestrator._step_saving_model(train_config, bucket_cfg, train_run_id, locals().get("training_result", {}))

        logger.info(f"det2: resume complete run={train_run_id}")
        return {
            "status": "completed",
            "train_run_id": train_run_id
        }
    except Exception as e:
        logger.error(f"det2: resume failed run={train_run_id} error={str(e)}")
        await orchestrator._update_step_status(train_run_id, resume_from or "training", "failed", {"error": str(e)})
        raise