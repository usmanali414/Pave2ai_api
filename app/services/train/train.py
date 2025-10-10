"""
Training service that orchestrates the complete training pipeline.
"""
import sys
import os
from datetime import datetime
from typing import Any, Dict

# Add the project root to Python path for orchestrator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bson import ObjectId
from app.database.conn import mongo_client
from config import database_config
from app.deep_models.AMCNN_orchestrator import run_amcnn_training
from app.utils.logger_utils import logger


async def start_training(train_config_id: str) -> Dict[str, Any]:
    """
    Start training using the appropriate orchestrator based on train config.
    
    Args:
        train_config_id: ID of the train configuration
        
    Returns:
        Training results
    """
    db = mongo_client.database
    train_configs = db[database_config["TRAIN_CONFIG_COLLECTION"]]
    train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]

    # Validate and fetch train_config by ObjectId
    try:
        cfg_oid = ObjectId(train_config_id)
    except Exception:
        raise ValueError("invalid train_config_id")

    config = await train_configs.find_one({"_id": cfg_oid})
    if not config:
        raise ValueError("train_config not found")

    # Validate train config before starting
    await _validate_train_config(config)

    # Check if a training run is already active for this config
    existing_run = await train_runs.find_one({
        "train_config_id": train_config_id,
        "status": "training",
        "ended_at": None,
    })
    if existing_run:
        # Indicate training already running
        raise RuntimeError("training already running for this configuration")

    # Create train run record
    now = datetime.utcnow()
    run_doc = {
        "train_config_id": train_config_id,
        "status": "training",
        "created_at": now,
        "updated_at": now,
        "ended_at": None,
        "step_status": {
            "loading_data": None,
            "training": None,
            "saving_model": None
        }
    }
    result = await train_runs.insert_one(run_doc)
    run_id = str(result.inserted_id)

    try:
        # Determine which orchestrator to use based on model configuration
        model_name = config["metadata"].get("model_name", "AMCNN")
        model_version = config.get("model_version", "v1")
        
        logger.info(f"Starting training for {model_name} v{model_version}")
        
        # Route to appropriate orchestrator
        if model_name == "AMCNN":
            training_results = await run_amcnn_training(train_config_id)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logger.info(f"Training completed successfully for config {train_config_id}")
        return training_results

    except Exception as e:
        # Update run as failed
        await train_runs.update_one(
            {"_id": ObjectId(run_id)},
            {
                "$set": {
                    "status": "failed", 
                    "error": str(e), 
                    "updated_at": datetime.utcnow(), 
                    "ended_at": datetime.utcnow()
                }
            }
        )
        logger.error(f"Training failed for config {train_config_id}: {str(e)}")
        raise


async def _validate_train_config(config: Dict[str, Any]) -> None:
    """
    Validate train configuration before starting training.
    
    Args:
        config: Train configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    required_fields = ["name", "tenant_id", "project_id", "model_version", "metadata"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate metadata
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    
    # Check required metadata fields
    required_metadata = ["data_parser", "model_name"]
    for field in required_metadata:
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")
    
    # Validate data parser
    data_parser = metadata.get("data_parser")
    supported_parsers = ["RIEGL_PARSER"]
    if data_parser not in supported_parsers:
        raise ValueError(f"Unsupported data parser: {data_parser}. Supported: {supported_parsers}")
    
    # Validate model
    model_name = metadata.get("model_name")
    model_version = config.get("model_version")
    supported_models = {
        "AMCNN": ["v1"]
    }
    
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(supported_models.keys())}")
    
    if model_version not in supported_models[model_name]:
        raise ValueError(f"Unsupported {model_name} version: {model_version}. Supported: {supported_models[model_name]}")
    
    # Validate bucket config exists
    db = mongo_client.database
    bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
    bucket_cfg = await bucket_configs.find_one({"project_id": config["project_id"]})
    
    if not bucket_cfg:
        raise ValueError(f"bucket_config not found for project: {config['project_id']}")
    
    # Validate required folder structure
    folder_structure = bucket_cfg.get("folder_structure", {})
    required_paths = ["data", "train_output_model"]
    for path in required_paths:
        if path not in folder_structure or not folder_structure[path]:
            raise ValueError(f"Required path '{path}' not configured in bucket_config")
    
    logger.info(f"Train config validation passed for {config['name']}")


async def get_training_status(train_run_id: str) -> Dict[str, Any]:
    """
    Get the current status of a training run.
    
    Args:
        train_run_id: ID of the training run
        
    Returns:
        Training run status information
    """
    try:
        db = mongo_client.database
        train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
        
        run = await train_runs.find_one({"_id": ObjectId(train_run_id)})
        
        if not run:
            raise ValueError(f"Training run not found: {train_run_id}")
        
        return {
            "train_run_id": train_run_id,
            "train_config_id": run["train_config_id"],
            "status": run["status"],
            "step_status": run.get("step_status", {}),
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "ended_at": run.get("ended_at"),
            "results": run.get("results"),
            "error": run.get("error")
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise


async def cancel_training(train_run_id: str) -> Dict[str, Any]:
    """
    Cancel a running training job.
    
    Args:
        train_run_id: ID of the training run to cancel
        
    Returns:
        Cancellation result
    """
    try:
        db = mongo_client.database
        train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
        
        # Check if run exists and is still running
        run = await train_runs.find_one({
            "_id": ObjectId(train_run_id),
            "status": "training"
        })
        
        if not run:
            raise ValueError(f"Training run not found or not running: {train_run_id}")
        
        # Update status to cancelled
        await train_runs.update_one(
            {"_id": ObjectId(train_run_id)},
            {
                "$set": {
                    "status": "cancelled",
                    "updated_at": datetime.utcnow(),
                    "ended_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Training run {train_run_id} cancelled")
        
        return {
            "train_run_id": train_run_id,
            "status": "cancelled",
            "message": "Training cancelled successfully"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling training: {str(e)}")
        raise