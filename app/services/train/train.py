import sys
import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

# Add the project root to Python path for orchestrator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bson import ObjectId
from app.database.conn import mongo_client
from config import database_config
from app.utils.logger_utils import logger
import importlib

# Global thread pool executor for CPU-bound training tasks
# Max 2 concurrent trainings to prevent resource exhaustion
training_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training_")


def normalize_component_name(name: str) -> str:
    """
    Normalize component names by replacing spaces with underscores.
    
    Args:
        name: Component name (e.g., "RIEGL PARSER", "AM CNN")
        
    Returns:
        Normalized name (e.g., "RIEGL_PARSER", "AM_CNN")
    """
    if not name:
        return ""
    return name.replace(" ", "_")

# Add after line 35:
async def resume_training(train_run_id: str) -> Dict[str, Any]:
    """
    Resume a failed or cancelled training run from the first incomplete step.
    
    Args:
        train_run_id: ID of the training run to resume
        
    Returns:
        Resume result information
    """
    db = mongo_client.database
    train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
    
    try:
        # Load existing run
        run = await train_runs.find_one({"_id": ObjectId(train_run_id)})
        if not run:
            raise ValueError(f"Training run not found: {train_run_id}")
        
        # Check if training can be resumed
        if run["status"] not in ["failed", "cancelled"]:
            raise RuntimeError("Training cannot be resumed. Only failed or cancelled trainings can be resumed.")
        
        # Check for concurrent resume attempts
        existing_resume = await train_runs.find_one({
            "_id": ObjectId(train_run_id),
            "status": "resuming"
        })
        if existing_resume:
            raise RuntimeError("Training is already being resumed")
        
        # Validate train config exists before starting resume
        train_config_id = run.get("train_config_id")
        if not train_config_id:
            raise ValueError("Training run has no train_config_id")
        
        train_configs = db[database_config["TRAIN_CONFIG_COLLECTION"]]
        config = await train_configs.find_one({"_id": ObjectId(train_config_id)})
        if not config:
            raise ValueError(f"Train config not found: {train_config_id}. The configuration may have been deleted.")
        
        # Determine resume point
        step_status = run.get("step_status", {})
        resume_from = _determine_resume_point(step_status)
        
        if not resume_from:
            raise RuntimeError("All steps are completed. Nothing to resume.")
        
        # Update status to resuming
        await train_runs.update_one(
            {"_id": ObjectId(train_run_id)},
            {
                "$set": {
                    "status": "resuming",
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Start resume process
        asyncio.create_task(_run_resume_training(run, resume_from))
        
        logger.info(f"Training resume started for run {train_run_id} from step: {resume_from}")
        
        return {
            "train_run_id": train_run_id,
            "status": "resuming",
            "resume_from": resume_from,
            "message": f"Training resumed from {resume_from} step"
        }
        
    except Exception as e:
        logger.error(f"Error resuming training: {str(e)}")
        raise


async def _run_resume_training(run: Dict[str, Any], resume_from: str) -> None:
    """
    Run resume training in background.
    
    Args:
        run: Training run document
        resume_from: Step to resume from
    """
    train_run_id = str(run["_id"])
    train_config_id = run["train_config_id"]
    
    try:
        # Load train config
        train_configs = mongo_client.database[database_config["TRAIN_CONFIG_COLLECTION"]]
        config = await train_configs.find_one({"_id": ObjectId(train_config_id)})
        
        if not config:
            # Update training run status to indicate config not found
            train_runs = mongo_client.database[database_config["TRAIN_RUN_COLLECTION"]]
            await train_runs.update_one(
                {"_id": ObjectId(train_run_id)},
                {
                    "$set": {
                        "status": "failed",
                        "error": f"Train config not found: {train_config_id}. The configuration may have been deleted.",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            raise ValueError(f"Train config not found: {train_config_id}. The configuration may have been deleted.")
        
        # Run orchestrator with resume
        orchestrator_result = await _run_orchestrator_with_resume(config, train_run_id, resume_from)
        
        # Update final status
        train_runs = mongo_client.database[database_config["TRAIN_RUN_COLLECTION"]]
        await train_runs.update_one(
            {"_id": ObjectId(train_run_id)},
            {
                "$set": {
                    "status": "completed",
                    "results": orchestrator_result,
                    "updated_at": datetime.utcnow(),
                    "ended_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Resume training completed successfully for run {train_run_id}")
        
    except Exception as e:
        # Update run as failed
        train_runs = mongo_client.database[database_config["TRAIN_RUN_COLLECTION"]]
        await train_runs.update_one(
            {"_id": ObjectId(train_run_id)},
            {
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "updated_at": datetime.utcnow(),
                    "ended_at": datetime.utcnow()
                }
            }
        )
        logger.error(f"Resume training failed for run {train_run_id}: {str(e)}")


def _determine_resume_point(step_status: Dict[str, Any]) -> str:
    """
    Determine which step to resume from based on step status.
    
    Args:
        step_status: Current step status dictionary
        
    Returns:
        Step name to resume from, or None if all completed
    """
    steps = ["loading_data", "preprocessing", "training", "saving_model"]
    
    for step in steps:
        step_info = step_status.get(step, {})
        if isinstance(step_info, dict):
            status = step_info.get("status")
        else:
            status = step_info  # Handle old format
        
        if status in [None, "failed", "in_progress"]:
            return step
        elif status == "completed":
            continue  # Skip completed steps
    
    return None  # All steps completed


async def _run_orchestrator_with_resume(train_config: Dict[str, Any], train_run_id: str, resume_from: str) -> Dict[str, Any]:
    """
    Run orchestrator with resume capability.
    
    Args:
        train_config: Training configuration
        train_run_id: Training run ID
        resume_from: Step to resume from
        
    Returns:
        Training results
    """
    try:
        model_name = normalize_component_name(train_config["metadata"].get("model_name"))
        model_version = normalize_component_name(train_config.get("model_version"))
        
        logger.info(f"Running {model_name} orchestrator with resume from {resume_from}")
        
        # Dynamically import and run orchestrator
        orchestrator_module_path = f"app.deep_models.{model_name}_orchestrator"
        orchestrator_module = importlib.import_module(orchestrator_module_path)
        orchestrator_function_name = f"run_{model_name.lower()}_training_with_resume"
        orchestrator_function = getattr(orchestrator_module, orchestrator_function_name)
        
        # Call orchestrator with resume capability
        result = await orchestrator_function(train_config["_id"], train_run_id, resume_from)
        return result
        
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load orchestrator for {model_name}: {str(e)}")
        raise ValueError(f"Unsupported model orchestrator: {model_name}. Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error running orchestrator with resume: {str(e)}")
        raise


async def get_all_training_runs(project_id: str = None, status: str = None) -> List[Dict[str, Any]]:
    """
    Get all training runs with optional filtering.
    
    Args:
        project_id: Optional project ID to filter by
        status: Optional status to filter by
        
    Returns:
        List of training runs
    """
    try:
        db = mongo_client.database
        train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
        
        # Build filter query
        filter_query = {}
        if project_id:
            # We need to join with train_configs to filter by project_id
            train_configs = db[database_config["TRAIN_CONFIG_COLLECTION"]]
            configs = await train_configs.find({"project_id": project_id}).to_list(length=None)
            config_ids = [str(config["_id"]) for config in configs]
            filter_query["train_config_id"] = {"$in": config_ids}
        
        if status:
            filter_query["status"] = status
        
        runs = await train_runs.find(filter_query).sort("created_at", -1).to_list(length=None)
        
        return [
            {
                "_id": str(run["_id"]),
                "train_run_id": str(run["_id"]),
                "train_config_id": run["train_config_id"],
                "status": run["status"],
                "step_status": run.get("step_status", {}),
                "created_at": run["created_at"],
                "updated_at": run["updated_at"],
                "ended_at": run.get("ended_at"),
                "error": run.get("error"),
                "input_weights_s3_url": run.get("input_weights_s3_url"),
                "output_weights_s3_url": run.get("output_weights_s3_url"),
                "output_logs_s3_url": run.get("output_logs_s3_url"),
                "evaluation_logs_s3_url": run.get("evaluation_logs_s3_url")
            }
            for run in runs
        ]
        
    except Exception as e:
        logger.error(f"Error getting all training runs: {str(e)}")
        raise


async def get_training_runs_for_config(train_config_id: str) -> List[Dict[str, Any]]:
    """
    Get all training runs for a specific train config.
    
    Args:
        train_config_id: ID of the train configuration
        
    Returns:
        List of training runs
    """
    try:
        db = mongo_client.database
        train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
        
        runs = await train_runs.find(
            {"train_config_id": train_config_id}
        ).sort("created_at", -1).to_list(length=None)
        
        return [
            {
                "train_run_id": str(run["_id"]),
                "train_config_id": run["train_config_id"],
                "status": run["status"],
                "step_status": run.get("step_status", {}),
                "created_at": run["created_at"],
                "updated_at": run["updated_at"],
                "ended_at": run.get("ended_at"),
                "error": run.get("error")
            }
            for run in runs
        ]
        
    except Exception as e:
        logger.error(f"Error getting training runs for config: {str(e)}")
        raise

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
        "step_status": {
            "loading_data": None,
            "preprocessing": None,
            "training": None,
            "saving_model": None
        },
        # canonical S3 url fields (initialized as null)
        "input_weights_s3_url": None,
        "output_weights_s3_url": None,
        "output_logs_s3_url": None,
        "evaluation_logs_s3_url": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    result = await train_runs.insert_one(run_doc)
    run_id = str(result.inserted_id)

    try:
        # Determine orchestrator based on train_config and call it
        orchestrator_result = await _run_orchestrator(config, run_id)
        
        # Use the orchestrator dispatcher
        training_results = orchestrator_result
        
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
    
    # Check required fields
    required_fields = ["tenant_id", "project_id", "model_version", "metadata"]
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
    
    # Validate data parser exists (normalize name first)
    data_parser = normalize_component_name(metadata.get("data_parser"))
    if not data_parser:
        raise ValueError("data_parser is required in metadata")
    
    # Validate model exists (normalize name first)
    model_name = normalize_component_name(metadata.get("model_name"))
    model_version = normalize_component_name(config.get("model_version"))
    
    if not model_name:
        raise ValueError("model_name is required in metadata")
    
    if not model_version:
        raise ValueError("model_version is required in train config")
    
    # Validate bucket config exists
    db = mongo_client.database
    bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
    bucket_cfg = await bucket_configs.find_one({"project_id": config["project_id"]})
    
    if not bucket_cfg:
        raise ValueError(f"bucket_config not found for project: {config['project_id']}")
    
    # Validate required folder structure (based on new bucket config structure)
    folder_structure = bucket_cfg.get("folder_structure", {})
    required_paths = ["preprocessed_data", "annotate_label", "train_output_model"]
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
            "model_logs_path": run.get("model_logs_path"),
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


async def _run_orchestrator(train_config: Dict[str, Any], train_run_id: str) -> Dict[str, Any]:
    try:
        model_name = normalize_component_name(train_config["metadata"].get("model_name"))
        model_version = normalize_component_name(train_config.get("model_version"))
        
        logger.info(f"Determining orchestrator for {model_name} {model_version}")
        
        # Dynamically import and run orchestrator based on model name
        orchestrator_module_path = f"app.deep_models.{model_name}_orchestrator"
        
        try:
            # Import the orchestrator module dynamically
            orchestrator_module = importlib.import_module(orchestrator_module_path)
            
            # Get the orchestrator function (assuming it's named run_{model_name}_training)
            orchestrator_function_name = f"run_{model_name.lower()}_training"
            orchestrator_function = getattr(orchestrator_module, orchestrator_function_name)
            
            train_config_id = train_config["_id"]
            logger.info(f"Running {model_name} orchestrator for train_config_id: {train_config_id}")
            
            # Call the orchestrator function with both train_config_id and train_run_id
            result = await orchestrator_function(train_config_id, train_run_id)
            logger.info(f"{model_name} orchestrator completed successfully")
            return result
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load orchestrator for {model_name}: {str(e)}")
            raise ValueError(f"Unsupported model orchestrator: {model_name}. Error: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error running orchestrator: {str(e)}")
        raise