"""
AMCNN Orchestrator - Main driver script for AMCNN training pipeline.
"""
import sys
import os
import importlib
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from bson import ObjectId
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.deep_models.base_interfaces import DataParser
from app.deep_models.Algorithms.AMCNN.v1.AMCNN import AMCNN
from app.services.s3.s3_operations import S3Operations
from app.database.conn import mongo_client
from config import database_config
from app.utils.logger_utils import logger


def normalize_component_name(name: str) -> str:
    if not name:
        return ""
    return name.replace(" ", "_")


class AMCNNOrchestrator:
    """Orchestrator for AMCNN training pipeline."""
    
    def __init__(self):
        self.s3_operations = S3Operations()
        self.train_config = None
        self.bucket_config = None
        self.data_parser = None
        self.model = None
        self.train_run_id = None

    async def execute_training_with_resume(self, train_config_id: str, train_run_id: str = None, resume_from: str = None) -> Dict[str, Any]:
        """
        Execute training with resume capability.
        
        Args:
            train_config_id: Training configuration ID
            train_run_id: Training run ID
            resume_from: Step to resume from
            
        Returns:
            Training results
        """
        try:
            logger.info(f"Starting AMCNN training pipeline with resume - config: {train_config_id}, run: {train_run_id}, resume_from: {resume_from}")
            
            self.train_run_id = train_run_id
            self.train_config = await self._load_train_config(train_config_id)
            self.bucket_config = await self._load_bucket_config(self.train_config["project_id"])
            
            # Load existing run to get current step status
            db = mongo_client.database
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            existing_run = await train_runs.find_one({"_id": ObjectId(train_run_id)})
            if not existing_run:
                raise ValueError(f"Training run not found: {train_run_id}")
            step_status = existing_run.get("step_status", {})
            
            # Execute steps based on resume point - skip completed steps
            loading_data_status = self._get_step_status(step_status, "loading_data")
            preprocessing_status = self._get_step_status(step_status, "preprocessing")
            training_status = self._get_step_status(step_status, "training")
            saving_model_status = self._get_step_status(step_status, "saving_model")
            
            logger.info(f"Step statuses - loading_data: {loading_data_status}, preprocessing: {preprocessing_status}, training: {training_status}, saving_model: {saving_model_status}")
            
            # Only execute loading_data if it's not completed
            if loading_data_status != "completed":
                logger.info("Executing loading_data step")
                await self._execute_loading_data_step()
            else:
                logger.info("Skipping loading_data step - already completed")
            
            # Only execute preprocessing if it's not completed
            if preprocessing_status != "completed":
                logger.info("Executing preprocessing step")
                await self._execute_preprocessing_step()
            else:
                logger.info("Skipping preprocessing step - already completed")
            
            # Only execute training if it's not completed
            if training_status != "completed":
                logger.info("Executing training step")
                await self._execute_training_step()
            else:
                logger.info("Skipping training step - already completed")
            
            # Only execute saving_model if it's not completed
            if saving_model_status != "completed":
                logger.info("Executing saving_model step")
                await self._execute_saving_model_step()
            else:
                logger.info("Skipping saving_model step - already completed")
            
            # Update final status
            await self._update_train_run_final_status(train_run_id, "completed")
            logger.info(f"Training pipeline completed with resume - run: {train_run_id}")
            
            return {
                "status": "completed",
                "train_run_id": self.train_run_id
            }
            
        except Exception as e:
            logger.error(f"Error in AMCNN training pipeline with resume: {str(e)}")
            await self._update_train_run_final_status(train_run_id, "failed")
            raise

    def _get_step_status(self, step_status: Dict[str, Any], step: str) -> str:
        """Get step status - simple string format."""
        return step_status.get(step, "pending")

    async def _execute_loading_data_step(self):
        """Execute loading_data step with proper status tracking."""
        try:
            await self._update_train_run_status(self.train_run_id, "loading_data", "in_progress")
            
            components = await self._load_training_components(self.train_config)
            self.data_parser = components["data_parser"]
            self.model = components["model"]

            transferred_data = await self.data_parser.load_data(self.train_config)
            if not self.data_parser.validate_transferred_data(transferred_data):
                raise ValueError("Data validation failed")
            
            await self._update_train_run_status(self.train_run_id, "loading_data", "completed")
            
        except Exception as e:
            await self._update_train_run_status(self.train_run_id, "loading_data", "failed")
            raise


    async def _execute_preprocessing_step(self):
        """Execute preprocessing step with proper status tracking."""
        try:
            await self._update_train_run_status(self.train_run_id, "preprocessing", "in_progress")
            
            # Only load data if not already loaded (loading_data step was skipped)
            if not hasattr(self, 'data_parser') or not self.data_parser:
                logger.info("Loading data for preprocessing step")
                components = await self._load_training_components(self.train_config)
                self.data_parser = components["data_parser"]
                transferred_data = await self.data_parser.load_data(self.train_config)
            else:
                logger.info("Using already loaded data for preprocessing step")
                # Data is already loaded, just get the transferred data info
                transferred_data = await self.data_parser.load_data(self.train_config)
            
            model_version = self.train_config.get("model_version", "v1")
            await self._run_amcnn_preprocessor(transferred_data, model_version)
            
            await self._update_train_run_status(self.train_run_id, "preprocessing", "completed")
            
        except Exception as e:
            await self._update_train_run_status(self.train_run_id, "preprocessing", "failed")
            raise

    async def _execute_training_step(self):
        """Execute training step with proper status tracking."""
        try:
            await self._update_train_run_status(self.train_run_id, "training", "in_progress")
            
            # Reload model if needed
            if not hasattr(self, 'model') or not self.model:
                components = await self._load_training_components(self.train_config)
                self.model = components["model"]
            
            # Prepare training config
            training_config = self.train_config.get("metadata", {})
            training_config["dataset_path"] = self._get_dataset_path()
            training_config["logs_output_path"] = self._get_logs_output_path()
            training_config["load_initial_weights"] = self.train_config.get("metadata", {}).get("initial_weights", False)
            training_config["initial_weights_path"] = self._get_initial_weights_path()
            
            training_results = self.model.train(training_config)
            if training_results["status"] != "completed":
                raise RuntimeError("Failed to train model")
            
            await self._update_train_run_status(self.train_run_id, "training", "completed")
            
        except Exception as e:
            await self._update_train_run_status(self.train_run_id, "training", "failed")
            raise

    async def _execute_saving_model_step(self):
        """Execute saving_model step with proper status tracking."""
        try:
            await self._update_train_run_status(self.train_run_id, "saving_model", "in_progress")
            
            # Reload model if needed
            if not hasattr(self, 'model') or not self.model:
                components = await self._load_training_components(self.train_config)
                self.model = components["model"]
            
            weights_path = self._get_weights_output_path()
            if not self.model.save_weights(weights_path):
                raise RuntimeError("Failed to save model weights")
            
            await self._update_train_run_status(self.train_run_id, "saving_model", "completed")
            
            # Update S3 URLs and results
            input_weights_url = self._get_initial_weights_path() + "/dummy_model_weights.h5"
            output_logs_url = self._get_logs_output_path()
            await self._update_train_run_s3_urls(self.train_run_id, input_weights_url, weights_path, output_logs_url)
            
            # Save training logs CSV URL
            training_logs_csv_url = f"{output_logs_url}/experiment1_accumulated_checkpointV1.csv"
            await self._update_train_run_results(self.train_run_id, training_logs_csv_url, "training")
            
            # Run evaluation
            eval_results = self.model.evaluate(
                dataset_path=self._get_dataset_path(),
                eval_weights_s3_url=weights_path,
                eval_logs_s3_url=output_logs_url
            )

            if eval_results.get("status") == "completed":
                evaluation_logs_csv_url = f"{output_logs_url}/evaluation_results.csv"
                await self._update_train_run_results(self.train_run_id, evaluation_logs_csv_url, "evaluation")
            
        except Exception as e:
            await self._update_train_run_status(self.train_run_id, "saving_model", "failed")
            raise

    async def execute_training(self, train_config_id: str, train_run_id: str = None) -> Dict[str, Any]:

        try:
            logger.info(f"Starting AMCNN training pipeline - config: {train_config_id}, run: {train_run_id}")
            
            self.train_run_id = train_run_id
            self.train_config = await self._load_train_config(train_config_id)
            self.bucket_config = await self._load_bucket_config(self.train_config["project_id"])
            
            await self._update_train_run_status(train_run_id, "loading_data", "in_progress")
            
            components = await self._load_training_components(self.train_config)
            self.data_parser = components["data_parser"]
            self.model = components["model"]

            transferred_data = await self.data_parser.load_data(self.train_config)
            if not self.data_parser.validate_transferred_data(transferred_data):
                raise ValueError("Data validation failed")
            await self._update_train_run_status(train_run_id, "loading_data", "completed")
            
            await self._update_train_run_status(train_run_id, "preprocessing", "in_progress")
            model_version = self.train_config.get("model_version", "v1")
            await self._run_amcnn_preprocessor(transferred_data, model_version)
            await self._update_train_run_status(train_run_id, "preprocessing", "completed")
            
            await self._update_train_run_status(train_run_id, "training", "in_progress")
            
            # Prepare training config with dataset path and initial weights info
            training_config = self.train_config.get("metadata", {})
            training_config["dataset_path"] = self._get_dataset_path()
            training_config["logs_output_path"] = self._get_logs_output_path()
            
            # Pass load_initial_weights from train_config
            training_config["load_initial_weights"] = self.train_config.get("metadata", {}).get("initial_weights", False)
            
            # Always set initial weights base path (used in both true/false cases)
            training_config["initial_weights_path"] = self._get_initial_weights_path()
            
            training_results = self.model.train(training_config)
            if training_results["status"] != "completed":
                await self._update_train_run_status(train_run_id, "training", "failed")
                raise RuntimeError("Failed to train model")
            
            await self._update_train_run_status(train_run_id, "training", "completed")
            
            await self._update_train_run_status(train_run_id, "saving_model", "in_progress")
            weights_path = self._get_weights_output_path()
            if not self.model.save_weights(weights_path):
                await self._update_train_run_status(train_run_id, "saving_model", "failed")
                raise RuntimeError("Failed to save model weights")
            await self._update_train_run_status(train_run_id, "saving_model", "completed")
            
            # Update S3 URLs
            input_weights_url = self._get_initial_weights_path() + "/dummy_model_weights.h5"
            output_logs_url = training_config.get("logs_output_path", "")
            await self._update_train_run_s3_urls(train_run_id, input_weights_url, weights_path, output_logs_url)
            
            # Save training logs CSV URL
            training_logs_csv_url = f"{output_logs_url}/experiment1_accumulated_checkpointV1.csv"
            await self._update_train_run_results(train_run_id, training_logs_csv_url, "training")
            
            # Run evaluation
            eval_results = self.model.evaluate(
                dataset_path=self._get_dataset_path(),
                eval_weights_s3_url=weights_path,
                eval_logs_s3_url=output_logs_url
            )

            if eval_results.get("status") == "completed":
                # Save evaluation CSV URL
                evaluation_logs_csv_url = f"{output_logs_url}/evaluation_results.csv"
                await self._update_train_run_results(train_run_id, evaluation_logs_csv_url, "evaluation")
            
            await self._update_train_run_final_status(train_run_id, "completed")
            logger.info(f"Training pipeline completed - run: {train_run_id}")
            
            return {
                "status": "completed",
                "train_run_id": self.train_run_id
            }
            
        except Exception as e:
            logger.error(f"Error in AMCNN training pipeline: {str(e)}")
            await self._update_train_run_final_status(train_run_id, "failed")
            raise
    
    async def _load_train_config(self, train_config_id: str) -> Dict[str, Any]:
        """Load train configuration from database."""
        try:
            db = mongo_client.database
            train_configs = db[database_config["TRAIN_CONFIG_COLLECTION"]]
            
            config = await train_configs.find_one({"_id": ObjectId(train_config_id)})
            
            if not config:
                raise ValueError(f"Train config not found: {train_config_id}")
            
            # Store train_run_id for status updates
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            run = await train_runs.find_one({
                "train_config_id": train_config_id,
                "status": "training"
            })
            
            if run:
                self.train_run_id = str(run["_id"])
            
            logger.info(f"Loaded train config: {config['name']}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading train config: {str(e)}")
            raise
    
    async def _load_bucket_config(self, project_id: str) -> Dict[str, Any]:
        """Load bucket configuration for the project."""
        try:
            db = mongo_client.database
            bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
            
            config = await bucket_configs.find_one({"project_id": project_id})
            
            if not config:
                raise ValueError(f"Bucket config not found for project: {project_id}")
            
            logger.info(f"Loaded bucket config for project: {project_id}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading bucket config: {str(e)}")
            raise
    
    
    
    def _get_data_paths(self) -> Dict[str, str]:
        """Get data paths from bucket configuration."""
        try:
            folder_structure = self.bucket_config.get("folder_structure", {})
            
            # Return the folder structure for RIEGL_PARSER to use
            # RIEGL_PARSER will use preprocessed_data and annotate_label from this structure
            logger.info(f"Data paths configured: {list(folder_structure.keys())}")
            return folder_structure
            
        except Exception as e:
            logger.error(f"Error getting data paths: {str(e)}")
            raise
    
    def _get_weights_output_path(self) -> str:
        """Get the output path for model weights."""
        try:
            folder_structure = self.bucket_config.get("folder_structure", {})
            output_base = folder_structure.get("train_output_model", "")
            if not output_base:
                raise ValueError("Model output path not configured in bucket config")
            
            weights_path = f"{output_base.rstrip('/')}/{self.train_run_id}/amcnn_v1_weights.h5"
            return weights_path
            
        except Exception as e:
            logger.error(f"Error getting weights output path: {str(e)}")
            raise
    
    def _get_logs_output_path(self) -> str:
        """Get the output path for training logs."""
        try:
            folder_structure = self.bucket_config.get("folder_structure", {})
            logs_base = folder_structure.get("train_output_metadata", "")
            if not logs_base:
                raise ValueError("Training logs output path not configured in bucket config")
            
            return f"{logs_base.rstrip('/')}/{self.train_run_id}"
            
        except Exception as e:
            logger.error(f"Error getting logs output path: {str(e)}")
            raise
    
    def _get_initial_weights_path(self) -> str:
        """Get the input path for initial weights from bucket config."""
        try:
            folder_structure = self.bucket_config.get("folder_structure", {})
            initial_weights_base = folder_structure.get("train_input_model", "")
            if not initial_weights_base:
                raise ValueError("Initial weights input path not configured in bucket config")
            
            return f"{initial_weights_base.rstrip('/')}/{self.train_run_id}"
            
        except Exception as e:
            logger.error(f"Error getting initial weights path: {str(e)}")
            raise
    
    async def _run_amcnn_preprocessor(self, transferred_data: Dict[str, Any], model_version: str):
        """
        Run AMCNN preprocessor based on model version.
        
        Args:
            transferred_data: Data from RIEGL_PARSER with local file paths
            model_version: AMCNN model version (e.g., "v1")
            
        Returns:
            Tuple of (X, y) for training
        """
        try:
            logger.info(f"Running AMCNN {model_version} preprocessor...")
            
            # Dynamically import the appropriate preprocessor based on model name and version
            model_name = normalize_component_name(self.train_config["metadata"].get("model_name"))
            preprocessor_module_path = f"app.deep_models.Algorithms.{model_name}.{model_version}.preprocessor"
            
            try:
                # Import the preprocessor module dynamically
                preprocessor_module = importlib.import_module(preprocessor_module_path)
                
                # Get parser name from train config and normalize it
                parser_name = normalize_component_name(self.train_config["metadata"].get("data_parser"))
                
                # Get preprocessor class using exact parser name
                preprocessor_class_name = f"{parser_name.replace('_PARSER', 'Preprocessor')}"
                preprocessor_class = getattr(preprocessor_module, preprocessor_class_name)
                
                # # Import config using exact parser name
                # parser_config_module_path = f"app.deep_models.data_parser.{parser_name}.config"
                # parser_config_module = importlib.import_module(parser_config_module_path)
                # parser_config = getattr(parser_config_module, f"{parser_name}_CONFIG")
                
                # Initialize preprocessor with config
                preprocessor = preprocessor_class()
                
                # Get local file paths from transferred data
                image_paths = transferred_data["image_paths"]
                annotation_paths = transferred_data["annotation_paths"]
                project_id = transferred_data["project_id"]
                
                # # Create image-annotation pairs
                # image_annotation_pairs = list(zip(image_paths, annotation_paths))
                
                # logger.info(f"Processing {len(image_annotation_pairs)} image-annotation pairs")
                
                # Run preprocessing (assuming the method is called preprocess_images_and_annotations)
                preprocessor.preprocess_images_and_annotations(
                    image_paths, annotation_paths, project_id
                )
                
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load preprocessor for version {model_version}: {str(e)}")
                raise ValueError(f"Unsupported AMCNN model version: {model_version}. Error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error running AMCNN preprocessor: {str(e)}")
            raise
    
    async def _load_training_components(self, train_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load training components dynamically using exact names from train_config.
        
        Args:
            train_config: Train configuration dictionary
            
        Returns:
            Dictionary containing loaded components (data_parser, model)
        """
        try:
            parser_name = normalize_component_name(train_config["metadata"].get("data_parser"))
            model_name = normalize_component_name(train_config["metadata"].get("model_name"))
            model_version = normalize_component_name(train_config.get("model_version"))
            
            logger.info(f"Loading {parser_name} parser and {model_name} {model_version}")
            
            # Load data parser using exact parser name
            data_parser = await self._load_component_dynamic(
                component_type="parser",
                component_name=parser_name
            )
            
            model = await self._load_component_dynamic(
                component_type="model",
                component_name=model_name,
                component_version=model_version,
                train_config=train_config
            )
            
            return {
                "data_parser": data_parser,
                "model": model
            }
            
        except Exception as e:
            logger.error(f"Error loading training components: {str(e)}")
            raise
    
    async def _load_component_dynamic(self, component_type: str, component_name: str, 
                                    s3_urls: Dict[str, str] = None, train_config: Dict[str, Any] = None,
                                    component_version: str = None):
        """
        Load any component dynamically using exact names from config.
        
        Args:
            component_type: Type of component ("parser" or "model")
            component_name: Exact name from config (e.g., "RIEGL_PARSER", "AMCNN")
            s3_urls: S3 URLs for parser (only for parser type)
            train_config: Train configuration
            component_version: Version for model (only for model type)
            
        Returns:
            Initialized component instance
        """
        try:
            if component_type == "parser":
                # Parser import path: app.deep_models.data_parser.{PARSER_NAME}.{PARSER_NAME}
                module_path = f"app.deep_models.data_parser.{component_name}.{component_name}"
                module = importlib.import_module(module_path)
                
                # Class name: Same as parser name
                component_class = getattr(module, component_name)
                
                # Initialize with parser config
                config = {}
                
            elif component_type == "model":
                # Model import path: app.deep_models.Algorithms.{MODEL_NAME}.{VERSION}.{MODEL_NAME}
                module_path = f"app.deep_models.Algorithms.{component_name}.{component_version}.{component_name}"
                module = importlib.import_module(module_path)
                
                # Class name: Same as model name
                component_class = getattr(module, component_name)
                
                # Initialize without config for model
                config = {}
                
            else:
                raise ValueError(f"Unsupported component type: {component_type}")
            
            # Instantiate component
            component = component_class(config) if config else component_class()
            return component
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load {component_type} {component_name}: {str(e)}")
            raise ValueError(f"Unsupported {component_type}: {component_name}. Error: {str(e)}")
    
    async def _update_train_run_status(self, train_run_id: str, step: str, status: str):
        """Update train run status for a specific step."""
        try:
            if not train_run_id:
                logger.warning("No train_run_id available for status update")
                return
            
            db = mongo_client.database
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            
            # Convert string train_run_id to ObjectId
            run_id = ObjectId(train_run_id)
            
            # Simple step_status structure - just the status string
            now = datetime.utcnow()
            
            # Update step_status
            update_data = {
                f"step_status.{step}": status,
                "updated_at": now
            }
            
            await train_runs.update_one(
                {"_id": run_id},
                {"$set": update_data}
            )
            
        except Exception as e:
            logger.error(f"Error updating train run status: {str(e)}")
    
    async def _update_train_run_final_status(self, train_run_id: str, final_status: str):
        """Update final train run status."""
        try:
            if not train_run_id:
                logger.warning("No train_run_id available for final status update")
                return
            
            db = mongo_client.database
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            
            # Convert string train_run_id to ObjectId
            run_id = ObjectId(train_run_id)
            
            update_data = {
                "status": final_status,
                "updated_at": datetime.utcnow(),
                "ended_at": datetime.utcnow()
            }
            
            # Clear error field if training completed successfully
            if final_status == "completed":
                update_data["error"] = ""
            
            await train_runs.update_one(
                {"_id": run_id},
                {"$set": update_data}
            )
            
        except Exception as e:
            logger.error(f"Error updating final train run status: {str(e)}")
    
    async def _update_train_run_s3_urls(self, train_run_id: str, input_weights_url: str, output_weights_url: str, output_logs_url: str):
        """Update train run with S3 URLs for all generated files."""
        try:
            if not train_run_id:
                logger.warning("No train_run_id available for S3 URLs update")
                return
            
            db = mongo_client.database
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            
            # Convert string train_run_id to ObjectId
            run_id = ObjectId(train_run_id)
            
            # Update S3 URLs
            update_data = {
                "input_weights_s3_url": input_weights_url,
                "output_weights_s3_url": output_weights_url,
                "output_logs_s3_url": output_logs_url,
                "updated_at": datetime.utcnow()
            }
            
            await train_runs.update_one(
                {"_id": run_id},
                {
                    "$set": update_data,
                    "$unset": {"model_logs_path": ""}  # Remove legacy field
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating train run S3 URLs: {str(e)}")
    
    def _get_dataset_path(self) -> str:
        """Get the dataset path where patches are stored."""
        try:
            current_file = Path(__file__)
            project_root = current_file.parents[2]
            dataset_path = project_root / "static"
            return str(dataset_path)
        except Exception as e:
            logger.error(f"Error getting dataset path: {str(e)}")
            return os.path.join(os.getcwd(), "static")

    async def _update_train_run_results(self, train_run_id: str, logs_s3_url: Any, target: str):
        """Update train run results under {target}_logs_s3_url (e.g., 'training_logs_s3_url' or 'evaluation_logs_s3_url')."""
        try:
            if not train_run_id or not logs_s3_url:
                logger.warning("No train_run_id or empty logs_s3_url, skipping update")
                return
            db = mongo_client.database
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            run_id = ObjectId(train_run_id)
            await train_runs.update_one(
                {"_id": run_id}, 
                {"$set": {f"{target}_logs_s3_url": logs_s3_url, "updated_at": datetime.utcnow()}}
            )
        except Exception as e:
            logger.error(f"Error updating train run {target} logs S3 URL: {str(e)}")


# Main execution function
async def run_amcnn_training(train_config_id: str, train_run_id: str = None) -> Dict[str, Any]:
    """
    Main function to run AMCNN training.
    
    Args:
        train_config_id: ID of the train configuration
        train_run_id: ID of the train run record
        
    Returns:
        Training results
    """
    orchestrator = AMCNNOrchestrator()
    return await orchestrator.execute_training(train_config_id, train_run_id)

async def run_amcnn_training_with_resume(train_config_id: str, train_run_id: str = None, resume_from: str = None) -> Dict[str, Any]:
    """
    Run AMCNN training with resume capability.
    
    Args:
        train_config_id: ID of the train configuration
        train_run_id: ID of the train run record
        resume_from: Step to resume from
        
    Returns:
        Training results
    """
    orchestrator = AMCNNOrchestrator()
    return await orchestrator.execute_training_with_resume(train_config_id, train_run_id, resume_from)

