"""
AMCNN Orchestrator - Main driver script for AMCNN training pipeline.
"""
import sys
import os
import importlib
from typing import Dict, Any, Optional
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.deep_models.base_interfaces import DataParser, Model
from app.deep_models.data_parser.RIEGL_PARSER.data_parser import RieglParser
from app.deep_models.Algorithms.AMCNN.v1.AMCNN import AMCNN
from app.services.s3.s3_operations import S3Operations
from app.database.conn import mongo_client
from config import database_config
from app.utils.logger_utils import logger


class AMCNNOrchestrator:
    """Orchestrator for AMCNN training pipeline."""
    
    def __init__(self):
        self.s3_operations = S3Operations()
        self.train_config = None
        self.bucket_config = None
        self.data_parser = None
        self.model = None
        self.train_run_id = None
    
    async def execute_training(self, train_config_id: str) -> Dict[str, Any]:
        """
        Execute the complete AMCNN training pipeline.
        
        Args:
            train_config_id: ID of the train configuration
            
        Returns:
            Dictionary containing training results
        """
        try:
            logger.info(f"Starting AMCNN training pipeline for config: {train_config_id}")
            
            # Step 1: Load and validate train configuration
            await self._update_train_run_status("loading_data", "started")
            self.train_config = await self._load_train_config(train_config_id)
            
            # Step 2: Load bucket configuration
            self.bucket_config = await self._load_bucket_config(self.train_config["project_id"])
            
            # Step 3: Initialize data parser
            await self._update_train_run_status("loading_data", "in_progress")
            self.data_parser = await self._initialize_data_parser()
            
            # Step 4: Load and preprocess data
            data_paths = self._get_data_paths()
            raw_data = self.data_parser.load_data(data_paths)
            
            if not self.data_parser.validate_data(raw_data):
                raise ValueError("Data validation failed")
            
            X, y = self.data_parser.preprocess(raw_data)
            await self._update_train_run_status("loading_data", "completed")
            
            # Step 5: Initialize model
            await self._update_train_run_status("training", "started")
            self.model = await self._initialize_model()
            
            # Step 6: Train model
            await self._update_train_run_status("training", "in_progress")
            training_results = self.model.train(X, y, self.train_config.get("metadata", {}))
            
            # Step 7: Save model weights
            await self._update_train_run_status("saving_model", "started")
            weights_path = self._get_weights_output_path()
            weights_saved = self.model.save_weights(weights_path)
            
            if not weights_saved:
                raise RuntimeError("Failed to save model weights")
            
            await self._update_train_run_status("saving_model", "completed")
            await self._update_train_run_status("training", "completed")
            
            # Step 8: Update final status
            await self._update_train_run_final_status("completed", training_results)
            
            logger.info("AMCNN training pipeline completed successfully")
            
            return {
                "status": "completed",
                "train_run_id": self.train_run_id,
                "training_results": training_results,
                "weights_path": weights_path,
                "model_info": self.model.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Error in AMCNN training pipeline: {str(e)}")
            await self._update_train_run_final_status("failed", {"error": str(e)})
            raise
    
    async def _load_train_config(self, train_config_id: str) -> Dict[str, Any]:
        """Load train configuration from database."""
        try:
            db = mongo_client.database
            train_configs = db[database_config["TRAIN_CONFIG_COLLECTION"]]
            
            from bson import ObjectId
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
    
    async def _initialize_data_parser(self) -> DataParser:
        """Initialize the data parser based on train config."""
        try:
            parser_name = self.train_config["metadata"].get("data_parser", "RIEGL_PARSER")
            
            if parser_name == "RIEGL_PARSER":
                parser = RieglParser()
                logger.info("Initialized RIEGL_PARSER")
                return parser
            else:
                raise ValueError(f"Unsupported data parser: {parser_name}")
                
        except Exception as e:
            logger.error(f"Error initializing data parser: {str(e)}")
            raise
    
    async def _initialize_model(self) -> Model:
        """Initialize the model based on train config."""
        try:
            model_name = self.train_config["metadata"].get("model_name", "AMCNN")
            model_version = self.train_config.get("model_version", "v1")
            
            if model_name == "AMCNN" and model_version == "v1":
                model = AMCNN()
                logger.info("Initialized AMCNN v1 model")
                return model
            else:
                raise ValueError(f"Unsupported model: {model_name} v{model_version}")
                
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _get_data_paths(self) -> Dict[str, str]:
        """Get data paths from bucket configuration."""
        try:
            folder_structure = self.bucket_config.get("folder_structure", {})
            
            # Get base paths
            data_base = folder_structure.get("data", "")
            if not data_base:
                raise ValueError("Data path not configured in bucket config")
            
            # Construct data paths
            data_paths = {
                "point_clouds": f"{data_base}/point_clouds/",
                "labels": f"{data_base}/labels/labels.txt"
            }
            
            logger.info(f"Data paths configured: {data_paths}")
            return data_paths
            
        except Exception as e:
            logger.error(f"Error getting data paths: {str(e)}")
            raise
    
    def _get_weights_output_path(self) -> str:
        """Get the output path for model weights."""
        try:
            folder_structure = self.bucket_config.get("folder_structure", {})
            
            # Get model output path
            output_base = folder_structure.get("train_output_model", "")
            if not output_base:
                raise ValueError("Model output path not configured in bucket config")
            
            # Generate weights filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_filename = f"amcnn_v1_weights_{timestamp}.h5"
            weights_path = f"{output_base.rstrip('/')}/{weights_filename}"
            
            logger.info(f"Model weights output path: {weights_path}")
            return weights_path
            
        except Exception as e:
            logger.error(f"Error getting weights output path: {str(e)}")
            raise
    
    async def _update_train_run_status(self, step: str, status: str):
        """Update train run status for a specific step."""
        try:
            if not self.train_run_id:
                logger.warning("No train_run_id available for status update")
                return
            
            db = mongo_client.database
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            
            # Initialize step_status if it doesn't exist
            update_data = {
                f"step_status.{step}": status,
                "updated_at": datetime.utcnow()
            }
            
            await train_runs.update_one(
                {"_id": self.train_run_id},
                {"$set": update_data}
            )
            
            logger.info(f"Updated train run status: {step} -> {status}")
            
        except Exception as e:
            logger.error(f"Error updating train run status: {str(e)}")
    
    async def _update_train_run_final_status(self, final_status: str, results: Dict[str, Any]):
        """Update final train run status."""
        try:
            if not self.train_run_id:
                logger.warning("No train_run_id available for final status update")
                return
            
            db = mongo_client.database
            train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]
            
            update_data = {
                "status": final_status,
                "updated_at": datetime.utcnow(),
                "ended_at": datetime.utcnow(),
                "results": results
            }
            
            if final_status == "failed":
                update_data["error"] = results.get("error", "Unknown error")
            
            await train_runs.update_one(
                {"_id": self.train_run_id},
                {"$set": update_data}
            )
            
            logger.info(f"Updated final train run status: {final_status}")
            
        except Exception as e:
            logger.error(f"Error updating final train run status: {str(e)}")


# Main execution function
async def run_amcnn_training(train_config_id: str) -> Dict[str, Any]:
    """
    Main function to run AMCNN training.
    
    Args:
        train_config_id: ID of the train configuration
        
    Returns:
        Training results
    """
    orchestrator = AMCNNOrchestrator()
    return await orchestrator.execute_training(train_config_id)


if __name__ == "__main__":
    # This allows the orchestrator to be run directly for testing
    import asyncio
    
    if len(sys.argv) != 2:
        print("Usage: python AMCNN_orchestrator.py <train_config_id>")
        sys.exit(1)
    
    train_config_id = sys.argv[1]
    
    async def main():
        try:
            results = await run_amcnn_training(train_config_id)
            print("Training completed successfully!")
            print(f"Results: {results}")
        except Exception as e:
            print(f"Training failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(main())
