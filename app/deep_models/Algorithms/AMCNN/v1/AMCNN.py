"""
AMCNN v1 model implementation with Model interface.
"""
import warnings
warnings.filterwarnings('ignore')
import os
import time
import tempfile
import numpy as np
from typing import Dict, Any, Tuple
from app.services.s3.s3_operations import S3Operations
from app.utils.logger_utils import logger

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, callbacks, optimizers  # type: ignore
    from tensorflow.keras.optimizers import Adam, Nadam, SGD  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger  # type: ignore
    from tqdm import tqdm as tq
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Model will use fallback implementation.")

# Import training components
from app.deep_models.Algorithms.AMCNN.v1.model import modelMCNN
from app.deep_models.Algorithms.AMCNN.v1.data_generators import DataGen
from app.deep_models.Algorithms.AMCNN.v1.config import AMCNN_V1_CONFIG

class AMCNN():
    """AMCNN v1 model implementation for point cloud classification."""
    
    def __init__(self, model_config: Dict[str, Any] = None):
        self.config = model_config or AMCNN_V1_CONFIG
        self.s3_operations = S3Operations()
        self.model = None
        self.training_history = None
        self.is_compiled = False
        
        # Initialize training configuration
        self.training_configs = AMCNN_V1_CONFIG()
        self.model_mcnn = None
    
    def train(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train the AMCNN model with given data.
        
        Args:
            config: Additional training configuration
            
        Returns:
            Training metrics and results
        """
        logger.info("Starting AMCNN training")
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for training")
            return {
                "status": "failed",
                "message": "TensorFlow not available for training",
                "metrics": {}
            }
        
        try:
            # Extract dataset path from config
            dataset_path = config.get('dataset_path') if config else None
            if not dataset_path:
                logger.error("Dataset path not provided in config")
                return {
                    "status": "failed",
                    "message": "Dataset path not provided",
                    "metrics": {}
                }
            
            # Start training
            start_time = time.time()
            
            # Setup paths
            split_data_path = os.path.join(dataset_path, self.training_configs.train_data_path)
            
            # Read data using data generators
            train_gen, val_gen, test_gen = self._reading_data(
                split_data_path, 
                self.training_configs.class_folder_dict,
                self.training_configs.modelConfg['batch_size']
            )
            
            # Setup model
            learningRate = self.training_configs.modelConfg['learning_rate']
            optim = Adam(learningRate)
            self.model_mcnn = modelMCNN(
                self.training_configs.labels_dict, 
                self.training_configs.modelConfg, 
                self.training_configs.base_patch_size
            )
            self.model = self.model_mcnn.classifier(optim, is_training=True)
            
            # Handle initial weights loading/creation
            load_initial_weights = config.get('load_initial_weights', False)
            initial_weights_path = config.get('initial_weights_path', '')
            
            if load_initial_weights:
                logger.info(f"Loading initial weights from S3: {initial_weights_path}")
                if not self._load_weights_from_s3(initial_weights_path):
                    logger.info("Creating new initial weights")
                    self._create_and_upload_initial_weights(initial_weights_path)
            else:
                logger.info("Creating new initial weights")
                self._create_and_upload_initial_weights(initial_weights_path)
            
            # Setup model saving - use S3 logs path if available
            logs_output_path = config.get('logs_output_path', '')
            if logs_output_path:
                # Create local temp directory for logs
                modelpath = tempfile.mkdtemp(prefix='amcnn_logs_')
                logger.info(f'Created temp model folder at {modelpath}')
                    
                model_checkpoint_name = os.path.join(modelpath, self.training_configs.modelname + '.h5')
                csv_checkpoint_name = os.path.join(modelpath, self.training_configs.modelname + '.csv')
                
                # Store S3 path for later upload
                self.s3_logs_path = logs_output_path
                self.modelpath = modelpath
            else:
                # Fallback to local path
                modelpath = os.path.join(os.getcwd(), self.training_configs.model_path)
                if not os.path.exists(modelpath):
                    logger.info(f'Created model folder at {self.training_configs.model_path}')
                    os.makedirs(modelpath)
                    
                model_checkpoint_name = os.path.join(modelpath, self.training_configs.modelname + '.h5')
                csv_checkpoint_name = os.path.join(modelpath, self.training_configs.modelname + '.csv')
                self.s3_logs_path = None
            
            # Setup callbacks
            callbacks = self._get_callbacks(csv_checkpoint_name, model_checkpoint_name)
            
            # Train the model
            logger.info("Starting model training...")
            self.training_history = self.model.fit_generator(
                train_gen,
                validation_data=val_gen,
                steps_per_epoch=len(train_gen),
                validation_steps=len(val_gen),
                epochs=self.training_configs.modelConfg['N_epoch'],
                callbacks=callbacks,
                class_weight=self.model_mcnn.class_weights_dict
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time} seconds")
            
            # Upload training logs to S3 if path is provided
            if hasattr(self, 's3_logs_path') and self.s3_logs_path:
                self._upload_training_logs_to_s3()
            
            return {
                "status": "completed",
                "message": "AMCNN training completed successfully",
                "model_logs_path": getattr(self, 's3_logs_path', None),
                "metrics": {
                    "training_time": total_time,
                    "final_loss": self.training_history.history['loss'][-1],
                    "final_accuracy": self.training_history.history['accuracy'][-1],
                    "final_val_loss": self.training_history.history['val_loss'][-1],
                    "final_val_accuracy": self.training_history.history['val_accuracy'][-1]
                }
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return {
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "metrics": {}
            }
    
    def save_weights(self, weights_path: str) -> bool:
        """
        Save model weights to S3.
        
        Args:
            weights_path: S3 path to save weights
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("Model not initialized")
                return False
            
            local_temp_dir = tempfile.mkdtemp(prefix='amcnn_final_weights_')
            weights_filename = "amcnn_v1_weights.h5"
            local_weights_path = os.path.join(local_temp_dir, weights_filename)
            
            # Save and upload
            self.model.save_weights(local_weights_path)
            upload_result = self.s3_operations.upload_file(local_weights_path, weights_path)
            
            # Cleanup
            import shutil
            shutil.rmtree(local_temp_dir)
            
            if upload_result:
                logger.info(f"Final weights saved to S3: {weights_path}")
                return True
            else:
                logger.error(f"Failed to upload weights to S3")
                return False
                
        except Exception as e:
            logger.error(f"Error saving trained model weights: {str(e)}")
            return False
    
    def _reading_data(self, split_data_path, class_folder_dict, batch_size):
        """Read training data using data generators."""
        logger.info("Setting up data generators...")
        
        image_size = None   
        train_gen = DataGen(image_size, class_folder_dict, split_data_path, batch_size, subset='train', shuffle_check=True)
        test_gen = DataGen(image_size, class_folder_dict, split_data_path, batch_size, subset='test', shuffle_check=False)
        val_gen = DataGen(image_size, class_folder_dict, split_data_path, batch_size, subset='val', shuffle_check=False)

        logger.info(f'Total training samples: {len(train_gen) * batch_size}')
        logger.info(f'Total validation samples: {len(val_gen) * batch_size}')
        logger.info(f'Total testing samples: {len(test_gen) * batch_size}')

        return train_gen, val_gen, test_gen
    
    def _get_callbacks(self, csv_checkpoint_name, model_checkpoint_name):
        """Setup training callbacks."""
        csv_logger = CSVLogger(csv_checkpoint_name, append=False)
        checkpoint = ModelCheckpoint(model_checkpoint_name, verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9, verbose=1)

        return [csv_logger, checkpoint, reduce_lr]
    
    def _upload_training_logs_to_s3(self):
        try:
            if not hasattr(self, 's3_logs_path') or not self.s3_logs_path:
                return
            
            # Upload only CSV (training history), skip .h5 checkpoint
            csv_file_path = os.path.join(self.modelpath, self.training_configs.modelname + '.csv')
            if os.path.exists(csv_file_path):
                csv_s3_path = f"{self.s3_logs_path}/experiment1_accumulated_checkpointV1.csv"
                self.s3_operations.upload_file(csv_file_path, csv_s3_path)
                logger.info(f"Training logs saved to S3: {csv_s3_path}")
            
            # Cleanup
            import shutil
            if hasattr(self, 'modelpath') and os.path.exists(self.modelpath):
                shutil.rmtree(self.modelpath)
                
        except Exception as e:
            logger.error(f"Error uploading training logs: {str(e)}")
    
    def _load_weights_from_s3(self, weights_path: str) -> bool:
        """Load model weights from S3."""
        try:
            logger.info(f"Loading weights from S3: {weights_path}")
            
            s3_result = self.s3_operations.list_files(weights_path)
            
            if not s3_result["success"]:
                logger.warning(f"Failed to list S3 path: {weights_path}")
                return False
            
            s3_files = s3_result["files"]
            h5_files = [f["key"] for f in s3_files if f["key"].lower().endswith('.h5')]
            
            # If no .h5 files, try to find any weight files
            if not h5_files:
                # Look for common weight file extensions
                weight_extensions = ['.h5', '.hdf5', '.weights', '.ckpt', '.pth', '.pt']
                weight_files = []
                for ext in weight_extensions:
                    files_with_ext = [f["key"] for f in s3_files if f["key"].lower().endswith(ext)]
                    weight_files.extend(files_with_ext)
                
                if weight_files:
                    h5_files = weight_files
                else:
                    logger.warning(f"No weight files in S3: {weights_path}")
                    return False
            
            # Use the first .h5 file found (or you can implement logic to select specific file)
            weights_file_key = h5_files[0]  # Full S3 key
            weights_file_name = os.path.basename(weights_file_key)  # Just the filename
            full_s3_path = f"{weights_path.rstrip('/')}/{weights_file_name}"
            
            # Download weights file to local temp directory
            local_temp_dir = tempfile.mkdtemp(prefix='amcnn_weights_')
            # Use the filename we already extracted
            local_weights_path = os.path.join(local_temp_dir, weights_file_name)
            
            download_result = self.s3_operations.download_file(full_s3_path, local_weights_path)
            if not download_result["success"]:
                logger.warning(f"Failed to download: {full_s3_path}")
                return False
            
            if self.model is not None:
                try:
                    self.model.load_weights(local_weights_path)
                    logger.info(f"Loaded weights from S3: {full_s3_path}")
                    
                    import shutil
                    shutil.rmtree(local_temp_dir)
                    return True
                    
                except Exception as load_error:
                    logger.warning(f"Architecture mismatch: {str(load_error)}")
                    import shutil
                    shutil.rmtree(local_temp_dir)
                    return False
            else:
                logger.error("Model not initialized")
                return False
                
        except Exception as e:
            logger.error(f"Error loading weights from S3: {str(e)}")
            return False
    
    def _create_and_upload_initial_weights(self, s3_base_path: str) -> bool:
        """Create dummy weights, upload to S3, and load from local (no re-download)."""
        try:
            if self.model is None:
                logger.error("Model not initialized")
                return False
            
            local_temp_dir = tempfile.mkdtemp(prefix='amcnn_initial_weights_')
            initial_weights_filename = "dummy_model_weights.h5"
            local_weights_path = os.path.join(local_temp_dir, initial_weights_filename)
            
            # Save, upload, and load from local
            self.model.save_weights(local_weights_path)
            full_s3_path = f"{s3_base_path.rstrip('/')}/{initial_weights_filename}"
            self.s3_operations.upload_file(local_weights_path, full_s3_path)
            self.model.load_weights(local_weights_path)
            
            # Cleanup
            import shutil
            shutil.rmtree(local_temp_dir)
            
            logger.info(f"Initial weights uploaded to S3: {full_s3_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error with initial weights: {str(e)}")
            return False

    def _write_eval_csv(self, file_path: str, evaluation_results: dict) -> None:
        with open(file_path, 'w') as f:
            f.write('Metric,Value\n')
            for metric, value in evaluation_results.items():
                f.write(f'{metric},{value}\n')

    def evaluate(self, dataset_path: str, eval_weights_s3_url: str = "", eval_logs_s3_url: str = "", checkpoint_path: str = None) -> Dict[str, Any]:
        """
        Evaluate AMCNN using:
        - eval_weights_s3_url: Full S3 URL to trained weights file (output_weights_s3_url)
        - OR local checkpoint_path override.
        Uses AMCNN_V1_CONFIG for all parameters and data layout. Outputs loss and accuracy (no F1).
        """
        logger.info("Starting evaluation")

        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for evaluation")
            return {"status": "failed", "message": "TensorFlow not available", "metrics": {}}

        try:
            start_time = time.time()
            configs = self.training_configs
            learningRate = configs.modelConfg['learning_rate']
            optim = Adam(learningRate)

            # Build model in inference mode
            self.model_mcnn = modelMCNN(configs.labels_dict, configs.modelConfg, configs.base_patch_size)
            self.model = self.model_mcnn.classifier(optim, is_training=False)

            # Resolve checkpoint: prefer S3 weights URL if provided
            temp_ckpt_dir = tempfile.mkdtemp(prefix='amcnn_eval_ckpt_')
            chosen_ckpt = None

            if eval_weights_s3_url:
                weights_filename = os.path.basename(eval_weights_s3_url)
                local_ckpt = os.path.join(temp_ckpt_dir, weights_filename)
                download = self.s3_operations.download_file(eval_weights_s3_url, local_ckpt)
                if not download.get("success"):
                    return {"status": "failed", "message": f"Failed to download weights", "metrics": {}}
                chosen_ckpt = local_ckpt
                logger.info(f"Loaded weights from S3: {eval_weights_s3_url}")

            # Local override or fallback
            if checkpoint_path and not chosen_ckpt:
                chosen_ckpt = checkpoint_path
            if not chosen_ckpt:
                modelpath = os.path.join(os.getcwd(), configs.model_path)
                default_ckpt = os.path.join(modelpath, configs.modelname + '.h5')
                logs_ckpt = os.path.join('./model_Logs', configs.modelname + '.h5')
                chosen_ckpt = logs_ckpt if os.path.exists(logs_ckpt) else default_ckpt

            if not os.path.exists(chosen_ckpt):
                raise Exception('Checkpoint does not exists!!!')

            self.model.load_weights(chosen_ckpt)

            # Build generators
            split_data_path = os.path.join(dataset_path, configs.train_data_path)
            if not os.path.exists(split_data_path):
                raise Exception('data path does not exists!!!', split_data_path)

            train_gen, val_gen, test_gen = self._reading_data(
                split_data_path,
                configs.class_folder_dict,
                configs.modelConfg['batch_size']
            )

            # Evaluate on test
            evaluation_results = self.model.evaluate(
                test_gen, batch_size=configs.modelConfg['batch_size']
            )

            loss = float(evaluation_results[0]) if isinstance(evaluation_results, (list, tuple)) else float(evaluation_results)
            accuracy = float(evaluation_results[1]) if isinstance(evaluation_results, (list, tuple)) and len(evaluation_results) > 1 else None

            evaluation_dict = {'Loss': loss}
            if accuracy is not None:
                evaluation_dict['Accuracy'] = accuracy

            self._write_eval_csv('./results.csv', evaluation_dict)
            
            # Upload evaluation CSV to S3
            if eval_logs_s3_url:
                eval_csv_s3_path = f"{eval_logs_s3_url}/evaluation_results.csv"
                self.s3_operations.upload_file('./results.csv', eval_csv_s3_path)
            
            total_time = time.time() - start_time
            acc_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
            logger.info(f"Evaluation completed - Loss: {loss:.4f}, Accuracy: {acc_str}")
            return {
                "status": "completed",
                "message": "Evaluation completed",
                "metrics": evaluation_dict,
                "execution_time_sec": total_time,
                "evaluation_logs_s3_url": eval_logs_s3_url
            }

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {"status": "failed", "message": f"Evaluation failed: {str(e)}", "metrics": {}}