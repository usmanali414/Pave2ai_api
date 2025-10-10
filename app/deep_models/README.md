# AMCNN Deep Models Package

This package contains the complete training pipeline for the AMCNN (Adaptive Multi-scale Convolutional Neural Network) system, including data parsers, model implementations, and orchestration logic.

## Architecture

```
app/deep_models/
├── base_interfaces.py          # Abstract base classes for parsers and models
├── AMCNN_orchestrator.py       # Main orchestrator for AMCNN training
├── test_pipeline.py           # Test script for the pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data_parser/               # Data parsing implementations
│   ├── RIEGL_PARSER/         # RIEGL point cloud parser
│   │   ├── data_parser.py    # Main parser implementation
│   │   ├── data_loader.py    # S3 data loading utilities
│   │   ├── preprocessor.py   # Data preprocessing
│   │   └── config.py         # Parser configuration
│   └── other_parser/         # Placeholder for other parsers
└── Algorithms/                # Model implementations
    └── AMCNN/                 # AMCNN model family
        ├── v1/               # AMCNN version 1
        │   ├── AMCNN.py      # Model implementation
        │   └── config.py     # Model configuration
        └── v2/               # Future versions
```

## Key Components

### 1. Base Interfaces (`base_interfaces.py`)

Defines abstract base classes that all parsers and models must implement:

- **DataParser**: Interface for data loading and preprocessing
  - `load_data()`: Load data from S3 paths
  - `preprocess()`: Preprocess data for training
  - `validate_data()`: Validate loaded data

- **Model**: Interface for ML models
  - `train()`: Train the model
  - `predict()`: Make predictions
  - `save_weights()`: Save model weights to S3
  - `load_weights()`: Load model weights from S3

### 2. Data Parsers

#### RIEGL_PARSER
- **Purpose**: Parse RIEGL point cloud data (.las, .laz, .ply, .pcd formats)
- **Features**:
  - S3 data loading
  - Point cloud normalization
  - Outlier removal
  - Data augmentation
  - Train/test splitting

### 3. Models

#### AMCNN v1
- **Purpose**: Point cloud classification using adaptive multi-scale convolutions
- **Features**:
  - TensorFlow/Keras implementation
  - Configurable architecture
  - Early stopping and learning rate scheduling
  - S3 weight storage
  - Fallback implementation when TensorFlow unavailable

### 4. Orchestrator

#### AMCNN_orchestrator.py
- **Purpose**: Coordinates the complete training pipeline
- **Pipeline Steps**:
  1. Load and validate train configuration
  2. Load bucket configuration
  3. Initialize data parser
  4. Load and preprocess data
  5. Initialize model
  6. Train model
  7. Save model weights
  8. Update database status

## Usage

### 1. Via API (Recommended)

The orchestrator is automatically called by the training service:

```python
from app.services.train.train import start_training

# Start training with a train_config_id
results = await start_training("507f1f77bcf86cd799439011")
```

### 2. Direct Orchestrator Usage

```python
from app.deep_models.AMCNN_orchestrator import run_amcnn_training

# Run training directly
results = await run_amcnn_training("507f1f77bcf86cd799439011")
```

### 3. Individual Components

```python
from app.deep_models.data_parser.RIEGL_PARSER.data_parser import RieglParser
from app.deep_models.Algorithms.AMCNN.v1.AMCNN import AMCNN

# Use parser directly
parser = RieglParser()
data_paths = {
    "point_clouds": ["s3://bucket/data/cloud1.las"],
    "labels": "s3://bucket/data/labels.txt"
}
raw_data = parser.load_data(data_paths)
X, y = parser.preprocess(raw_data)

# Use model directly
model = AMCNN()
results = model.train(X, y)
```

## Configuration

### Train Configuration

The system expects a train configuration with the following structure:

```python
{
    "name": "My Training Job",
    "tenant_id": "tenant_123",
    "project_id": "project_456",
    "model_version": "v1",
    "metadata": {
        "data_parser": "RIEGL_PARSER",
        "model_name": "AMCNN",
        "initial_weights": false
    }
}
```

### Bucket Configuration

The system requires bucket configuration with S3 paths:

```python
{
    "project_id": "project_456",
    "folder_structure": {
        "data": "s3://bucket/tenant_123/project_456/data/",
        "train_output_model": "s3://bucket/tenant_123/project_456/models/"
    }
}
```

## Database Schema

### Train Runs Collection

The system tracks training progress in the `train_runs` collection:

```python
{
    "_id": "run_id",
    "train_config_id": "config_id",
    "status": "training|completed|failed|cancelled",
    "step_status": {
        "loading_data": "started|in_progress|completed",
        "training": "started|in_progress|completed",
        "saving_model": "started|in_progress|completed"
    },
    "results": {...},  # Training results and metrics
    "error": "error_message",  # Only present if failed
    "created_at": "datetime",
    "updated_at": "datetime",
    "ended_at": "datetime"
}
```

## Dependencies

Install the required dependencies:

```bash
pip install -r app/deep_models/requirements.txt
```

Key dependencies:
- `tensorflow==2.15.0`
- `numpy==1.26.4`
- `scikit-learn==1.3.2`

## Testing

Run the test suite:

```bash
python app/deep_models/test_pipeline.py
```

This will test:
- Import functionality
- Data parser components
- Model components
- Orchestrator (with expected failures due to missing DB)

## Error Handling

The system includes comprehensive error handling:

1. **Validation**: Train config validation before starting
2. **Graceful Failures**: Proper error messages and status updates
3. **Fallbacks**: Fallback implementations when dependencies unavailable
4. **Logging**: Detailed logging throughout the pipeline

## Extending the System

### Adding New Data Parsers

1. Create a new parser directory under `data_parser/`
2. Implement the `DataParser` interface
3. Add configuration file
4. Update imports in `__init__.py` files

### Adding New Models

1. Create a new model directory under `Algorithms/`
2. Implement the `Model` interface
3. Add configuration file
4. Update imports in `__init__.py` files
5. Add orchestrator logic if needed

### Adding New Orchestrators

1. Create a new `{MODEL_NAME}_orchestrator.py` file
2. Implement the training pipeline
3. Update the train service to route to the new orchestrator

## Monitoring

The system provides real-time status updates:

- **Step Status**: Track progress through each pipeline step
- **Database Updates**: Real-time updates to train_runs collection
- **Logging**: Comprehensive logging for debugging
- **Metrics**: Training metrics and model performance

## Security Considerations

- All S3 operations use configured credentials
- Database operations are validated and sanitized
- Error messages don't expose sensitive information
- File operations include proper cleanup
