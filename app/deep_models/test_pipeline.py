"""
Test script for the AMCNN training pipeline.
This script can be used to test the pipeline without going through the full API.
"""
import asyncio
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.deep_models.AMCNN_orchestrator import run_amcnn_training
from app.deep_models.data_parser.RIEGL_PARSER.data_parser import RieglParser
from app.deep_models.Algorithms.AMCNN.v1.AMCNN import AMCNN


async def test_riegl_parser():
    """Test RIEGL parser functionality."""
    print("Testing RIEGL Parser...")
    
    parser = RieglParser()
    
    # Test with dummy data paths
    data_paths = {
        "point_clouds": ["s3://test-bucket/data/cloud1.las", "s3://test-bucket/data/cloud2.las"],
        "labels": "s3://test-bucket/data/labels.txt"
    }
    
    try:
        # This will fail with S3 errors, but we can test the structure
        raw_data = parser.load_data(data_paths)
        print("✓ RIEGL Parser load_data method works")
        
        # Test validation
        if parser.validate_data(raw_data):
            print("✓ RIEGL Parser validation works")
        else:
            print("✗ RIEGL Parser validation failed")
            
    except Exception as e:
        print(f"✗ RIEGL Parser test failed (expected due to S3): {str(e)[:100]}")


async def test_amcnn_model():
    """Test AMCNN model functionality."""
    print("\nTesting AMCNN Model...")
    
    model = AMCNN()
    
    # Test model info
    info = model.get_model_info()
    print(f"✓ AMCNN Model info: {info['model_name']} v{info['version']}")
    
    # Test with dummy data
    import numpy as np
    X = np.random.rand(10, 1000, 3).astype(np.float32)  # 10 samples, 1000 points, 3 features
    y = np.random.randint(0, 10, 10).astype(np.int32)   # 10 labels
    
    try:
        # Test training
        results = model.train(X, y)
        print("✓ AMCNN Model training works")
        
        # Test prediction
        predictions = model.predict(X[:2])
        print(f"✓ AMCNN Model prediction works: {predictions}")
        
    except Exception as e:
        print(f"✗ AMCNN Model test failed: {str(e)}")


async def test_orchestrator():
    """Test orchestrator (this will fail without proper train config in DB)."""
    print("\nTesting AMCNN Orchestrator...")
    
    # This will fail without a real train_config_id in the database
    try:
        results = await run_amcnn_training("507f1f77bcf86cd799439011")  # Dummy ObjectId
        print("✓ AMCNN Orchestrator works")
        print(f"Results: {results}")
    except Exception as e:
        print(f"✗ AMCNN Orchestrator test failed (expected without DB): {str(e)[:100]}")


def test_imports():
    """Test that all imports work correctly."""
    print("Testing Imports...")
    
    try:
        from app.deep_models.base_interfaces import DataParser, Model
        print("✓ Base interfaces imported")
        
        from app.deep_models.data_parser.RIEGL_PARSER.data_parser import RieglParser
        print("✓ RIEGL Parser imported")
        
        from app.deep_models.Algorithms.AMCNN.v1.AMCNN import AMCNN
        print("✓ AMCNN Model imported")
        
        from app.deep_models.AMCNN_orchestrator import AMCNNOrchestrator
        print("✓ AMCNN Orchestrator imported")
        
    except Exception as e:
        print(f"✗ Import test failed: {str(e)}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("AMCNN Training Pipeline Test Suite")
    print("=" * 60)
    
    test_imports()
    await test_riegl_parser()
    await test_amcnn_model()
    await test_orchestrator()
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)
    print("\nNote: Some tests may fail due to missing dependencies or database setup.")
    print("This is expected in a test environment.")


if __name__ == "__main__":
    asyncio.run(main())
