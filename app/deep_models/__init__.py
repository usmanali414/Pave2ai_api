"""
Deep Models Package - Contains all ML models and data parsers for the AMCNN training pipeline.
"""

__version__ = "1.0.0"
__author__ = "AMCNN Team"

# Import base interfaces
from .base_interfaces import DataParser, Model

# Import available parsers
from .data_parser.RIEGL_PARSER.data_parser import RieglParser

# Import available models
from .Algorithms.AMCNN.v1.AMCNN import AMCNN

# Import orchestrator
from .AMCNN_orchestrator import AMCNNOrchestrator, run_amcnn_training

__all__ = [
    # Base interfaces
    "DataParser",
    "Model",
    
    # Parsers
    "RieglParser",
    
    # Models
    "AMCNN",
    
    # Orchestrators
    "AMCNNOrchestrator",
    "run_amcnn_training"
]
