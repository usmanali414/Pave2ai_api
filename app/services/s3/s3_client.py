import os
import boto3
from botocore.exceptions import ClientError
from app.utils.logger_utils import logger
from dotenv import load_dotenv

load_dotenv()

class S3Client:
    """Singleton S3 client for AWS operations"""
    
    _instance = None
    _client = None
    _resource = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(S3Client, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize boto3 S3 client and resource"""
        try:
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            
            if not aws_access_key_id or not aws_secret_access_key:
                logger.warning("AWS credentials not found in environment variables")
                return
            
            # Initialize S3 client
            self._client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region
            )
            
            # Initialize S3 resource
            self._resource = boto3.resource(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region
            )
            
            logger.info(f"S3 client initialized successfully for region: {aws_region}")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    @property
    def client(self):
        """Get boto3 S3 client"""
        if self._client is None:
            raise RuntimeError("S3 client not initialized. Check AWS credentials.")
        return self._client
    
    @property
    def resource(self):
        """Get boto3 S3 resource"""
        if self._resource is None:
            raise RuntimeError("S3 resource not initialized. Check AWS credentials.")
        return self._resource
    
    def is_initialized(self) -> bool:
        """Check if S3 client is properly initialized"""
        return self._client is not None and self._resource is not None


# Singleton instance
s3_client = S3Client()

