import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from loguru import logger
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

MONGO_URI = os.getenv("MONGO_URI")
S3_DATA_BUCKET = os.getenv("S3_BUCKET_NAME")
print(S3_DATA_BUCKET)

database_config = {
    "MONGO_URI": MONGO_URI,
    "DB_NAME": "amcnn_service_database",
    "TENANT_COLLECTION": "tenants",
    "USER_COLLECTION": "users",
    "ADMIN_COLLECTION": "admins",
    "BUCKET_CONFIG_COLLECTION": "bucket_configs",
    "DATASET_CONFIG_COLLECTION": "dataset_configs",
    "MODEL_CONFIG_COLLECTION": "model_configs",
    "PROJECT_COLLECTION": "projects",
    "TRAIN_CONFIG_COLLECTION": "train_configs",
    "TRAIN_RUN_COLLECTION": "train_runs"
}

# try:
#     key = os.getenv("CIPHER_KEY")
#     if key is None:
#         raise ValueError("CIPHER_KEY is not set")
#     logger.info(f"Cipher key: {key}")
#     cipher = Fernet(key)
# except Exception as e:
#     logger.error(f"Error initializing cipher: {e}")
#     raise e

JWT_CONFIG = {
    "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY"),
    "JWT_REFRESH_SECRET_KEY": os.getenv("JWT_REFRESH_SECRET_KEY"),
    "JWT_ALGORITHM": os.getenv("JWT_ALGORITHM"),
    "ACCESS_TOKEN_EXPIRE_MINUTES": 30,
    "REFRESH_TOKEN_EXPIRE_DAYS": 7
}