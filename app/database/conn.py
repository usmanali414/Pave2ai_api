from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from config import database_config
from app.utils.logger_utils import logger

class MongoDBClient:
    """Singleton MongoDB connection handler with connection pool support."""

    _instance: Optional["MongoDBClient"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        uri: str,
        db_name: str,
        max_pool_size: int = 5000,
        min_pool_size: int = 10,
        server_selection_timeout_ms: int = 5000,
    ):
        if self._initialized:
            return  # Prevent re-initialization

        self._uri = uri
        self._db_name = db_name
        self._max_pool_size = max_pool_size
        self._min_pool_size = min_pool_size
        self._server_selection_timeout_ms = server_selection_timeout_ms
        self._client: Optional[AsyncIOMotorClient] = None
        self._db = None
        self._initialized = True

    async def connect(self):
        """Establish a MongoDB connection with pooling."""
        if self._client is None:
            self._client = AsyncIOMotorClient(
                self._uri,
                maxPoolSize=self._max_pool_size,
                minPoolSize=self._min_pool_size,
                serverSelectionTimeoutMS=self._server_selection_timeout_ms,
            )
            self._db = self._client[self._db_name]
            logger.info("✅ Connected to MongoDB")

    async def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("❌ Disconnected from MongoDB")

    @property
    def database(self):
        """Return active database instance."""
        if self._db is None:
            raise RuntimeError("Database connection is not initialized. Call `connect()` first.")
        return self._db

    @property
    def client(self):
        """Return raw MongoDB client instance."""
        if self._client is None:
            raise RuntimeError("MongoDB client is not initialized. Call `connect()` first.")
        return self._client


# Global singleton instance
mongo_client = MongoDBClient(
    uri=database_config["MONGO_URI"],
    db_name=database_config["DB_NAME"]
)
