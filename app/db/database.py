import os
import logging
import certifi
from typing import Optional
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    client: Optional[MongoClient] = None

db_manager = DatabaseManager()

def get_client() -> MongoClient:
    """Returns the standard synchronous MongoDB client required by the v1.0 checkpointer."""
    if db_manager.client is None:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            logger.error("❌ Fatal Error: MONGODB_URI environment variable missing.")
            raise ValueError("MONGODB_URI environment variable is missing!")
        
        try:
            # The crucial fix: Added tlsCAFile=certifi.where() to bypass the SSL Handshake error
            db_manager.client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
            logger.info("✅ Global MongoDB connection pool established.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise e
            
    return db_manager.client