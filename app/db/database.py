import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# Configure professional logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    A singleton class to hold the MongoDB connection globally.
    This prevents creating a new connection pool on every API request.
    """
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

# Initialize the empty manager
db_manager = DatabaseManager()

def get_client() -> AsyncIOMotorClient:
    """
    Returns the MongoDB client. Initializes it safely if it doesn't exist.
    LangGraph's MongoDBSaver checkpointer specifically requires the raw client!
    """
    if db_manager.client is None:
        mongo_uri = os.getenv("MONGODB_URI")
        
        if not mongo_uri:
            logger.error("❌ Fatal Error: MONGODB_URI environment variable is not set.")
            raise ValueError("MONGODB_URI environment variable is missing!")
        
        try:
            # Initialize the async client
            db_manager.client = AsyncIOMotorClient(mongo_uri)
            
            # Set the exact database name requested
            db_manager.db = db_manager.client["portfolio_chatbot_db"]
            
            logger.info("✅ Successfully established global MongoDB connection pool.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise e
            
    return db_manager.client

def get_db() -> AsyncIOMotorDatabase:
    """
    FastAPI Dependency to inject the active database instance into your routes.
    """
    if db_manager.db is None:
        get_client()  # This ensures the client and db are initialized
    return db_manager.db