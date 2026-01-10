import redis.asyncio as aioredis
import os
from dotenv import load_dotenv

load_dotenv()


def _create_redis_client() -> aioredis.Redis:
    """
    Create and return a Redis client.
    Validates that REDIS_URL is set and attempts connection.
    """
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError(
            "REDIS_URL environment variable is not set. "
            "Please set it in your .env file (e.g., REDIS_URL=redis://localhost:6379)"
        )
    
    try:
        client = aioredis.from_url(redis_url, decode_responses=True)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create Redis client: {e}") from e


redis_client = _create_redis_client()