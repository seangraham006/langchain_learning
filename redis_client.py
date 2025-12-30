import redis.asyncio as aioredis
import os
from dotenv import load_dotenv

load_dotenv()

redis_client = aioredis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=True
)