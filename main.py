# main.py
import asyncio
from runtime.scheduler import start_agents
from redis_client import redis_client
from redis.exceptions import ResponseError
from config import STREAM_TOPICS

async def main():
    # Create consumer groups for all agents
    for group_name in ["villagers", "mayor", "judge"]:
        try:
            await redis_client.xgroup_create(STREAM_TOPICS, group_name, id="$", mkstream=True)
        except ResponseError:
            pass  # Group already exists
    
    await start_agents()

async def cleanup():
    try:
        await redis_client.aclose()
    except RuntimeError:
        pass

try:
    asyncio.run(main())
finally:
    try:
        asyncio.run(cleanup())
    except RuntimeError:
        pass  # Event loop already closed
