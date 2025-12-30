# main.py
from redis.asyncio import Redis
from redis.exceptions import ResponseError
import asyncio
from typeguard import typechecked
from runtime.scheduler import start_agents
from redis_client import redis_client
from redis.exceptions import ResponseError
from config import TOWNHALL_STREAM

@typechecked
async def setup_redis(redis: Redis, stream_name: str, group_names: list[str], start_id: str = "$") -> None:
    """
    Ensure that the specified stream exists and create consumer groups for each specified role if they don't exist.

    This function is idempotent.
    """

    for group in group_names:
        try:
            await redis_client.xgroup_create(
                name = stream_name,
                groupname = group,
                id = start_id,
                mkstream = True
            )
            print(f"Created group '{group}' on stream '{stream_name}'")
        except ResponseError:
            print(f"Group '{group}' already exists on stream '{stream_name}'")

async def main():
    await setup_redis(redis_client, TOWNHALL_STREAM, ["villagers", "mayor", "judge"])
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