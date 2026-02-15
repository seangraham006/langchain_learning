# main.py
from redis.asyncio import Redis
from redis.exceptions import ResponseError, ConnectionError
import asyncio
import subprocess
import time
from typeguard import typechecked
from redis_client import redis_client
from config import TOWNHALL_STREAM

from agents.Villager import Villager
from agents.Mayor import Mayor
from agents.Judge import Judge
from agents.Agent import Agent
from agents.ChronicleAgent import ChronicleAgent
from utils.embeddings import _get_embedding_model

# Pre-load the embedding model on the main thread before any agents start
print("Loading embedding model...")
_get_embedding_model()
print("Embedding model loaded.")

async def ensure_redis_running(redis: Redis, max_retries: int = 3) -> None:
    """
    Check if Redis is running and start it if not.
    """
    for attempt in range(max_retries):
        try:
            await redis.ping()
            print("Redis is running")
            return
        except (ConnectionError, Exception) as e:
            print(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                print("Attempting to start Redis server...")
                try:
                    subprocess.Popen(
                        ["redis-server", "--daemonize", "yes"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print("Waiting for Redis to start...")
                    time.sleep(2)
                except Exception as start_error:
                    print(f"Failed to start Redis: {start_error}")
            else:
                time.sleep(1)
    
    raise RuntimeError("Could not connect to Redis after multiple attempts. Please start Redis manually.")

@typechecked
async def setup_redis(redis: Redis, stream_name: str, group_names: list[str], start_id: str = "$") -> None:
    """
    Ensure Redis is running and that the specified stream exists.
    Create consumer groups for each specified role if they don't exist.

    This function is idempotent.
    """
    await ensure_redis_running(redis)

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

async def kickoff_meeting(redis: Redis, stream_name: str, role: str, message: str) -> None:
    """
    Add an initial message to the stream if it's empty to kick off the meeting.
    """
    stream_length = await redis.xlen(stream_name)
    
    if stream_length == 0:
        await redis.xadd(
            stream_name,
            {"role": role, "text": message}
        )
        print(f"Added kickoff message to '{stream_name}'")
    
    else:
        print(f"Stream '{stream_name}' already has {stream_length} messages, skipping kickoff")

async def clear_conversation(redis: Redis, stream_name: str) -> None:
    """
    Delete the entire conversation stream.
    """
    try:
        deleted = await redis.delete(stream_name)
        if deleted:
            print(f"Cleared stream '{stream_name}'")
        else:
            print(f"Stream '{stream_name}' did not exist")
    except Exception as e:
        print(f"Error clearing stream '{stream_name}': {e}")

async def start_agents(agents: list[Agent]) -> None:
    await asyncio.gather(*(agent.run() for agent in agents))

async def main():
    try:
        agents = [
            Villager(color="green"),
            Mayor(color="blue"),
            Judge(color="red"),
            ChronicleAgent(color="yellow")
        ]

        await clear_conversation(redis_client, TOWNHALL_STREAM)
        await setup_redis(redis_client, TOWNHALL_STREAM, [agent.role for agent in agents])
        await kickoff_meeting(
            redis_client,
            TOWNHALL_STREAM,
            role="Captain",
            message="There are ever growing numbers of bandits in the woods.  We need to come up with a plan to deal with them before they start attacking our town!"
        )
        await start_agents(agents)
    finally:
        await redis_client.aclose()

async def cleanup():
    try:
        await redis_client.aclose()
    except RuntimeError:
        pass

asyncio.run(main())