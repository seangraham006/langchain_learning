import asyncio
from redis_client import redis_client
from config import STREAM_TOPICS

class Agent:
    def __init__(self, role: str):
        self.role = role
    
    async def handle(self, message: dict):
        raise NotImplementedError("This method should be overridden by subclasses.")

    async def run(self):
        while True:
            messages = await redis_client.xreadgroup(
                groupname=self.role,
                consumername=self.role,
                streams={STREAM_TOPICS: '>'},
                count=1,
                block=0
            )

            for _, entries in messages:
                for msg_id, fields in entries:
                    await self.handle(fields)
                    await redis_client.xack(
                        STREAM_TOPICS,
                        self.role,
                        msg_id
                    )