import asyncio
from typing import Any
from redis_client import redis_client
from config import STREAM_TOPICS, REPLIES_STREAM


class Agent:
    """Base class for townhall agents that listen to and respond to stream messages."""

    def __init__(self, role: str) -> None:
        """Initialize agent with a role (consumer group name).
        
        Args:
            role: Name of this agent's role (e.g., 'mayor', 'judge', 'villagers')
        """
        self.role = role

    async def respond(self, text: str) -> None:
        """Send a response message to the replies stream.
        
        Args:
            text: Response message to send
        """
        print(f"\n{self.role} responding: {text}")
        await redis_client.xadd(
            REPLIES_STREAM,
            {"role": self.role, "text": text}
        )

    async def handle(self, message: dict[str, Any]) -> None:
        """Process an incoming message. Subclasses must override.
        
        Args:
            message: Message data from the stream
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Subclasses must implement handle()")

    async def run(self) -> None:
        """Main event loop: listen for messages and process them."""
        while True:
            messages = await redis_client.xreadgroup(
                groupname=self.role,
                consumername=self.role,
                streams={STREAM_TOPICS: ">"},
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