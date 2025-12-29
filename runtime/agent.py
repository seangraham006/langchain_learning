import asyncio
from typing import Any
from redis_client import redis_client
from config import TOWNHALL_STREAM, REPLY_COOLDOWN_SECONDS, MAX_REPLIES_PER_MESSAGE


class Agent:
    """Base class for townhall agents that listen to and respond to stream messages."""

    def __init__(self, role: str) -> None:
        """Initialize agent with a role (consumer group name).
        
        Args:
            role: Name of this agent's role (e.g., 'mayor', 'judge', 'villagers')
        """
        self.role = role
        self.replied_to: set[str] = set()
        self.last_reply_time: float = 0.0

    async def respond(self, text: str) -> None:
        """Send a response message to the replies stream.
        
        Args:
            text: Response message to send
        """
        print(f"\n{self.role} responding: {text}")
        await redis_client.xadd(
            TOWNHALL_STREAM,
            {"role": self.role, "text": text}
        )

    async def should_respond_to(self, msg_id: str, message: dict[str, Any]) -> bool:
        """Determine if the agent should respond to a given message.
        
        Args:
            msg_id: ID of the message in the stream
            message: Message data from the stream

        Returns:
            True if the agent should respond, False otherwise
        """

        if message.get("role", message.get("speaker")) == self.role:
            return False

        if msg_id in self.replied_to:
            return False

        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_reply_time < REPLY_COOLDOWN_SECONDS:
            return False

        return True

    async def get_context(self, msg_id: str, count: int = 5) -> list[dict[str, Any]]:
        """Retrieve context messages preceding the given message ID.
        
        Args:
            msg_id: ID of the message in the stream
            count: Number of preceding messages to retrieve

        Returns:
            List of preceding message data dictionaries
        """
        messages = await redis_client.xrevrange(
            TOWNHALL_STREAM,
            max=msg_id,
            min='-',
            count=count
        )

        context = []
        for _, fields in reversed(messages):
            context.append(fields)

        return context

    async def handle(self, message: dict[str, Any], msg_id: str, context: list[dict[str, Any]]) -> None:
        """Process an incoming message. Subclasses must override.
        
        Args:
            message: Message data from the stream
            msg_id: ID of the message in the stream
            context: List of preceding messages for context        Args:
            message: Message data from the stream
            context: Context messages preceding the current message
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
                streams={TOWNHALL_STREAM: ">"},
                count=1,
                block=0
            )

            for _, entries in messages:
                for msg_id, fields in entries:
                    try:
                        if await self.should_respond_to(msg_id, fields):
                            context = await self.get_context(msg_id, count=5)
                            await self.handle(fields, msg_id, context)
                            self.replied_to.add(msg_id)

                    except Exception as e:
                        print(f"{self.__class__} Error handling message {msg_id}: {e}")

                    finally:
                        await redis_client.xack(
                            TOWNHALL_STREAM,
                            self.role,
                            msg_id
                        )