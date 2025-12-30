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
        self.last_reply_time: float = 0.0
        self.stream_name: str = TOWNHALL_STREAM
        self.context_window: int = 5  # Number of preceding messages to consider as context

    async def respond(self, text: str) -> None:
        """Send a response message to the replies stream.
        
        Args:
            text: Response message to send
        """
        print(f"\n{self.role} responding: {text}")
        await redis_client.xadd(
            self.stream_name,
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

        if message.get("role") == self.role:
            return False

        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_reply_time < REPLY_COOLDOWN_SECONDS:
            return False

        return True

    async def get_context(self, bound_msg_id: str, count: int) -> list[dict[str, Any]]:
        """
        Retrieve context messages preceding the given message ID.
        """

        messages = await redis_client.xrevrange(
            self.stream_name,
            max=bound_msg_id,
            min='-',
            count=count
        )
        messages.reverse()

        parsed_batch = []

        for message in messages:
            msg_id, fields = message
            if msg_id == bound_msg_id:
                continue
            parsed_batch.append({"msg_id": msg_id, "fields": fields})

        return parsed_batch

    async def think(self, message: dict[str, Any], msg_id: str, context: list[dict[str, Any]]) -> None:
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
        raise NotImplementedError("Subclasses must implement think()")

    async def run(self) -> None:
        """Main event loop: listen for messages and process them."""

        while True:
            unread_messages = await redis_client.xreadgroup(
                groupname=self.role,
                consumername=self.role,
                streams={self.stream_name: ">"},
                count=self.context_window,
                block=0
            )

            if not unread_messages:
                continue

            print(f"{self.role} reading {len(unread_messages)} message(s):\n{unread_messages}\n")

            parsed_batch = []
            
            for stream_name, entries in unread_messages:
                for msg_id, fields in entries:
                    parsed_batch.append({"msg_id": msg_id, "fields": fields})
                    await redis_client.xack(
                        self.stream_name,
                        self.role,
                        msg_id
                    )

            if len(parsed_batch) < self.context_window:
                context = await self.get_context(parsed_batch[0]["msg_id"], count=self.context_window - len(parsed_batch))
                parsed_batch = context + parsed_batch


            print(f"{self.role} parsed batch: {parsed_batch}")
            thought = await self.think(parsed_batch)
            await self.respond(thought)