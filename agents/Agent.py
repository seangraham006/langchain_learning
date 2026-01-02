import asyncio
from typing import Any
from redis_client import redis_client
from config import TOWNHALL_STREAM, REPLY_COOLDOWN_SECONDS, MAX_REPLIES_PER_MESSAGE
from typeguard import typechecked
from typing import List, Tuple, Dict, Sequence, Union, Mapping
from dataclasses import dataclass
from pydantic import BaseModel

class StreamMessage(BaseModel):
    msg_id: str
    role: str
    text: str

class Agent:
    """Base class for townhall agents that listen to and respond to stream messages."""

    def __init__(self) -> None:
        """Initialize agent with a role (consumer group name).
        
        Args:
            role: Name of this agent's role (e.g., 'mayor', 'judge', 'villagers')
        """

        self.role = self.__class__.__name__
        self.last_reply_time: float = 0.0
        self.stream_name: str = TOWNHALL_STREAM
        self.context_window: int = 5  # Number of preceding messages to consider as context

    @typechecked
    async def process_unread_messages(self, raw: Any) -> list[StreamMessage]:
        """
        Process a batch of unread messages from the stream and acknowledge them.
        """

        parsed_batch = []

        for stream_name, entries in raw:
            for msg_id, fields in entries:
                await redis_client.xack(
                    self.stream_name,
                    self.role,
                    msg_id
                )

                role = fields.get("role", None)
                text = fields.get("text", None)

                if role is None or text is None:
                    continue

                message = StreamMessage(
                    msg_id=msg_id,
                    role=role,
                    text=text
                )

                parsed_batch.append(message)

        return parsed_batch

    @typechecked
    def process_read_messages(self, raw: Any) -> list[StreamMessage]:
        """
        Process a batch of read messages from the stream.
        """

        parsed_batch = []

        for message in raw:
            msg_id, fields = message

            role = fields.get("role", None)
            text = fields.get("text", None)

            if role is None or text is None:
                continue

            message = StreamMessage(
                msg_id=msg_id,
                role=role,
                text=text
            )
            parsed_batch.append(message)

        return parsed_batch

    @typechecked
    async def get_context(self, bound_msg_id: str, count: int) -> list[StreamMessage]:
        """
        Retrieve context messages from the redis stream before a given message ID.
        """

        messages = await redis_client.xrevrange(
            self.stream_name,
            max=bound_msg_id,
            min='-',
            count=count
        )
        messages.reverse()

        parsed_batch = self.process_read_messages(messages)

        for message in parsed_batch:
            if message.msg_id == bound_msg_id:
                parsed_batch.remove(message)

        return parsed_batch

    # @typechecked
    # async def should_respond_to(self, context: list[dict[str, Any]]) -> bool:
    #     """Determine if the agent should respond to a given message."""
    #     # Basic implementation: always respond

    #     #Rule: An agent should not respond to themselves
    #     if message.get("role") == self.role:
    #         return False
    #     return True

    @typechecked
    def format_context(self, context: list[StreamMessage]) -> str:
        """
        Format context messages into a string for prompt inclusion.
        """

        return "".join(f"{msg.role if msg.role != self.role else 'You'}: {msg.text}\n\n" for msg in context)

    async def think(self, context: str) -> str:
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

    @typechecked
    async def respond(self, text: str) -> None:
        """
        Post a response message to the redis stream.
        """

        print(f"\n{self.role} responding: {text}")
        await redis_client.xadd(
            self.stream_name,
            {"role": self.role, "text": text}
        )

    async def run(self) -> None:
        """Main event loop: listen for messages and process them."""

        while True:
            # Listen for new messages in the stream
            unread_messages = await redis_client.xreadgroup(
                groupname=self.role,
                consumername=self.role,
                streams={self.stream_name: ">"},
                count=self.context_window,
                block=0
            )

            print(f"{self.role} received {len(unread_messages)} new messages.  Unread messages: {unread_messages}")
            parsed_batch = await self.process_unread_messages(unread_messages)
            
            if len(parsed_batch) < self.context_window:
                context = await self.get_context(parsed_batch[0].msg_id, count=self.context_window - len(parsed_batch) + 1)
                parsed_batch = context + parsed_batch

            formatted_context = self.format_context(parsed_batch)

            print(f"Formatted thought of {self.role}:\n{formatted_context}\n")
            thought = await self.think(formatted_context)
            await self.respond(thought)