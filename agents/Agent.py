import asyncio
from typing import Any
from redis_client import redis_client
from typeguard import typechecked
import time

from config import TOWNHALL_STREAM, REPLY_COOLDOWN_SECONDS, MAX_REPLIES_PER_AGENT, CONTEXT_WINDOW
from llms.MistralModel import MistralModel
from schemas.core import StreamMessage, AgentPersona
from utils.parse_redis import process_unread_messages, process_read_messages

class Agent:
    """Base class for townhall agents that listen to and respond to stream messages."""

    def __init__(
            self,
            role: str = None,
            reply_cooldown_seconds: float = REPLY_COOLDOWN_SECONDS,
            stream_name: str = TOWNHALL_STREAM,
            context_window: int = CONTEXT_WINDOW,
            max_replies_per_agent: int = MAX_REPLIES_PER_AGENT
        ) -> None:
        """
        Initialize the agent with default parameters.
        """

        self.role: str = role if role else self.__class__. __name__
        self.reply_cooldown_seconds: float = reply_cooldown_seconds
        self.stream_name: str = stream_name
        self.context_window: int = context_window
        self.max_replies_per_agent: int = max_replies_per_agent

        self.replies_sent: int = 0

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

        parsed_batch: list[StreamMessage] = process_read_messages(messages)

        for message in parsed_batch:
            if message.msg_id == bound_msg_id:
                parsed_batch.remove(message)

        return parsed_batch

    @typechecked
    def should_respond_to(self, context: list[StreamMessage]) -> bool:
        """Determine if the agent should respond the given context."""

        last_message = context[-1]
        if last_message.role == self.role:
            return False

        return True

    @typechecked
    def format_context(self, context: list[StreamMessage]) -> str:
        """
        Format context messages into a string for prompt inclusion.
        """

        return "".join(f"{msg.role if msg.role != self.role else 'You'}: {msg.text}\n\n" for msg in context)

    @typechecked
    def generate_prompt(self, context: str) -> AgentPersona:
        """
        Format the prompt for the agent.  Include their personality and what they are to do with the provided information.
        """

        raise NotImplementedError("Subclasses must implement format_prompt()")

    @typechecked
    async def think(self, persona: AgentPersona) -> str:
        """
        Generate a response based on the provided context.  Override to change model or generation method.
        """

        try:
            response = await asyncio.to_thread(MistralModel.invoke, persona.dynamically_generated_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text
        except Exception as e:
            print(f"\n{self.__class__} Error generating response: {e}")
            return persona.backup_message

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

        time.sleep(self.reply_cooldown_seconds)

    async def run(self) -> None:
        """Main event loop: listen for messages and process them."""

        while self.replies_sent < self.max_replies_per_agent:
            # Listen for new messages in the stream
            unread_messages = await redis_client.xreadgroup(
                groupname=self.role,
                consumername=self.role,
                streams={self.stream_name: ">"},
                count=self.context_window,
                block=0
            )

            # print(f"{self.role} received {len(unread_messages)} new messages.  Unread messages: {unread_messages}")
            parsed_batch: list[StreamMessage] = await process_unread_messages(self.role, self.stream_name, unread_messages)
            
            if len(parsed_batch) < self.context_window:
                context = await self.get_context(parsed_batch[0].msg_id, count=self.context_window - len(parsed_batch) + 1)
                parsed_batch = context + parsed_batch

            if not self.should_respond_to(parsed_batch):
                continue

            formatted_context = self.format_context(parsed_batch)
            persona = self.generate_prompt(formatted_context)
            thought = await self.think(persona)
            await self.respond(thought)