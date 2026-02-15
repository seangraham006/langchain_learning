import asyncio
from typing import Any
from redis_client import redis_client
from typeguard import typechecked

from config import TOWNHALL_STREAM, REPLY_COOLDOWN_SECONDS, MAX_REPLIES_PER_AGENT, CONTEXT_WINDOW
from llms.MistralModel import MistralModel
from schemas.core import StreamMessage, AgentPersona
from utils.parse_redis import process_unread_messages, process_read_messages
from memory.retriever import retrieve, RetrievalResult

class Agent:
    """Base class for townhall agents that listen to and respond to stream messages."""

    # ANSI color codes
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "orange": "\033[38;5;208m",
    }
    RESET = "\033[0m"

    def __init__(
            self,
            role: str = None,
            color: str = None,
            reply_cooldown_seconds: float = REPLY_COOLDOWN_SECONDS,
            stream_name: str = TOWNHALL_STREAM,
            context_window: int = CONTEXT_WINDOW,
            max_replies_per_agent: int = MAX_REPLIES_PER_AGENT
        ) -> None:
        """
        Initialize the agent with default parameters.
        """

        self.role: str = role if role else self.__class__.__name__
        self.color: str = self.COLORS.get(color, "") if color else ""
        self.reply_cooldown_seconds: float = reply_cooldown_seconds
        self.stream_name: str = stream_name
        self.context_window: int = context_window
        self.max_replies_per_agent: int = max_replies_per_agent

        self.replies_sent: int = 0

    def cprint(self, text: str) -> None:
        """Print text in this agent's assigned color with role prefix."""
        print(f"{self.color}[{self.role}] {text}{self.RESET}")

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
    def format_context(self, context: list[StreamMessage]) -> str:
        """
        Format context messages into a string for prompt inclusion.
        """

        return "".join(f"{msg.role if msg.role != self.role else 'You'}: {msg.text}\n\n" for msg in context)

    def format_memories(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieved summaries into a memories block for prompt inclusion.
        """
        if not results:
            return ""

        body = "\n".join(f"- {r.summary.summary_text}" for r in results)
        return body

    @typechecked
    def generate_prompt(self, context: str, memories: str) -> AgentPersona:
        """
        Format the prompt for the agent.  Include their personality, retrieved memories, and current conversation context.
        """

        raise NotImplementedError("Subclasses must implement generate_prompt()")

    @typechecked
    async def think(self, persona: AgentPersona) -> str:
        """
        Generate a response based on the provided context.  Override to change model or generation method.
        """

        try:
            response = await asyncio.to_thread(MistralModel.invoke, persona.formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text
        except Exception as e:
            self.cprint(f"Error generating response: {e}")
            return persona.backup_message

    @typechecked
    async def respond(self, text: str) -> None:
        """
        Post a response message to the redis stream.
        """

        self.cprint(f"responding: {text}")
        await redis_client.xadd(
            self.stream_name,
            {"role": self.role, "text": text}
        )

        await asyncio.sleep(self.reply_cooldown_seconds)

    async def run(self) -> None:
        """Main event loop: listen for messages and process them."""

        while self.replies_sent < self.max_replies_per_agent:
            try:
                # Listen for new messages in the stream
                unread_messages = await redis_client.xreadgroup(
                    groupname=self.role,
                    consumername=self.role,
                    streams={self.stream_name: ">"},
                    count=self.context_window,
                    block=0
                )

                # Parse unread messages and acknowledge them
                parsed_batch: list[StreamMessage] = await process_unread_messages(self.role, self.stream_name, unread_messages)

                # If the last message in the batch is from this agent, skip processing to avoid self-replies
                if not parsed_batch or parsed_batch[-1].role == self.role:
                    continue
                
                # If the batch is smaller than the context window, backfill with historical messages to provide more context for the agent's response
                if len(parsed_batch) < self.context_window:
                    context = await self.get_context(parsed_batch[0].msg_id, count=self.context_window - len(parsed_batch) + 1)
                    parsed_batch = context + parsed_batch

                # This agent was not the last to speak, so proceed to generate response
                formatted_context = self.format_context(parsed_batch)

                memories = await asyncio.to_thread(retrieve, formatted_context, 2)
                formatted_memories = self.format_memories(memories)

                persona = self.generate_prompt(formatted_context, formatted_memories)

                self.cprint(persona.formatted_prompt)

                thought = await self.think(persona)
                await self.respond(thought)
                self.replies_sent += 1
            except asyncio.CancelledError:
                # Graceful shutdown requested — re-raise to exit cleanly
                raise
            except Exception as e:
                # Log error but continue processing — don't let one bad message kill the agent
                self.cprint(f"error in run loop: {e}")
                await asyncio.sleep(1)  # Brief backoff before retrying