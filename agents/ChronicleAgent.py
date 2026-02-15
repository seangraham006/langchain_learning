from typeguard import typechecked
import asyncio
import numpy as np

from config import EVENTS_BEFORE_SUMMARY, TOWNHALL_STREAM, SUMMARY_SOFT_LIMIT_WORDS, RETRIES_FOR_SUMMARISATION, SUMMARISATION_CHECK_COOLDOWN_SECONDS
from schemas.core import StreamMessage, SummaryRecord
from redis_client import redis_client
from utils.parse_redis import process_read_messages
from utils.embeddings import generate_embedding
from llms.MistralModel import MistralModel
from memory.SQLiteSummaryStore import SQLiteSummaryStore
from memory.FaissVectorStore import FaissVectorStore

class SummarisationError(Exception):
    """Raised when summarisation fails after all retries."""
    pass

class ChronicleAgent:
    """Agent that creates a rolling summary of events in the townhall and saves them to a vector database."""

    # ANSI color codes (same as Agent)
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
            stream_name: str = TOWNHALL_STREAM,
            summarisation_check_cooldown_seconds: float = SUMMARISATION_CHECK_COOLDOWN_SECONDS,
            retries_for_summarisation: int = RETRIES_FOR_SUMMARISATION,
            events_before_summary: int = EVENTS_BEFORE_SUMMARY
        ) -> None:
        """
        Initialize the ChronicleAgent.
        """

        self.role: str = role if role else self.__class__.__name__
        self.color: str = self.COLORS.get(color, "") if color else ""
        self.stream_name: str = stream_name
        self.summarisation_check_cooldown_seconds = summarisation_check_cooldown_seconds
        self.retries_for_summarisation: int = retries_for_summarisation
        self.events_before_summary: int = events_before_summary

        self.last_summarised_event_id: str | None = None

    def cprint(self, text: str) -> None:
        """Print text in this agent's assigned color with role prefix."""
        print(f"{self.color}[{self.role}] {text}{self.RESET}")

    @typechecked
    async def summarise_events(self, events: list[StreamMessage]) -> str:
        """
        Send a list of events to an LLM to be summarised, with retries on failure.
        """

        context = "".join(f"{msg.role}: {msg.text}\n\n" for msg in events)
        prompt = f"""
        You are an AI agent tasked with summarising the following events from a townhall meeting.

        Create an objective, concise summary that captures the key points and decisions made during these events.

        Be sure to include any action items, facts, and important discussions.

        Keep the summary under {SUMMARY_SOFT_LIMIT_WORDS} words.

        Here are the events to summarise:
        {context}
        """

        retry_count = 0
        while retry_count < self.retries_for_summarisation:
            try:
                response = await asyncio.to_thread(MistralModel.invoke, prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                return response_text.strip()
            except Exception as e:
                print(f"Error during summarisation attempt {retry_count + 1}: {e}")
                retry_count += 1

        raise SummarisationError(f"Failed to summarise events after {self.retries_for_summarisation} attempts.")

    def next_id(self, stream_id: str) -> str:
        ms, seq = stream_id.split("-")
        return f"{ms}-{int(seq) + 1}"

    @typechecked
    async def run(self):
        """
        Continuously monitor the configured Redis stream and create rolling summaries.

        The loop wakes every self.summarisation_check_cooldown_seconds, reads up to
        self.events_before_summary unread events after the last summarised message,
        and only proceeds when enough events are available. When triggered, it:

        1. Summarises the batch with summarise_events.
        2. Generates an embedding for the summary.
        3. Persists the summary + embedding bytes in SQLite (source of truth).
        4. Adds the embedding to Faiss for fast vector lookup.
        5. Advances self.last_summarised_event_id to avoid reprocessing.

        The method runs until cancelled. asyncio.CancelledError is re-raised for
        graceful shutdown, while other exceptions are logged and retried after a brief
        backoff so the agent remains resilient.
        """

        while True:
            await asyncio.sleep(self.summarisation_check_cooldown_seconds)

            try:
                # Set the bottom range of what messages you will fetch
                start_id = (
                    "-" if self.last_summarised_event_id is None
                    else self.next_id(self.last_summarised_event_id)
                )

                # Fetch all messages after start_id, up to the number of events we want to summarise
                unread_messages = await redis_client.xrange(
                    self.stream_name,
                    min=start_id,
                    max="+",
                    count=self.events_before_summary
                )

                # Parse raw Redis messages into StreamMessage objects
                parsed_messages: list[StreamMessage] = process_read_messages(unread_messages)

                if len(parsed_messages) >= self.events_before_summary:
                    # Determine the range of message IDs being summarised
                    start_msg_id = parsed_messages[0].msg_id
                    end_msg_id = parsed_messages[-1].msg_id

                    try:
                        summary = await self.summarise_events(parsed_messages)
                    except SummarisationError as e:
                        self.cprint(f"Failed to summarise events: {e}")
                        continue

                    # Here you would save the summary to a vector database
                    self.cprint(f"Generated Summary for messages from {start_msg_id} to {end_msg_id}:\n{summary}")

                    # Generate embedding once (expensive operation)
                    embedding: np.ndarray = generate_embedding(summary)
                    self.cprint(f"Generated embedding.")
                    embedding_bytes: bytes = embedding.tobytes()  # Serialize to bytes for SQLite BLOB

                    summary_record: SummaryRecord = SummaryRecord(
                        stream_name=self.stream_name,
                        start_msg_id=start_msg_id,
                        end_msg_id=end_msg_id,
                        summary_text=summary,
                        embedding=embedding_bytes
                    )

                    # Store summary + embedding in SQLite (source of truth)
                    with SQLiteSummaryStore() as summary_store:
                        summary_id = summary_store.insert_summary(summary_record)

                    # Store in Faiss index (rebuildable cache for fast search)
                    with FaissVectorStore() as vector_store:
                        vector_store.add(sqlite_id=summary_id, embedding=embedding)

                    self.last_summarised_event_id = end_msg_id
            except asyncio.CancelledError:
                # Graceful shutdown requested — re-raise to exit cleanly
                raise
            except Exception as e:
                # Log error but continue processing
                self.cprint(f"error in run loop: {e}")
                await asyncio.sleep(1)  # Brief backoff before retrying