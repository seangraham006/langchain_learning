from typeguard import typechecked
import asyncio
import time

from config import EVENTS_BEFORE_SUMMARY, TOWNHALL_STREAM, SUMMARY_SOFT_LIMIT_WORDS, RETRIES_FOR_SUMMARISATION
from schemas.core import StreamMessage
from redis_client import redis_client
from utils.parse_redis import process_read_messages
from llms.MistralModel import MistralModel

class ChronicleAgent:
    """Agent that creates a rolling summary of events in the townhall and saves them to a vector database."""

    def __init__(
            self,
            role: str = None,
            stream_name: str = TOWNHALL_STREAM,
            events_before_summary: int = EVENTS_BEFORE_SUMMARY
        ) -> None:
        """
        Initialize the ChronicleAgent.
        """

        self.role: str = role if role else self.__class__.__name__
        self.stream_name: str = stream_name
        self.events_before_summary: int = events_before_summary

        self.last_summarised_event_id: str | None = None

    @typechecked
    async def summarise_events(self, events: list[StreamMessage], retries: int = RETRIES_FOR_SUMMARISATION) -> str:
        """
        Summarise a list of events into a concise summary.
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
        while retry_count < retries:
            try:
                response = await asyncio.to_thread(MistralModel.invoke, prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                return response_text.strip()
            except Exception as e:
                print(f"Error during summarisation attempt {retry_count + 1}: {e}")
                retry_count += 1

        raise Exception("Failed to summarise events after multiple attempts.")

    def next_id(self, stream_id: str) -> str:
        ms, seq = stream_id.split("-")
        return f"{ms}-{int(seq) + 1}"

    async def run(self):
        """
        Run the ChronicleAgent to summarise events.
        """

        while True:
            time.sleep(1)

            start_id = (
                "-" if self.last_summarised_event_id is None
                else self.next_id(self.last_summarised_event_id)
            )

            unread_messages = await redis_client.xrange(
                self.stream_name,
                min=start_id,
                max="+",
                count=self.events_before_summary
            )

            parsed_messages: list[StreamMessage] = process_read_messages(unread_messages)

            # print(f"\n{self.role} num parsed_messages: {len(parsed_messages)}")

            if len(parsed_messages) >= self.events_before_summary:

                end_id = parsed_messages[-1].msg_id

                try:
                    summary = await self.summarise_events(parsed_messages)
                except Exception as e:
                    print(f"\n{self.role} Failed to summarise events: {e}")
                    continue

                # Here you would save the summary to a vector database
                print(f"\n{self.role} Generated Summary for messages from {start_id} to {end_id}:\n{summary}")
                self.last_summarised_event_id = end_id