from typeguard import typechecked

from config import EVENTS_BEFORE_SUMMARY, TOWNHALL_STREAM
from schemas.core import StreamMessage

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

    def run(self):
        """
        Run the ChronicleAgent to summarise events.
        """

        pass  # Implementation of summarisation logic goes here