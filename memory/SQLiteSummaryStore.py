import sqlite3
import threading
from pathlib import Path
from typing import Optional

from config import DB_PATH
from schemas.core import SummaryRecord


class SQLiteSummaryStore:
    """
    Handles persistent storage of Chronicle summaries.
    SQLite is used as the source of truth for derived memory.
    
    Thread-safe via internal lock. Supports context manager protocol.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        """
        Initialize SQLiteSummaryStore with the path to the database file.
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def __enter__(self) -> "SQLiteSummaryStore":
        """Context manager entry: connect and create schema."""
        self.connect()
        self.create_schema()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: close connection."""
        self.close()

    def connect(self) -> None:
        """
        Connect to SQLite database. Creates file if missing.
        Fails loudly if connection is impossible.
        No-op if already connected.
        """
        with self._lock:
            if self.conn is not None:
                return  # Already connected
            
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,  # safe for multi-thread if access is serialized via lock
                )
                self.conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
                self.conn.execute("PRAGMA journal_mode=WAL;")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to SQLite DB: {e}") from e

    def create_schema(self) -> None:
        """
        Create tables and indexes if they do not exist.
        Safe to call multiple times.
        """
        with self._lock:
            if self.conn is None:
                raise RuntimeError("SQLite connection not initialized")

            try:
                cursor = self.conn.cursor()
                cursor.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stream_name TEXT NOT NULL,
                        start_msg_id TEXT NOT NULL,
                        end_msg_id TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(stream_name, start_msg_id, end_msg_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_summaries_stream
                        ON summaries(stream_name);

                    CREATE INDEX IF NOT EXISTS idx_summaries_end_msg
                        ON summaries(end_msg_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_summaries_created
                        ON summaries(created_at);
                    """
                )
                self.conn.commit()
            except Exception as e:
                raise RuntimeError(f"Failed to create SQLite schema: {e}") from e

    def insert_summary(self, record: SummaryRecord) -> None:
        """
        Insert a new summary record.
        Raises RuntimeError if duplicate (stream_name, start_msg_id, end_msg_id) exists.
        """
        with self._lock:
            if self.conn is None:
                raise RuntimeError("SQLite connection not initialized")

            try:
                self.conn.execute(
                    """
                    INSERT INTO summaries (
                        stream_name,
                        start_msg_id,
                        end_msg_id,
                        summary_text
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (record.stream_name, record.start_msg_id, record.end_msg_id, record.summary_text),
                )
                self.conn.commit()
            except sqlite3.IntegrityError as e:
                raise RuntimeError(f"Duplicate summary record: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to insert summary: {e}") from e

    def insert_summaries(self, records: list[SummaryRecord]) -> None:
        """
        Batch insert multiple summary records in a single transaction.
        More efficient than calling insert_summary() repeatedly.
        """
        with self._lock:
            if self.conn is None:
                raise RuntimeError("SQLite connection not initialized")

            try:
                self.conn.executemany(
                    """
                    INSERT INTO summaries (
                        stream_name,
                        start_msg_id,
                        end_msg_id,
                        summary_text
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    [(r.stream_name, r.start_msg_id, r.end_msg_id, r.summary_text) for r in records],
                )
                self.conn.commit()
            except sqlite3.IntegrityError as e:
                self.conn.rollback()
                raise RuntimeError(f"Duplicate summary record in batch: {e}") from e
            except Exception as e:
                self.conn.rollback()
                raise RuntimeError(f"Failed to insert summaries: {e}") from e

    def get_summaries_by_stream(self, stream_name: str, limit: int = 100) -> list[SummaryRecord]:
        """
        Retrieve summaries for a given stream, ordered by creation time (newest first).
        """
        with self._lock:
            if self.conn is None:
                raise RuntimeError("SQLite connection not initialized")

            try:
                cursor = self.conn.execute(
                    """
                    SELECT stream_name, start_msg_id, end_msg_id, summary_text
                    FROM summaries
                    WHERE stream_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (stream_name, limit),
                )
                return [
                    SummaryRecord(
                        stream_name=row["stream_name"],
                        start_msg_id=row["start_msg_id"],
                        end_msg_id=row["end_msg_id"],
                        summary_text=row["summary_text"],
                    )
                    for row in cursor.fetchall()
                ]
            except Exception as e:
                raise RuntimeError(f"Failed to get summaries: {e}") from e

    def get_latest_summary(self, stream_name: str) -> Optional[SummaryRecord]:
        """
        Retrieve the most recent summary for a given stream.
        Returns None if no summaries exist.
        """
        with self._lock:
            if self.conn is None:
                raise RuntimeError("SQLite connection not initialized")

            try:
                cursor = self.conn.execute(
                    """
                    SELECT stream_name, start_msg_id, end_msg_id, summary_text
                    FROM summaries
                    WHERE stream_name = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (stream_name,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return SummaryRecord(
                    stream_name=row["stream_name"],
                    start_msg_id=row["start_msg_id"],
                    end_msg_id=row["end_msg_id"],
                    summary_text=row["summary_text"],
                )
            except Exception as e:
                raise RuntimeError(f"Failed to get latest summary: {e}") from e

    def get_summary_after(self, stream_name: str, msg_id: str) -> Optional[SummaryRecord]:
        """
        Retrieve the first summary that starts after the given message ID.
        Returns None if no such summary exists.
        """
        with self._lock:
            if self.conn is None:
                raise RuntimeError("SQLite connection not initialized")

            try:
                cursor = self.conn.execute(
                    """
                    SELECT stream_name, start_msg_id, end_msg_id, summary_text
                    FROM summaries
                    WHERE stream_name = ? AND start_msg_id > ?
                    ORDER BY start_msg_id ASC
                    LIMIT 1
                    """,
                    (stream_name, msg_id),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return SummaryRecord(
                    stream_name=row["stream_name"],
                    start_msg_id=row["start_msg_id"],
                    end_msg_id=row["end_msg_id"],
                    summary_text=row["summary_text"],
                )
            except Exception as e:
                raise RuntimeError(f"Failed to get summary after msg_id: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self.conn:
                self.conn.close()
                self.conn = None