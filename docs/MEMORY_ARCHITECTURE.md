# Memory Architecture: SQLite + Faiss Integration Guide

This document outlines the next steps for integrating SQLite (structured storage) with Faiss (vector search) into the Chronicle Agent memory system.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Chronicle Agent                          │
│                                                                 │
│  1. Read N messages from Redis stream                          │
│  2. Generate summary via LLM                                   │
│  3. Store structured record in SQLite                          │
│  4. Generate embedding, store in Faiss                         │
│  5. Update last_summarised_event_id                            │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                                       ▼
┌──────────────────────┐              ┌──────────────────────┐
│       SQLite         │              │        Faiss         │
│  (Source of Truth)   │              │   (Vector Index)     │
├──────────────────────┤              ├──────────────────────┤
│ • Summary text       │              │ • Embedding vectors  │
│ • Message ID range   │              │ • Maps to SQLite ID  │
│ • Stream name        │              │                      │
│ • Timestamps         │              │                      │
│ • Metadata           │              │                      │
└──────────────────────┘              └──────────────────────┘
```

---

## Step 1: Create the Faiss Store

Create `memory/FaissVectorStore.py`:

```python
from pathlib import Path
from typing import Optional
import threading
import numpy as np
import faiss

class FaissVectorStore:
    """
    Handles vector storage and similarity search for summary embeddings.
    Faiss index maps to SQLite summary IDs.
    """
    
    def __init__(self, index_path: str, dimension: int = 768) -> None:
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product for cosine sim
        self.id_map: list[int] = []  # Maps Faiss internal ID → SQLite summary ID
        self._lock = threading.Lock()
    
    def __enter__(self) -> "FaissVectorStore":
        self.load_or_create()
        return self
    
    def __exit__(self, *args) -> None:
        self.save()
    
    def load_or_create(self) -> None:
        with self._lock:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                # Load id_map from companion file
                map_path = self.index_path.with_suffix(".idmap.npy")
                if map_path.exists():
                    self.id_map = np.load(map_path).tolist()
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.id_map = []
    
    def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
        """Add a single embedding, linked to a SQLite summary ID."""
        with self._lock:
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
            self.id_map.append(sqlite_id)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (sqlite_id, score) pairs."""
        with self._lock:
            query = query_embedding / np.linalg.norm(query_embedding)
            scores, indices = self.index.search(query.reshape(1, -1).astype(np.float32), k)
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.id_map) and idx >= 0:
                    results.append((self.id_map[idx], float(score)))
            return results
    
    def save(self) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            np.save(self.index_path.with_suffix(".idmap.npy"), np.array(self.id_map))
```

---

## Step 2: Create an Embedding Service

Create `llms/EmbeddingModel.py`:

```python
from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


def _create_embedding_model() -> MistralAIEmbeddings:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY environment variable is not set.")
    
    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key
    )


EmbeddingModel = _create_embedding_model()
```

---

## Step 3: Extend SQLite Schema

Update `SQLiteSummaryStore` to return the inserted ID:

```python
def insert_summary(self, record: SummaryRecord) -> int:
    """Insert a new summary record. Returns the SQLite row ID."""
    with self._lock:
        # ... existing code ...
        cursor = self.conn.execute(...)
        self.conn.commit()
        return cursor.lastrowid  # Return the ID for Faiss linking
```

---

## Step 4: Create a Unified Memory Manager

Create `memory/MemoryManager.py` to coordinate both stores:

```python
from memory.SQLiteSummaryStore import SQLiteSummaryStore
from memory.FaissVectorStore import FaissVectorStore
from llms.EmbeddingModel import EmbeddingModel
from schemas.core import SummaryRecord
import asyncio


class MemoryManager:
    """
    Coordinates SQLite (structured) and Faiss (vector) storage.
    Ensures both stores stay in sync.
    """
    
    def __init__(self, db_path: str, index_path: str) -> None:
        self.sqlite_store = SQLiteSummaryStore(db_path)
        self.faiss_store = FaissVectorStore(index_path)
    
    def __enter__(self) -> "MemoryManager":
        self.sqlite_store.__enter__()
        self.faiss_store.__enter__()
        return self
    
    def __exit__(self, *args) -> None:
        self.faiss_store.__exit__(*args)
        self.sqlite_store.__exit__(*args)
    
    async def store_summary(self, record: SummaryRecord) -> int:
        """
        Store a summary in both SQLite and Faiss.
        Returns the SQLite row ID.
        """
        # 1. Insert into SQLite (source of truth)
        sqlite_id = self.sqlite_store.insert_summary(record)
        
        # 2. Generate embedding
        embedding = await asyncio.to_thread(
            EmbeddingModel.embed_query,
            record.summary_text
        )
        
        # 3. Add to Faiss with reference to SQLite ID
        self.faiss_store.add(sqlite_id, embedding)
        
        return sqlite_id
    
    async def search_similar(self, query: str, k: int = 5) -> list[SummaryRecord]:
        """
        Find summaries semantically similar to the query.
        """
        # 1. Embed the query
        query_embedding = await asyncio.to_thread(
            EmbeddingModel.embed_query,
            query
        )
        
        # 2. Search Faiss for similar vectors
        results = self.faiss_store.search(query_embedding, k)
        
        # 3. Fetch full records from SQLite
        summaries = []
        for sqlite_id, score in results:
            record = self.sqlite_store.get_summary_by_id(sqlite_id)
            if record:
                summaries.append(record)
        
        return summaries
```

---

## Step 5: Wire into ChronicleAgent

Update `ChronicleAgent` to use the MemoryManager:

```python
from memory.MemoryManager import MemoryManager
from schemas.core import SummaryRecord

class ChronicleAgent:
    def __init__(self, ..., memory_manager: MemoryManager = None) -> None:
        # ...
        self.memory = memory_manager
    
    async def run(self):
        while True:
            # ... existing summarisation logic ...
            
            if len(parsed_messages) >= self.events_before_summary:
                summary = await self.summarise_events(parsed_messages)
                
                record = SummaryRecord(
                    stream_name=self.stream_name,
                    start_msg_id=start_msg_id,
                    end_msg_id=end_msg_id,
                    summary_text=summary
                )
                
                # Store in both SQLite and Faiss
                await self.memory.store_summary(record)
                
                self.last_summarised_event_id = end_msg_id
```

---

## Step 6: Persist `last_summarised_event_id`

Add a metadata table to SQLite:

```sql
CREATE TABLE IF NOT EXISTS agent_state (
    agent_role TEXT PRIMARY KEY,
    last_summarised_event_id TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Then add methods to `SQLiteSummaryStore`:

```python
def get_last_summarised_id(self, agent_role: str) -> Optional[str]:
    ...

def set_last_summarised_id(self, agent_role: str, msg_id: str) -> None:
    ...
```

---

## Step 7: Update main.py

```python
from memory.MemoryManager import MemoryManager
from config import DB_PATH, FAISS_INDEX_PATH

async def main():
    with MemoryManager(DB_PATH, FAISS_INDEX_PATH) as memory:
        agents = [
            Villager(),
            Mayor(),
            Judge(),
            ChronicleAgent(memory_manager=memory)
        ]
        # ...
```

---

## File Structure After Integration

```
memory/
    __init__.py
    SQLiteSummaryStore.py   # Structured storage (source of truth)
    FaissVectorStore.py     # Vector index
    MemoryManager.py        # Coordinates both stores
llms/
    __init__.py
    MistralModel.py         # Chat model
    EmbeddingModel.py       # Embedding model (new)
config.py                   # Add FAISS_INDEX_PATH
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite is source of truth | Faiss index can be rebuilt from SQLite if corrupted |
| Faiss stores only IDs | Full text lives in SQLite; avoids duplication |
| Thread locks on both stores | Safe for multi-threaded access |
| MemoryManager coordinates | Single point of control; easier testing |
| Embedding on write, not read | Summaries are written rarely, searched often |

---

## Future Enhancements

1. **Index rebuild command** — Regenerate Faiss from SQLite if index is lost
2. **Batch embedding** — Embed multiple summaries at once for efficiency
3. **Hybrid search** — Combine vector similarity with SQLite filters (e.g., time range)
4. **Periodic Faiss persistence** — Save index every N inserts instead of on exit
5. **Async SQLite** — Use `aiosqlite` for true async database access
