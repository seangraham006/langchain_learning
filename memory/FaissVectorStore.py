from pathlib import Path
from typing import Optional, Self
import threading
import numpy as np
import faiss

class FaissVectorStore:
    """
    Handles vector storage and similarity search for summary embeddings.
    Faiss index maps to SQLite summary IDs.
    """
    
    def __init__(self, index_path: str, dimension: int = 768) -> None:
        # Where Faiss index is stored on disk
        self.index_path = Path(index_path)

        # Embedding size (must match embedding model).  Used by Faiss to initialize index
        self.dimension = dimension

        # Flat: Brute-force search
        # IP: Inner Product (for cosine similarity with normalized vectors)
        self.index: Optional[faiss.IndexFlatIP] = None

        # Maps Faiss internal ID → SQLite summary ID
        self.id_map: list[int] = []

        self._lock = threading.Lock()
    
    def __enter__(self) -> Self:
        self.load_or_create()
        return self
    
    def __exit__(self, *args) -> None:
        self.save()
    
    def load_or_create(self) -> None:
        with self._lock:
            map_path = self.index_path.with_suffix(".idmap.npy")

            # Check both Faiss index and id_map file exist
            if self.index_path.exists() and map_path.exists():
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    self.id_map = np.load(map_path).tolist()

                    if self.index.ntotal != len(self.id_map):
                        raise ValueError("Faiss index size and id_map size mismatch.")
                    
                except Exception as e:
                    print(f"Error loading Faiss index or id_map: {e}. Creating new index.")
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.id_map = []
            else:
                # Missing one or both files, create new index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.id_map = []
    
    def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
        """
        Add a single embedding, linked to a SQLite summary ID.
        
        Args:
            sqlite_id: The SQLite summary ID to link to this embedding.
            embedding: The embedding vector to add (a list of all its values in each dimension).
        """
        
        with self._lock:
            # Normalize for cosine similarity
            euclidean_norm = np.linalg.norm(embedding)
            embedding = embedding / euclidean_norm
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
            self.id_map.append(sqlite_id)

            if self.index.ntotal != len(self.id_map):
                raise ValueError("Faiss index size and id_map size mismatch after addition.")
    
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