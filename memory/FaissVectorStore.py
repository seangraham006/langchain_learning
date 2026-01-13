from pathlib import Path
from typing import Optional
import threading
import numpy as np
import faiss
from config import FAISS_INDEX_PATH

class FaissVectorStore:
    """
    Handles vector storage and similarity search for summary embeddings.
    Faiss index maps to SQLite summary IDs.
    """
    
    def __init__(self, index_path: str = FAISS_INDEX_PATH, dimension: int = 768) -> None:
        # Where Faiss index is stored on disk
        self.index_path = Path(index_path)

        # Embedding size (must match embedding model).  Used by Faiss to initialize index
        self.dimension = dimension

        # Flat: Brute-force search
        # IP: Inner Product (for cosine similarity with normalized vectors)
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        # Maps Faiss internal ID → SQLite summary ID
        self.id_map: list[int] = []

        self._lock = threading.Lock()
    
    def __enter__(self) -> "FaissVectorStore":
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
                    self.faiss_index = faiss.read_index(str(self.index_path))
                    self.id_map = np.load(map_path).tolist()

                    if self.faiss_index.ntotal != len(self.id_map):
                        raise ValueError("Faiss index size and id_map size mismatch.")
                    
                except Exception as e:
                    print(f"Error loading Faiss index or id_map: {e}. Creating new index.")
                    self.faiss_index = faiss.IndexFlatIP(self.dimension)
                    self.id_map = []
            else:
                # Missing one or both files, create new index
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                self.id_map = []
    
    def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
        """
        Add a single embedding, linked to a SQLite summary ID.
        
        Args:
            sqlite_id: The SQLite summary ID to link to this embedding.
            embedding: The embedding vector to add (a list of all its values in each dimension).
        """

        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match index dimension {self.dimension}.")

        with self._lock:
            if sqlite_id in self.id_map:
                raise ValueError(f"SQLite ID {sqlite_id} already exists in Faiss index.")

            if self.faiss_index.ntotal != len(self.id_map):
                raise ValueError("Faiss index size and id_map size mismatch.")

            # Normalize for cosine similarity
            # Normalize the embedding vector to unit length, enabling Faiss to use its dot product func to calculate cosine similarity.

            # 1. Compute Euclidean norm (This is the resultant vector length in n-dimensional space)
            euclidean_norm: float = np.linalg.norm(embedding)

            if euclidean_norm == 0:
                raise ValueError("Cannot add zero vector to Faiss index.")

            # 2. Divide each component by the Euclidean norm to get unit vector
            normalised_embedding: np.ndarray = embedding / euclidean_norm

            # 3. Insert into Faiss index
            self.faiss_index.add(normalised_embedding.reshape(1, -1).astype(np.float32))

            # 4. Update id_map
            self.id_map.append(sqlite_id)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Search for the k most similar vectors to the query.
        
        Args:
            query_embedding: The query vector (will be normalised internally)
            k: Number of results to return
            
        Returns:
            List of (sqlite_id, similarity_score) tuples, sorted by descending similarity.
            Scores range from -1 to 1 (cosine similarity).
        """
        # Validate dimension before acquiring lock
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[0]} does not match index dimension {self.dimension}.")
        
        with self._lock:
            # Handle empty index
            if self.faiss_index.ntotal == 0:
                return []
            
            # Validate sync state
            if self.faiss_index.ntotal != len(self.id_map):
                raise RuntimeError("Index corrupted: Faiss/id_map out of sync")
            
            # Normalise query to unit length
            euclidean_norm: float = np.linalg.norm(query_embedding)
            if euclidean_norm == 0:
                raise ValueError("Query embedding cannot be zero vector.")
            
            normalised_query_embedding = query_embedding / euclidean_norm

            # Perform search - Faiss returns (scores, indices)
            scores, indices = self.faiss_index.search(
                normalised_query_embedding.reshape(1, -1).astype(np.float32), 
                k
            )
            
            # Map Faiss indices to SQLite IDs
            results = []
            for idx, score in zip(indices[0], scores[0]):
                # Filter invalid indices (Faiss returns -1 when k > ntotal)
                if 0 <= idx < len(self.id_map):
                    results.append((self.id_map[idx], float(score)))
            return results
    
    def save(self) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.faiss_index, str(self.index_path))
            np.save(self.index_path.with_suffix(".idmap.npy"), np.array(self.id_map))