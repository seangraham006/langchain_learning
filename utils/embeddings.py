"""
Utility for generating text embeddings using sentence-transformers.
This runs locally without API calls.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it.
    Using 'all-mpnet-base-v2' which produces 768-dimensional embeddings.
    """
    return SentenceTransformer('all-mpnet-base-v2')


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a 768-dimensional embedding vector for the given text.
    
    This runs locally using sentence-transformers, no API calls required.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of shape (768,) with dtype float64
    """
    model = _get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float64)
