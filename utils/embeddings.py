"""
Utility for generating text embeddings using sentence-transformers.
This runs locally without API calls.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import threading

_lock = threading.Lock()
_model: SentenceTransformer = None


def _get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it.
    Using 'all-mpnet-base-v2' which produces 768-dimensional embeddings.
    Thread-safe via lock — ensures only one thread loads the model.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                _model = SentenceTransformer('all-mpnet-base-v2', device=device)
    return _model


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
    with _lock, torch.no_grad():
        embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float64)
