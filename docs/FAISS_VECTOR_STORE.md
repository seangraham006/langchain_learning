# FaissVectorStore: Complete Technical Guide

## Table of Contents
1. [Why IndexFlatIP?](#why-indexflatip)
2. [Native Similarity Support](#native-similarity-support)
3. [Complete Pipeline: String → Vector → Storage → Search](#complete-pipeline)
4. [Code Analysis & Improvements](#code-analysis--improvements)

---

## Why IndexFlatIP?

### What is IndexFlatIP?

`faiss.IndexFlatIP` is a Faiss index type where:
- **Flat**: Brute-force exhaustive search (compares query against ALL vectors)
- **IP**: Inner Product (dot product) as the similarity metric

### Pros

| Advantage | Explanation |
|-----------|-------------|
| **100% Accuracy** | Brute-force guarantees finding the true nearest neighbours - no approximations |
| **No Training Required** | Ready to use immediately, unlike IVF/HNSW indexes that need clustering |
| **Simple Implementation** | Few moving parts, easy to debug and understand |
| **Good for Small-Medium Datasets** | Perfectly efficient for up to ~100k vectors |
| **Low Memory Overhead** | Stores only raw vectors, no auxiliary structures |
| **Supports Incremental Adds** | Can add vectors one at a time without rebuilding |

### Cons

| Disadvantage | Explanation |
|--------------|-------------|
| **O(n) Search Complexity** | Must compare query against every stored vector |
| **Slow at Scale** | At 1M+ vectors, search becomes noticeably slow |
| **CPU Intensive** | Every search does `n × dimension` multiplications |
| **No Built-in Filtering** | Can't efficiently filter by metadata during search |

### When to Choose IndexFlatIP

```
✅ Use IndexFlatIP when:
   - Dataset < 100k vectors
   - Accuracy is critical (no approximate results acceptable)
   - Prototyping / development phase
   - Vectors are added incrementally
   - Simplicity is preferred over raw speed

❌ Consider alternatives when:
   - Dataset > 500k vectors → Use IndexIVFFlat or IndexHNSWFlat
   - Need sub-millisecond search → Use approximate indexes
   - Memory constrained → Use IndexScalarQuantizer
```

### Alternatives Comparison

| Index Type | Speed | Accuracy | Memory | Training |
|------------|-------|----------|--------|----------|
| IndexFlatIP | Slow (exact) | 100% | Low | None |
| IndexIVFFlat | Fast | 95-99% | Medium | Required |
| IndexHNSWFlat | Very Fast | 95-99% | High | None |
| IndexScalarQuantizer | Fast | 90-95% | Very Low | None |

---

## Native Similarity Support

### What IndexFlatIP Actually Computes

**IndexFlatIP computes RAW DOT PRODUCT only.** It does NOT natively compute cosine similarity.

```python
# What Faiss IndexFlatIP calculates:
dot_product(a, b) = Σ(a[i] × b[i]) for all dimensions i

# This is NOT the same as:
cosine_similarity(a, b) = dot_product(a, b) / (||a|| × ||b||)
```

### The Normalisation Trick

Since Faiss only knows dot product, we **transform** the problem:

1. **Pre-normalise all vectors** to unit length (||v|| = 1)
2. When both vectors have length 1:
   ```
   cosine(a, b) = dot(a, b) / (1 × 1) = dot(a, b)
   ```
3. Now **dot product equals cosine similarity**

This is why every vector is divided by its Euclidean norm before storage.

### Other Faiss Index Types

| Index | Native Metric |
|-------|---------------|
| IndexFlatIP | Inner Product (dot product) |
| IndexFlatL2 | L2 (Euclidean) distance |
| IndexHNSWFlat | Configurable (IP or L2) |

**Note**: There is no `IndexFlatCosine` in Faiss. Cosine similarity is always achieved via normalisation + IP.

---

## Complete Pipeline

### Step 1: Raw Text Input

```python
text = "The mayor announced new housing policies yesterday"
```

A plain Python string containing semantic information we want to store and search.

---

### Step 2: Text → Embedding Vector

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)

# Result: np.ndarray with shape (384,)
# embedding = [0.234, -0.156, 0.872, 0.411, ..., 0.621]
```

**What happens internally:**
1. **Tokenisation**: Text split into subword tokens
2. **Token Embeddings**: Each token mapped to learned vector
3. **Transformer Layers**: 6-12 layers of attention mix token information
4. **Pooling**: Token vectors combined (usually mean pooling)
5. **Output**: Single vector representing entire text's meaning

**Why 384/768 dimensions?**
- The model architecture determines this
- Each dimension captures some learned semantic feature
- More dimensions = more nuance, but more storage/compute

---

### Step 3: Compute Euclidean Norm

```python
euclidean_norm: float = np.linalg.norm(embedding)
```

**What this calculates:**

$$\text{norm} = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{x_1^2 + x_2^2 + ... + x_n^2}$$

**Why:**
- This is the vector's "length" in n-dimensional space
- Needed to scale the vector to unit length
- Different texts produce different magnitude embeddings

**Example:**
```python
embedding = np.array([3.0, 4.0])
norm = np.sqrt(3² + 4²) = np.sqrt(25) = 5.0
```

---

### Step 4: Normalise to Unit Vector

```python
embedding: np.ndarray = embedding / euclidean_norm
```

**What this does:**
- Divides every component by the norm
- Results in a vector with length exactly 1.0
- Preserves direction, eliminates magnitude

**Example:**
```python
original = [3.0, 4.0]       # length = 5.0
normalised = [0.6, 0.8]     # length = √(0.36 + 0.64) = 1.0
```

**Why this matters:**
- Without normalisation: longer vectors score higher regardless of meaning
- With normalisation: only direction (semantic meaning) affects similarity scores
- Converts Faiss's dot product into cosine similarity

---

### Step 5: Reshape for Faiss

```python
embedding.reshape(1, -1)
```

**What this does:**
- Changes shape from `(384,)` to `(1, 384)`
- Faiss expects a 2D array: `(num_vectors, dimension)`
- Even for single vectors, it needs batch format

**Before:** `[0.234, -0.156, 0.872, ...]` → shape `(384,)`
**After:** `[[0.234, -0.156, 0.872, ...]]` → shape `(1, 384)`

---

### Step 6: Cast to Float32

```python
embedding.astype(np.float32)
```

**Why:**
- Faiss is implemented in C++ and requires `float32` specifically
- NumPy defaults to `float64` (double precision)
- `float32` uses half the memory with sufficient precision for similarity search
- Passing `float64` would cause errors or silent precision loss

---

### Step 7: Add to Faiss Index

```python
self.index.add(embedding.reshape(1, -1).astype(np.float32))
```

**What happens internally:**
1. Faiss copies the vector data into its internal storage
2. Assigns a sequential internal ID (0, 1, 2, 3...)
3. Vector is now searchable

**Important:** Faiss doesn't return the assigned ID - it's implicitly the next sequential position.

---

### Step 8: Track ID Mapping

```python
self.id_map.append(sqlite_id)
```

**Why this is necessary:**
- Faiss uses internal sequential IDs (0, 1, 2...)
- Your SQLite database has its own IDs (42, 17, 99...)
- This list bridges them: `id_map[faiss_position] = sqlite_id`

**Example state after 3 adds:**
```python
# Faiss internal: positions 0, 1, 2
# id_map = [42, 17, 99]
# Meaning: Faiss position 0 → SQLite ID 42
#          Faiss position 1 → SQLite ID 17
#          Faiss position 2 → SQLite ID 99
```

---

### Step 9: Search Flow

```python
def search(self, query_embedding: np.ndarray, k: int = 5):
    # 1. Normalise query (same as stored vectors)
    query = query_embedding / np.linalg.norm(query_embedding)
    
    # 2. Faiss search - returns internal indices and scores
    scores, indices = self.index.search(
        query.reshape(1, -1).astype(np.float32), 
        k
    )
    
    # 3. Map Faiss indices → SQLite IDs
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(self.id_map) and idx >= 0:
            results.append((self.id_map[idx], float(score)))
    return results
```

**Search internals:**
1. Query normalised to unit length (same space as stored vectors)
2. Faiss computes dot product against ALL stored vectors
3. Returns top-k indices with highest scores
4. Scores range from -1 to 1 (cosine similarity because both are normalised)
5. Map back to SQLite IDs for actual data retrieval

---

## Code Analysis & Improvements

### Issue 1: Zero Vector Vulnerability

**Problem:** If embedding is all zeros, division by zero occurs.

```python
# Current code - will crash or produce NaN
euclidean_norm = np.linalg.norm(embedding)  # = 0.0
embedding = embedding / euclidean_norm       # Division by zero!
```

**Fix:**
```python
euclidean_norm = np.linalg.norm(embedding)
if euclidean_norm == 0:
    raise ValueError("Cannot normalise zero vector - embedding model returned invalid output")
embedding = embedding / euclidean_norm
```

---

### Issue 2: Dimension Mismatch Not Validated

**Problem:** No check that incoming embedding matches expected dimension.

```python
# Current code - silent failure or cryptic Faiss error
def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
    # What if embedding.shape[0] != self.dimension?
```

**Fix:**
```python
def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
    if embedding.shape[0] != self.dimension:
        raise ValueError(f"Embedding dimension {embedding.shape[0]} doesn't match index dimension {self.dimension}")
```

---

### Issue 3: No Duplicate ID Check

**Problem:** Same SQLite ID can be added multiple times, wasting space and causing duplicate results.

```python
# Current code allows:
store.add(sqlite_id=42, embedding=vec1)
store.add(sqlite_id=42, embedding=vec2)  # Duplicate!
```

**Fix:**
```python
def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
    with self._lock:
        if sqlite_id in self.id_map:
            raise ValueError(f"SQLite ID {sqlite_id} already exists in index")
        # ... rest of add logic
```

Or use a `set` for O(1) lookup:
```python
self._id_set: set[int] = set()

def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
    with self._lock:
        if sqlite_id in self._id_set:
            raise ValueError(f"Duplicate ID: {sqlite_id}")
        # ... add logic ...
        self._id_set.add(sqlite_id)
```

---

### Issue 4: Search Doesn't Handle Empty Index

**Problem:** Searching an empty index may behave unexpectedly.

```python
# Current code - what happens with empty index?
scores, indices = self.index.search(query, k)
```

**Fix:**
```python
def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    with self._lock:
        if self.index.ntotal == 0:
            return []
        # ... rest of search logic
```

---

### Issue 5: search() Comments Missing

**Problem:** `search()` lacks the detailed comments that `add()` has.

**Fix:** Add consistent documentation:
```python
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
    with self._lock:
        # Normalise query to unit length (must match stored vectors)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            raise ValueError("Cannot search with zero vector")
        query = query_embedding / query_norm
        
        # ... rest
```

---

### Issue 6: Inefficient ID Lookup for Large Datasets

**Problem:** `id_map` is a list - checking for duplicates is O(n).

**Current structure:**
```python
self.id_map: list[int] = []  # O(n) lookup
```

**Better for large datasets:**
```python
self.id_map: list[int] = []           # Position → SQLite ID (keep for index mapping)
self._id_set: set[int] = set()        # O(1) duplicate checking
self._id_to_pos: dict[int, int] = {}  # SQLite ID → Position (for future removal)
```

---

### Issue 7: No Removal Support

**Problem:** Cannot remove vectors from the index.

**Note:** `IndexFlatIP` doesn't support removal. For this feature, you'd need:
- `IndexIDMap` wrapper around `IndexFlatIP`
- Or rebuild the entire index periodically
- Or use a different index type

---

### Issue 8: Sync Check After Add is Redundant

**Problem:** The sync check can never actually catch a desync.

```python
self.index.add(embedding)       # If this fails, exception thrown
self.id_map.append(sqlite_id)   # If this fails, exception thrown

# This check is unreachable if either above failed
if self.index.ntotal != len(self.id_map):
    raise ValueError("...")
```

The check is defensive but will never trigger in practice. Consider checking **before** operations instead:

```python
def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
    with self._lock:
        # Pre-check sync state (catches corruption from previous runs)
        if self.index.ntotal != len(self.id_map):
            raise RuntimeError("Index corrupted: Faiss/id_map out of sync")
        
        # ... add logic ...
```

---

### Summary of Recommended Changes

```python
def add(self, sqlite_id: int, embedding: np.ndarray) -> None:
    """Add a single embedding, linked to a SQLite summary ID."""
    with self._lock:
        # Validate sync state
        if self.index.ntotal != len(self.id_map):
            raise RuntimeError("Index corrupted: Faiss/id_map out of sync")
        
        # Validate dimension
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Dimension mismatch: got {embedding.shape[0]}, expected {self.dimension}")
        
        # Check for duplicate
        if sqlite_id in self._id_set:
            raise ValueError(f"Duplicate SQLite ID: {sqlite_id}")
        
        # Normalise to unit vector
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Cannot add zero vector")
        embedding = embedding / norm
        
        # Add to Faiss and tracking structures
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        self.id_map.append(sqlite_id)
        self._id_set.add(sqlite_id)
```
