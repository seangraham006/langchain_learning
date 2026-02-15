# Dot Product vs Cosine Similarity

This guide explains what each metric measures, when to use each one, and trade-offs in retrieval systems.

## Quick Summary

- **Dot Product** measures directional alignment **and** is affected by vector magnitude.
- **Cosine Similarity** measures directional alignment only (angle between vectors), after accounting for magnitude.

If vectors are unit-normalized (length = 1), dot product and cosine similarity are numerically the same.

---

## Definitions

### Dot Product

For vectors $a$ and $b$:

$$
\text{dot}(a,b) = \sum_{i=1}^{n} a_i b_i
$$

Interpretation:
- Larger positive value → vectors point in similar directions (and/or have larger magnitudes)
- Near zero → mostly orthogonal / unrelated
- Negative value → opposite directions

### Cosine Similarity

$$
\cos(a,b) = \frac{a \cdot b}{\|a\|\|b\|}
$$

Interpretation:
- 1 → same direction
- 0 → orthogonal
- -1 → opposite direction

Cosine similarity removes magnitude effects by dividing by both vector norms.

---

## Pros and Cons

### Dot Product

| Pros | Cons |
|------|------|
| Fast and simple to compute | Sensitive to vector magnitude |
| Works well when magnitude carries signal (e.g., confidence, frequency, popularity) | Longer vectors can dominate rankings even if direction is less aligned |
| Commonly optimized in ANN/vector databases | Harder to compare scores across datasets with varying norm distributions |
| Natural choice for some model training objectives | Can introduce bias if norms are artifacts, not meaningful signal |

### Cosine Similarity

| Pros | Cons |
|------|------|
| Invariant to vector scale; compares semantic direction | Requires normalization (explicitly or implicitly) |
| Often more stable for text embedding retrieval | Slight extra preprocessing cost |
| Scores are bounded in [-1, 1], easier to reason about | Ignores magnitude that might contain useful signal |
| Reduces norm-related ranking bias | If magnitude is meaningful, cosine may discard valuable information |

---

## Practical Guidance

### Use Dot Product when

- Embedding magnitudes are intentionally meaningful in your model/system.
- You want magnitude to influence ranking.
- Your pipeline and index are already tuned for inner product retrieval.

### Use Cosine Similarity when

- You primarily care about semantic alignment of direction.
- Vector norms vary due to artifacts (text length, preprocessing, encoder quirks).
- You want more consistent ranking behavior across varied inputs.

---

## Important Equivalence

If vectors are normalized to unit length:

$$
\|a\| = \|b\| = 1 \Rightarrow \cos(a,b) = a \cdot b
$$

This is why many systems (including Faiss `IndexFlatIP`) can serve cosine similarity by normalizing vectors first and then using inner product search.

---

## Example

Let:
- $a = [1, 1]$
- $b = [2, 2]$
- $c = [2, 0]$

Dot products:
- $a \cdot b = 4$
- $a \cdot c = 2$

Dot product ranks $b$ above $c$ strongly, partly due to larger magnitude.

Cosine similarities:
- $\cos(a,b) = 1.0$ (same direction)
- $\cos(a,c) \approx 0.707$

Cosine still ranks $b$ above $c$, but now based on direction only, not scale inflation.

---

## Recommendation for Typical Text Retrieval

For most semantic text retrieval/RAG pipelines:

1. L2-normalize embeddings.
2. Use inner product search in the vector index.
3. Treat resulting scores as cosine similarity.

This usually provides robust, interpretable rankings while preserving search performance.
