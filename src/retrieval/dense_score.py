"""Dense scoring for the BiModernVBERT bi-encoder (single L2-normalized
vector per query/document -- not multi-vector late-interaction)."""

import numpy as np
import torch


def dense_score_matrix_single_vector(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
) -> np.ndarray:
    """query/doc_embeddings: (N, dim), already L2-normalized by the model ->
    dot product == cosine similarity. Returns (n_queries, n_docs)."""
    return (query_embeddings @ doc_embeddings.T).numpy()
