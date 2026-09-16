"""Dense/sparse score fusion via Relative Score Fusion (RSF)."""

import numpy as np


def _minmax_normalize_per_query(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize each row (query) independently to [0, 1]."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / (hi - lo + 1e-8)


def relative_score_fusion(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    w_dense: float = 0.4,
    w_sparse: float = 0.6,
) -> np.ndarray:
    """Relative Score Fusion (RSF), as used in the V-SPLADE paper (arXiv
    2605.30917, sec 5.3 "Complementarity with dense retrieval"): per-query
    min-max normalization of each retriever's scores to [0, 1], then a
    weighted sum with weights that sum to 1. The paper's reported weights for
    V-SPLADE + BiModernVBERT are w_sparse=0.6, w_dense=0.4 -- fixed by the
    paper, not tuned here on any split."""
    return w_dense * _minmax_normalize_per_query(dense_scores) + w_sparse * _minmax_normalize_per_query(sparse_scores)
