"""Local (per-tile) sparse scoring for V-SPLADE.

Local sparse score = max over a document's local regions of dot(query, region)
-- NEVER summed. Summing would erase the locality claim: the whole point of
"local" sparse is that a single region should carry the evidence, not an
average/sum across regions.
"""

import numpy as np
import torch


def local_sparse_score_matrix(
    query_mat: torch.Tensor,
    doc_local_vecs: list[torch.Tensor],
    device: str = "cuda",
) -> np.ndarray:
    """query_mat: (Q, V) dense query sparse vectors. doc_local_vecs: list of
    (K_d, V) per-document local sparse vectors, K_d allowed to vary per
    document (V-SPLADE's per-tile vectors, see src/models/vsplade.py -- tile
    count depends on image aspect ratio/size, there is no fixed grid).
    Returns (Q, D) score matrix, score[q, d] = max_k dot(query_mat[q],
    doc_local_vecs[d][k]) -- max over regions, never summed."""
    q = query_mat.to(device)
    D = len(doc_local_vecs)
    Q = q.shape[0]
    out = torch.empty((Q, D), dtype=torch.float32)
    for d, local in enumerate(doc_local_vecs):
        if local.shape[0] == 0:
            out[:, d] = 0.0
            continue
        dots = q @ local.to(device).T  # (Q, K_d)
        out[:, d] = dots.max(dim=1).values.cpu()
    return out.numpy()
