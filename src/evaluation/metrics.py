"""Small, independently-testable retrieval metrics: nDCG@k, Recall@k, MRR@k.

All functions take a single ranked list of relevance labels (int, 0 = not
relevant) for one query — standard IR metric definitions, no framework
dependency. `evaluate_run` aggregates over a whole query set given a score
matrix + qrels.
"""

import math
from typing import Any

import numpy as np


def dcg_at_k(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # i is 0-indexed -> position i+1
    return dcg


def ndcg_at_k(relevances: list[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(relevances: list[int], k: int, num_relevant: int) -> float:
    if num_relevant == 0:
        return 0.0
    hits = sum(1 for rel in relevances[:k] if rel > 0)
    return hits / num_relevant


def mrr_at_k(relevances: list[int], k: int) -> float:
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(
    scores: np.ndarray,
    query_ids: list[Any],
    corpus_ids: list[Any],
    qrels: dict[Any, dict[Any, int]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """scores: (n_queries, n_corpus) matrix aligned with query_ids/corpus_ids order."""
    corpus_index = {cid: i for i, cid in enumerate(corpus_ids)}
    per_metric: dict[str, list[float]] = {}

    for qi, qid in enumerate(query_ids):
        rel_map = qrels.get(qid, {})
        if not rel_map:
            continue  # no judgments for this query (e.g. truncated corpus in a smoke run)
        order = np.argsort(-scores[qi])
        relevances = [rel_map.get(corpus_ids[ci], 0) for ci in order]
        num_relevant = sum(1 for v in rel_map.values() if v > 0)

        for k in ks:
            per_metric.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(relevances, k))
            per_metric.setdefault(f"recall@{k}", []).append(recall_at_k(relevances, k, num_relevant))
            per_metric.setdefault(f"mrr@{k}", []).append(mrr_at_k(relevances, k))

    return {name: float(np.mean(vals)) if vals else 0.0 for name, vals in per_metric.items()}


def per_query_ranks(
    scores: np.ndarray,
    query_ids: list[Any],
    corpus_ids: list[Any],
    qrels: dict[Any, dict[Any, int]],
) -> list[dict[str, Any]]:
    """Per-query top-ranked corpus id + whether it's correct, for predictions.csv / error analysis."""
    rows = []
    for qi, qid in enumerate(query_ids):
        rel_map = qrels.get(qid, {})
        order = np.argsort(-scores[qi])
        top1_cid = corpus_ids[order[0]]
        rows.append({
            "query_id": qid,
            "top1_corpus_id": top1_cid,
            "top1_score": float(scores[qi, order[0]]),
            "top1_correct": int(rel_map.get(top1_cid, 0) > 0),
            "num_relevant": sum(1 for v in rel_map.values() if v > 0),
        })
    return rows
