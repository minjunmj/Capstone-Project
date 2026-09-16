"""Standalone sanity checks for metrics.py (run: python src/evaluation/test_metrics.py)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.metrics import dcg_at_k, evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k


def test_dcg_perfect_order():
    # single relevant doc at rank 1 -> dcg = 1/log2(2) = 1
    assert abs(dcg_at_k([1, 0, 0], 3) - 1.0) < 1e-9


def test_ndcg_perfect_vs_worst():
    assert ndcg_at_k([1, 0, 0], 3) == 1.0  # relevant doc already at top -> ideal
    assert abs(ndcg_at_k([0, 0, 1], 3) - (1 / math.log2(4)) / (1 / math.log2(2))) < 1e-9


def test_recall_at_k():
    assert recall_at_k([1, 0, 1, 0], 2, num_relevant=2) == 0.5
    assert recall_at_k([1, 1, 0, 0], 2, num_relevant=2) == 1.0
    assert recall_at_k([0, 0, 0], 3, num_relevant=0) == 0.0  # no relevant docs -> 0, not div-by-zero


def test_mrr_at_k():
    assert mrr_at_k([0, 1, 0], 3) == 0.5
    assert mrr_at_k([0, 0, 0], 3) == 0.0


def test_evaluate_run_matches_manual():
    # 2 queries, 3 docs. q0's relevant doc is doc1 (ranked 1st by score); q1's relevant doc is doc2 (ranked last).
    scores = np.array([
        [0.1, 0.9, 0.2],   # ranking: doc1, doc2, doc0
        [0.9, 0.5, 0.1],   # ranking: doc0, doc1, doc2 -- relevant doc2 is last
    ])
    query_ids, corpus_ids = ["q0", "q1"], ["d0", "d1", "d2"]
    qrels = {"q0": {"d1": 1}, "q1": {"d2": 1}}
    metrics = evaluate_run(scores, query_ids, corpus_ids, qrels, ks=(1, 3))
    assert metrics["recall@1"] == 0.5  # q0 hits at rank1, q1 doesn't
    assert metrics["recall@3"] == 1.0  # both eventually recalled within top-3
    assert abs(metrics["mrr@3"] - (1.0 + 1 / 3) / 2) < 1e-9


def test_evaluate_run_skips_queries_without_qrels():
    scores = np.array([[0.5, 0.1]])
    metrics = evaluate_run(scores, ["q0"], ["d0", "d1"], qrels={}, ks=(1,))
    assert metrics == {}  # no judged queries -> nothing to average, not a crash


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
