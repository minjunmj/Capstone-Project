"""Dense (BiModernVBERT) vs. Global Sparse (V-SPLADE, page-level) vs. their
fusion, using the V-SPLADE paper's own fixed Relative Score Fusion weights
(arXiv 2605.30917 sec 5.3: w_sparse=0.6, w_dense=0.4, per-query min-max
normalization) -- no lambda sweep, these weights are the paper's, not tuned
here on any split.

Usage:
    python src/run_vsplade_dense_sparse.py --subset tatdqa
    python src/run_vsplade_dense_sparse.py --subset tatdqa --limit-corpus 20 --limit-queries 30
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_vidore import load_subset
from src.evaluation.metrics import evaluate_run
from src.models.bimodernvbert import BiModernVBertModel
from src.models.vsplade import VSpladeModel
from src.retrieval.dense_score import dense_score_matrix_single_vector
from src.retrieval.fusion import relative_score_fusion
from src.utils.io import make_run_dir, save_config_snapshot, save_json
from src.utils.logging import get_run_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--limit-corpus", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--w-dense", type=float, default=0.4)
    parser.add_argument("--w-sparse", type=float, default=0.6)
    args = parser.parse_args()

    config = {
        "dense_model": "ModernVBERT/bimodernvbert",
        "sparse_model": "naver/v-splade-quality",
        "fusion": "relative_score_fusion (paper weights, not tuned here)",
        "w_dense": args.w_dense,
        "w_sparse": args.w_sparse,
        "subset": args.subset,
    }
    run_dir = make_run_dir("outputs", f"vsplade_dense_sparse_{args.subset}")
    save_config_snapshot(run_dir, config)
    logger = get_run_logger(run_dir)
    logger.info(f"Run dir: {run_dir}")

    t0 = time.time()
    data = load_subset("vidore_v1", args.subset, args.limit_corpus, args.limit_queries)
    logger.info(f"corpus={len(data.corpus_ids)} queries={len(data.query_ids)} qrels={len(data.qrels)}")

    t1 = time.time()
    dense_model = BiModernVBertModel()
    doc_dense = dense_model.encode_images(data.corpus_images, batch_size=16)
    query_dense = dense_model.encode_queries(data.query_texts, batch_size=32)
    dense_scores = dense_score_matrix_single_vector(query_dense, doc_dense)
    logger.info(f"Dense (BiModernVBERT) encoded + scored in {time.time() - t1:.1f}s")
    del dense_model
    torch.cuda.empty_cache()

    t2 = time.time()
    sparse_model = VSpladeModel()
    doc_sparse = sparse_model.encode_images(data.corpus_images, batch_size=8)
    query_sparse = sparse_model.encode_queries(data.query_texts, batch_size=64)
    sparse_scores = (query_sparse @ doc_sparse.T).numpy()
    logger.info(f"Sparse (V-SPLADE, global/page-level) encoded + scored in {time.time() - t2:.1f}s")
    del sparse_model
    torch.cuda.empty_cache()

    metrics_dense = evaluate_run(dense_scores, data.query_ids, data.corpus_ids, data.qrels)
    metrics_sparse = evaluate_run(sparse_scores, data.query_ids, data.corpus_ids, data.qrels)
    fused_scores = relative_score_fusion(dense_scores, sparse_scores, args.w_dense, args.w_sparse)
    metrics_fused = evaluate_run(fused_scores, data.query_ids, data.corpus_ids, data.qrels)

    logger.info(f"[dense-only, BiModernVBERT]: {metrics_dense}")
    logger.info(f"[sparse-only, V-SPLADE global]: {metrics_sparse}")
    logger.info(f"[RSF fused, w_dense={args.w_dense} w_sparse={args.w_sparse}]: {metrics_fused}")

    elapsed = time.time() - t0
    save_json(run_dir / "metrics.json", {
        "subset": args.subset, "num_corpus": len(data.corpus_ids), "num_queries": len(data.query_ids),
        "w_dense": args.w_dense, "w_sparse": args.w_sparse, "elapsed_seconds": elapsed,
        "dense_only": metrics_dense, "sparse_only_global": metrics_sparse, "rsf_fused": metrics_fused,
    })
    row = pd.DataFrame([{
        "subset": args.subset, "num_corpus": len(data.corpus_ids), "num_queries": len(data.query_ids),
        "elapsed_seconds": round(elapsed, 1),
        **{f"dense_only_{k}": v for k, v in metrics_dense.items()},
        **{f"sparse_only_global_{k}": v for k, v in metrics_sparse.items()},
        **{f"rsf_fused_{k}": v for k, v in metrics_fused.items()},
    }])
    out_path = Path("outputs/tables/vsplade_dense_sparse.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        row = pd.concat([pd.read_csv(out_path), row], ignore_index=True)
    row.to_csv(out_path, index=False)
    logger.info(f"Done in {elapsed:.1f}s. Outputs in {run_dir}, table appended to {out_path}")


if __name__ == "__main__":
    main()
