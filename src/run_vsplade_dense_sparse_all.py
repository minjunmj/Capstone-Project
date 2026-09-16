"""Dense (BiModernVBERT) vs. Global Sparse (V-SPLADE) vs. RSF Fusion across
all ViDoRe v1 subsets. Loads both models once, then iterates over subsets
(mirrors src/run_dense_all.py's pattern). See src/run_vsplade_dense_sparse.py
for the single-subset version and the fusion-weight rationale.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_vidore import VIDORE_V1_BEIR_SUBSETS, load_subset
from src.evaluation.metrics import evaluate_run
from src.models.bimodernvbert import BiModernVBertModel
from src.models.vsplade import VSpladeModel
from src.retrieval.dense_score import dense_score_matrix_single_vector
from src.retrieval.fusion import relative_score_fusion
from src.utils.io import make_run_dir, save_config_snapshot, save_json
from src.utils.logging import get_run_logger

W_DENSE = 0.4
W_SPARSE = 0.6


def main():
    dense_model = BiModernVBertModel()
    sparse_model = VSpladeModel()

    rows = []
    for subset in VIDORE_V1_BEIR_SUBSETS:
        t0 = time.time()
        run_dir = make_run_dir("outputs", f"vsplade_dense_sparse_{subset}")
        save_config_snapshot(run_dir, {
            "dense_model": "ModernVBERT/bimodernvbert", "sparse_model": "naver/v-splade-quality",
            "w_dense": W_DENSE, "w_sparse": W_SPARSE, "subset": subset,
        })
        logger = get_run_logger(run_dir, name=f"vdr.{subset}")
        logger.info(f"=== subset: {subset} ===")

        try:
            data = load_subset(benchmark="vidore_v1", name=subset)
            logger.info(f"corpus={len(data.corpus_ids)} queries={len(data.query_ids)} qrels={len(data.qrels)}")

            doc_dense = dense_model.encode_images(data.corpus_images, batch_size=16, show_progress=False)
            query_dense = dense_model.encode_queries(data.query_texts, batch_size=32, show_progress=False)
            dense_scores = dense_score_matrix_single_vector(query_dense, doc_dense)

            doc_sparse = sparse_model.encode_images(data.corpus_images, batch_size=8, show_progress=False)
            query_sparse = sparse_model.encode_queries(data.query_texts, batch_size=64, show_progress=False)
            sparse_scores = (query_sparse @ doc_sparse.T).numpy()

            metrics_dense = evaluate_run(dense_scores, data.query_ids, data.corpus_ids, data.qrels)
            metrics_sparse = evaluate_run(sparse_scores, data.query_ids, data.corpus_ids, data.qrels)
            fused_scores = relative_score_fusion(dense_scores, sparse_scores, W_DENSE, W_SPARSE)
            metrics_fused = evaluate_run(fused_scores, data.query_ids, data.corpus_ids, data.qrels)

            elapsed = time.time() - t0
            logger.info(f"dense-only={metrics_dense}")
            logger.info(f"sparse-only(global)={metrics_sparse}")
            logger.info(f"rsf-fused={metrics_fused}")
            logger.info(f"elapsed={elapsed:.1f}s")

            save_json(run_dir / "metrics.json", {
                "subset": subset, "num_corpus": len(data.corpus_ids), "num_queries": len(data.query_ids),
                "w_dense": W_DENSE, "w_sparse": W_SPARSE, "elapsed_seconds": elapsed,
                "dense_only": metrics_dense, "sparse_only_global": metrics_sparse, "rsf_fused": metrics_fused,
            })

            rows.append({
                "subset": subset, "num_corpus": len(data.corpus_ids), "num_queries": len(data.query_ids),
                "elapsed_seconds": round(elapsed, 1),
                **{f"dense_only_{k}": v for k, v in metrics_dense.items()},
                **{f"sparse_only_global_{k}": v for k, v in metrics_sparse.items()},
                **{f"rsf_fused_{k}": v for k, v in metrics_fused.items()},
            })
        except Exception as e:
            logger.exception(f"FAILED subset {subset}: {e}")
            rows.append({"subset": subset, "error": str(e)})
        torch.cuda.empty_cache()

    out_path = Path("outputs/tables/vsplade_dense_sparse_all_v1.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.to_string())
    print(f"Saved consolidated table to {out_path}")


if __name__ == "__main__":
    main()
