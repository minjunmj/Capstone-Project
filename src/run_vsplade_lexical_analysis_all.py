"""Lexical-heavy stratified analysis (RQ3, ColQwen-era terminology) for the
new BiModernVBERT + V-SPLADE pipeline, across all 10 ViDoRe v1 subsets.

Reuses src/data/lexical_subset.py (unchanged, from the original ColQwen-era
research) to auto-label every query lexical-heavy / mixed / non-lexical, then
computes all 5 methods (dense-only, global-sparse-only, local-sparse-only,
dense+global fusion, dense+local fusion -- see
outputs/vsplade_integration_notes.md) stratified by that label. One encoding
pass per model per subset (dense, global-sparse via encode_images, local-sparse
via encode_images_local); the query-sparse vector is shared between global and
local scoring since Li-LSR query encoding doesn't depend on doc granularity.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.lexical_subset import build_candidate_table
from src.data.load_vidore import VIDORE_V1_BEIR_SUBSETS, load_subset
from src.evaluation.metrics import evaluate_run
from src.models.bimodernvbert import BiModernVBertModel
from src.models.vsplade import VSpladeModel
from src.retrieval.dense_score import dense_score_matrix_single_vector
from src.retrieval.fusion import relative_score_fusion
from src.retrieval.sparse_score import local_sparse_score_matrix
from src.utils.io import make_run_dir, save_config_snapshot, save_json
from src.utils.logging import get_run_logger

W_DENSE = 0.4
W_SPARSE = 0.6


def main():
    dense_model = BiModernVBertModel()
    sparse_model = VSpladeModel()

    all_analysis_rows = []
    for subset in VIDORE_V1_BEIR_SUBSETS:
        t0 = time.time()
        run_dir = make_run_dir("outputs", f"vsplade_lexical_analysis_{subset}")
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

            doc_global = sparse_model.encode_images(data.corpus_images, batch_size=8, show_progress=False)
            doc_local = sparse_model.encode_images_local(data.corpus_images, batch_size=8, show_progress=False)
            query_sparse = sparse_model.encode_queries(data.query_texts, batch_size=64, show_progress=False)
            global_scores = (query_sparse @ doc_global.T).numpy()
            local_scores = local_sparse_score_matrix(query_sparse, doc_local)

            fused_global = relative_score_fusion(dense_scores, global_scores, W_DENSE, W_SPARSE)
            fused_local = relative_score_fusion(dense_scores, local_scores, W_DENSE, W_SPARSE)

            methods_scores = {
                "1_dense_only": dense_scores,
                "2_global_sparse": global_scores,
                "3_local_sparse": local_scores,
                "4_dense_plus_global": fused_global,
                "5_dense_plus_local": fused_local,
            }

            # 1) lexical candidate labeling (sec 9.2 human-reviewable file, reused unchanged)
            candidate_rows = build_candidate_table(data.query_ids, data.query_texts)
            candidates_df = pd.DataFrame(candidate_rows)
            candidates_path = Path(f"outputs/lexical_subset_candidates_vsplade_{subset}.csv")
            candidates_df.to_csv(candidates_path, index=False)
            label_by_qid = dict(zip(candidates_df["query_id"], candidates_df["auto_label"]))
            label_counts = candidates_df["auto_label"].value_counts().to_dict()
            logger.info(f"Label distribution: {label_counts}")

            all_labels = np.array([label_by_qid[q] for q in data.query_ids])

            # 2) stratified metrics per method x label
            for method_name, scores in methods_scores.items():
                for label in ["lexical-heavy", "mixed", "non-lexical", "ALL"]:
                    if label == "ALL":
                        sel_qids = data.query_ids
                        sel_scores = scores
                    else:
                        sel_idx = np.where(all_labels == label)[0]
                        if len(sel_idx) == 0:
                            continue
                        sel_qids = [data.query_ids[i] for i in sel_idx]
                        sel_scores = scores[sel_idx]
                    m = evaluate_run(sel_scores, sel_qids, data.corpus_ids, data.qrels)
                    all_analysis_rows.append({
                        "subset": subset, "method": method_name, "label": label,
                        "n_queries": len(sel_qids), **m,
                    })

            elapsed = time.time() - t0
            logger.info(f"Done subset {subset} in {elapsed:.1f}s")
            save_json(run_dir / "summary.json", {"subset": subset, "label_counts": label_counts, "elapsed_seconds": elapsed})
        except Exception as e:
            logger.exception(f"FAILED subset {subset}: {e}")
        torch.cuda.empty_cache()

    out_path = Path("outputs/tables/vsplade_lexical_analysis_all_v1.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_analysis_rows)
    df.to_csv(out_path, index=False)
    print(df.to_string())
    print(f"Saved consolidated table to {out_path}")


if __name__ == "__main__":
    main()
