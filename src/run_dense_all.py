"""Run the Dense-only baseline across all ViDoRe v1 subsets and write a
consolidated outputs/tables/main_results.csv row per subset (dense-only rows;
global/local sparse methods will append their own rows later).
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_vidore import VIDORE_V1_BEIR_SUBSETS, load_subset
from src.evaluation.metrics import evaluate_run, per_query_ranks
from src.models.bimodernvbert import BiModernVBertModel
from src.retrieval.dense_score import dense_score_matrix_single_vector
from src.utils.io import load_config, make_run_dir, save_config_snapshot, save_json
from src.utils.logging import get_run_logger

CONFIG_PATH = "configs/dense.yaml"


def main():
    config = load_config(CONFIG_PATH)
    model = BiModernVBertModel(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
    )

    rows = []
    for subset in VIDORE_V1_BEIR_SUBSETS:
        t0 = time.time()
        run_dir = make_run_dir(config["output_dir"], f"dense_only_{subset}")
        save_config_snapshot(run_dir, {**config, "dataset": {**config["dataset"], "subset": subset}})
        logger = get_run_logger(run_dir, name=f"vdr.{subset}")
        logger.info(f"=== subset: {subset} ===")

        try:
            data = load_subset(benchmark="vidore_v1", name=subset)
            logger.info(f"corpus={len(data.corpus_ids)} queries={len(data.query_ids)} qrels={len(data.qrels)}")

            doc_embeddings = model.encode_images(data.corpus_images, batch_size=config["batch_size"], show_progress=False)
            query_embeddings = model.encode_queries(data.query_texts, batch_size=config["batch_size"] * 2, show_progress=False)
            scores = dense_score_matrix_single_vector(query_embeddings, doc_embeddings)

            metrics = evaluate_run(scores, data.query_ids, data.corpus_ids, data.qrels)
            elapsed = time.time() - t0
            logger.info(f"metrics={metrics} elapsed={elapsed:.1f}s")

            save_json(run_dir / "metrics.json", {
                "subset": subset, "method": "dense_bimodernvbert", "model": config["model"]["name"],
                "seed": config["seed"], "num_corpus": len(data.corpus_ids),
                "num_queries": len(data.query_ids), "elapsed_seconds": elapsed, **metrics,
            })
            pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
            pd.DataFrame(per_query_ranks(scores, data.query_ids, data.corpus_ids, data.qrels)).to_csv(
                run_dir / "predictions.csv", index=False
            )

            rows.append({"method": "dense_bimodernvbert", "subset": subset, "num_corpus": len(data.corpus_ids),
                          "num_queries": len(data.query_ids), "elapsed_seconds": round(elapsed, 1), **metrics})
        except Exception as e:
            logger.exception(f"FAILED subset {subset}: {e}")
            rows.append({"method": "dense_bimodernvbert", "subset": subset, "error": str(e)})

    out_path = Path("outputs/tables/main_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out_path.exists():
        df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(df.to_string())
    print(f"Saved consolidated table to {out_path}")


if __name__ == "__main__":
    main()
