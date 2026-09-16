"""Dense-only baseline: encode corpus + queries with ColQwen2.5, score with
MaxSim late interaction, evaluate nDCG/Recall/MRR, save outputs/runs/<ts>_*/.

Usage:
    python src/run_dense.py --config configs/dense.yaml --subset tabfquad
    python src/run_dense.py --config configs/dense.yaml --subset tabfquad --limit-corpus 8 --limit-queries 5
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_vidore import load_subset
from src.evaluation.metrics import evaluate_run, per_query_ranks
from src.models.bimodernvbert import BiModernVBertModel
from src.retrieval.dense_score import dense_score_matrix_single_vector
from src.utils.io import load_config, make_run_dir, save_config_snapshot, save_json
from src.utils.logging import get_run_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", required=True, help="ViDoRe subset short name, e.g. tabfquad")
    parser.add_argument("--limit-corpus", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset"]["subset"] = args.subset
    if args.limit_corpus is not None:
        config["dataset"]["limit_corpus"] = args.limit_corpus
    if args.limit_queries is not None:
        config["dataset"]["limit_queries"] = args.limit_queries

    run_dir = make_run_dir(config["output_dir"], f"{config['experiment_name']}_{args.subset}")
    save_config_snapshot(run_dir, config)
    logger = get_run_logger(run_dir)

    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Config: {config}")

    t0 = time.time()
    logger.info(f"Loading dataset: benchmark={config['dataset']['benchmark']} subset={args.subset}")
    data = load_subset(
        benchmark=config["dataset"]["benchmark"],
        name=args.subset,
        limit_corpus=config["dataset"].get("limit_corpus"),
        limit_queries=config["dataset"].get("limit_queries"),
    )
    logger.info(f"Corpus size: {len(data.corpus_ids)}, Query count: {len(data.query_ids)}, "
                f"queries with qrels: {len(data.qrels)}")

    logger.info(f"Loading model: {config['model']['name']}")
    model = BiModernVBertModel(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    doc_embeddings = model.encode_images(data.corpus_images, batch_size=config["batch_size"])
    logger.info(f"Encoded {doc_embeddings.shape[0]} corpus images in {time.time() - t1:.1f}s")

    t2 = time.time()
    query_embeddings = model.encode_queries(data.query_texts, batch_size=config["batch_size"] * 2)
    logger.info(f"Encoded {query_embeddings.shape[0]} queries in {time.time() - t2:.1f}s")

    t3 = time.time()
    scores = dense_score_matrix_single_vector(query_embeddings, doc_embeddings)
    logger.info(f"Scored {scores.shape} matrix in {time.time() - t3:.1f}s")

    metrics = evaluate_run(scores, data.query_ids, data.corpus_ids, data.qrels)
    logger.info(f"Metrics: {metrics}")

    save_json(run_dir / "metrics.json", {
        "subset": args.subset,
        "method": config["method"],
        "model": config["model"]["name"],
        "seed": config["seed"],
        "num_corpus": len(data.corpus_ids),
        "num_queries": len(data.query_ids),
        "num_queries_evaluated": len(data.qrels),
        "elapsed_seconds": time.time() - t0,
        **metrics,
    })
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)

    predictions = per_query_ranks(scores, data.query_ids, data.corpus_ids, data.qrels)
    pd.DataFrame(predictions).to_csv(run_dir / "predictions.csv", index=False)

    failure_cases = [p for p in predictions if p["num_relevant"] > 0 and not p["top1_correct"]]
    pd.DataFrame(failure_cases).to_csv(run_dir / "failure_cases.csv", index=False)

    logger.info(f"Done in {time.time() - t0:.1f}s total. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
