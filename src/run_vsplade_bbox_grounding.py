"""RQ4 (Phase 2 version): does V-SPLADE's highest-scoring local (per-tile)
region overlap with the actual evidence bounding box (ViDoRe v3)?

Mirrors src/run_bbox_grounding.py (Phase 1, ColQwen fixed-grid) but for
V-SPLADE's Idefics3-tile-based local sparse (src/models/vsplade.py). Unlike
the fixed grid, tile pixel bboxes must be derived from Idefics3's actual
resize+split algorithm (transformers/models/idefics3/image_processing_idefics3.py):
  1. resize the image so its longest edge = processor.size['longest_edge'] (2048),
     preserving aspect ratio (get_resize_output_image_size)
  2. split the resized image into a rows x cols grid of <=512x512 tiles,
     row-major order (split_image) -- verified empirically to match the tile
     order in pixel_values / the local sparse vector order (see
     outputs/vsplade_integration_notes.md)
  3. append one more "global" tile: the whole image downscaled to 512x512
     (always last). This is EXCLUDED from grounding candidates below (its
     bbox is the whole page, which would trivially inflate hit-rate for any
     GT box) -- it's still used elsewhere (retrieval scoring) as one of the K
     local vectors, just not here.
  4. tile bboxes are computed in resized-image pixel space, then scaled back
     to the ORIGINAL image's pixel space (to match how ViDoRe v3 annotates
     bounding_boxes).
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.data.load_vidore import VIDORE_V3_SUBSETS
from src.models.vsplade import VSpladeModel
from src.utils.io import make_run_dir, save_config_snapshot, save_json
from src.utils.logging import get_run_logger


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def idefics3_tile_pixel_bboxes(image, image_processor):
    """Returns a list of (x1, y1, x2, y2) in ORIGINAL image pixel space, one
    per split tile, row-major order -- matching the order of the local
    sparse vectors from VSpladeModel.encode_images_local (the global/thumbnail
    tile is NOT included here, see module docstring)."""
    from transformers.models.idefics3.image_processing_idefics3 import get_resize_output_image_size

    orig_w, orig_h = image.size
    arr = np.array(image.convert("RGB"))
    resize_h, resize_w = get_resize_output_image_size(arr, resolution_max_side=image_processor.size["longest_edge"])

    max_side = image_processor.max_image_size["longest_edge"]
    if resize_h <= max_side and resize_w <= max_side:
        return []  # no splitting happens (image already small) -- only the global tile exists, no local regions

    num_splits_h = math.ceil(resize_h / max_side)
    num_splits_w = math.ceil(resize_w / max_side)
    optimal_h = math.ceil(resize_h / num_splits_h)
    optimal_w = math.ceil(resize_w / num_splits_w)

    scale_x = orig_w / resize_w
    scale_y = orig_h / resize_h

    boxes = []
    for r in range(num_splits_h):
        for c in range(num_splits_w):
            x1 = c * optimal_w
            y1 = r * optimal_h
            x2 = min(x1 + optimal_w, resize_w)
            y2 = min(y1 + optimal_h, resize_h)
            boxes.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))
    return boxes


def sample_bbox_triples(hf_id, n_samples, seed):
    qrels_ds = load_dataset(hf_id, "qrels")
    qrels_ds = qrels_ds[list(qrels_ds.keys())[0]]
    candidates = [r for r in qrels_ds if r["score"] > 0 and r["bounding_boxes"]]
    rng = random.Random(seed)
    return rng.sample(candidates, min(n_samples, len(candidates)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="finance_en", choices=list(VIDORE_V3_SUBSETS))
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = make_run_dir("outputs", f"vsplade_bbox_grounding_{args.subset}")
    save_config_snapshot(run_dir, vars(args))
    logger = get_run_logger(run_dir)
    logger.info(f"Run dir: {run_dir}")

    hf_id = VIDORE_V3_SUBSETS[args.subset]
    t0 = time.time()
    triples = sample_bbox_triples(hf_id, args.n_samples, args.seed)
    logger.info(f"Sampled {len(triples)} (query, corpus, bbox) triples with score>0 from {hf_id}")

    query_ids_needed = sorted({t["query_id"] for t in triples})
    corpus_ids_needed = sorted({t["corpus_id"] for t in triples})
    logger.info(f"Unique queries needed: {len(query_ids_needed)}, unique docs needed: {len(corpus_ids_needed)}")

    queries_ds = load_dataset(hf_id, "queries")
    queries_ds = queries_ds[list(queries_ds.keys())[0]]
    qid_set = set(query_ids_needed)
    query_text_by_id = {r["query_id"]: r["query"] for r in queries_ds if r["query_id"] in qid_set}

    corpus_ds = load_dataset(hf_id, "corpus")
    corpus_ds = corpus_ds[list(corpus_ds.keys())[0]]
    cid_set = set(corpus_ids_needed)
    corpus_filtered = corpus_ds.filter(lambda r: r["corpus_id"] in cid_set)
    image_by_cid = {r["corpus_id"]: r["image"] for r in corpus_filtered}
    logger.info(f"Loaded {len(image_by_cid)} doc images, {len(query_text_by_id)} query texts "
                f"in {time.time() - t0:.1f}s (no full-corpus encoding needed)")

    vs = VSpladeModel()
    image_processor = vs.processor.image_processor

    doc_local_sparse, doc_region_boxes = {}, {}
    for i, (cid, image) in enumerate(image_by_cid.items()):
        image = image.convert("RGB")
        local_vecs = vs.encode_images_local([image], batch_size=1, show_progress=False)[0]  # (K, V), K = splits + 1 global
        boxes = idefics3_tile_pixel_bboxes(image, image_processor)
        n_split = len(boxes)
        # drop the global/thumbnail tile (always last) from grounding candidates -- see module docstring
        if n_split > 0 and local_vecs.shape[0] == n_split + 1:
            local_vecs = local_vecs[:n_split]
        else:
            local_vecs = local_vecs[:0]  # image wasn't split (too small) -- no local regions to ground against
            boxes = []
        doc_local_sparse[cid] = local_vecs
        doc_region_boxes[cid] = boxes
        if (i + 1) % 20 == 0:
            logger.info(f"  encoded {i + 1}/{len(image_by_cid)} docs")

    query_sparse_by_id = {}
    for qid, text in query_text_by_id.items():
        query_sparse_by_id[qid] = vs.encode_queries([text], batch_size=1, show_progress=False)[0]

    logger.info(f"Encoded all needed docs+queries in {time.time() - t0:.1f}s total")

    rng = np.random.RandomState(args.seed)
    rows = []
    skipped_no_regions = 0
    for t in triples:
        qid, cid = t["query_id"], t["corpus_id"]
        gt_boxes = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in t["bounding_boxes"]]

        local_vecs = doc_local_sparse[cid]
        region_boxes = doc_region_boxes[cid]
        K_actual = len(region_boxes)
        if K_actual == 0:
            skipped_no_regions += 1
            continue

        q_vec = query_sparse_by_id[qid]
        region_scores = (local_vecs @ q_vec).numpy()  # (K,)

        ranked = np.argsort(-region_scores)
        top1_box = region_boxes[ranked[0]]
        top1_iou = max(iou(top1_box, gt) for gt in gt_boxes)
        top1_hit = top1_iou > 0.0

        top_k_hit = any(
            max(iou(region_boxes[r], gt) for gt in gt_boxes) > 0.0
            for r in ranked[: min(3, K_actual)]
        )

        random_region = region_boxes[rng.randint(K_actual)]
        random_iou = max(iou(random_region, gt) for gt in gt_boxes)

        rows.append({
            "query_id": qid, "corpus_id": cid, "content_type": ";".join(t["content_type"] or []),
            "num_tiles": K_actual, "top1_iou": top1_iou, "top1_hit": top1_hit,
            "top3_hit": top_k_hit, "random_iou": random_iou,
        })

    df = pd.DataFrame(rows)
    out_path = Path(f"outputs/tables/vsplade_bbox_grounding_{args.subset}.csv")
    df.to_csv(out_path, index=False)

    summary = {
        "subset": args.subset, "n_triples": len(df), "n_skipped_no_regions": skipped_no_regions,
        "region_hit@1": float(df["top1_hit"].mean()) if len(df) else None,
        "region_hit@3": float(df["top3_hit"].mean()) if len(df) else None,
        "mean_top1_iou": float(df["top1_iou"].mean()) if len(df) else None,
        "mean_random_iou": float(df["random_iou"].mean()) if len(df) else None,
        "elapsed_seconds": time.time() - t0,
    }
    logger.info(f"Summary: {summary}")
    save_json(run_dir / "summary.json", summary)
    logger.info(f"Saved per-query results to {out_path}")


if __name__ == "__main__":
    main()
