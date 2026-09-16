"""Local-objective LoRA fine-tuning proof-of-concept (see
outputs/vsplade_integration_notes.md "왜 Phase 2가 Phase 1보다..." section for
the hypothesis this tests).

Trains TWO short LoRA fine-tunes from the SAME base checkpoint
(ModernVBERT/modernvbert), SAME data, SAME step count, SAME seed -- the only
difference is the training objective:
  - Model A ("global"):  the released recipe's objective (compute_logits,
    supervises the single whole-sequence max-pooled passage vector)
  - Model B ("local"):   compute_logits_local (added to train/models/model.py),
    supervises a MaxSim-over-tiles objective matching eval-time local scoring

Then evaluates BOTH models in BOTH scoring modes (global / local) on a ViDoRe
v1 subset, to see whether Model B's local-mode score improves relative to
Model A's -- i.e. whether the released checkpoint's poor local-mode
performance (see outputs/vsplade_integration_notes.md sec 8/9) is because it
was only ever trained with the global objective.

This is NOT a reproduction of V-SPLADE's full training recipe (no captions,
no FLOPS regularization, no hard negatives, no multi-GPU, ~500 steps instead
of a full run) -- it is a small, directional test of one specific hypothesis.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/workspace/third_party/v-splade/train")

from colpali_engine.models import BiModernVBertProcessor
from dataset import ModernVBertRetrieverCollator, VisualDataset
from models import build_model

from src.data.load_vidore import load_v1_beir_subset
from src.evaluation.metrics import evaluate_run
from src.utils.io import make_run_dir, save_json
from src.utils.logging import get_run_logger

BASE_MODEL = "ModernVBERT/modernvbert"
TRAIN_DATA_PATH = "/workspace/data/vsplade_train_slice"
BATCH_SIZE = 8
NUM_STEPS = 500
LEARNING_RATE = 1e-5
SEED = 42
EVAL_SUBSET = "tatdqa"


def build_and_train(local_supervision: bool, logger, run_dir):
    tag = "local" if local_supervision else "global"
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = build_model(
        mode="from_scratch", encoder_type="vbert", head_type="sparse", query_encoder_type="bow",
        model_name=BASE_MODEL, lm_head_model=BASE_MODEL,
        temperature=1.0, cap_weight=0.0, reg_weight_p=0.0, reg_weight_cap=0.0,
        splade_pooling="max", lm_head_lora_r=32, encoder_lora_r=32,
        local_supervision=local_supervision,
    ).to("cuda")
    model.train()
    model.gradient_checkpointing_enable()

    processor = BiModernVBertProcessor.from_pretrained(BASE_MODEL)
    ds = VisualDataset(dataset_path=TRAIN_DATA_PATH, caption_column="caption")
    collator = ModernVBertRetrieverCollator(processor=processor, num_hard_negatives=0, caption_field=None)
    # Fixed generator seed -> both runs see batches in the same order for a fair comparison.
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator,
                         generator=gen, drop_last=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=LEARNING_RATE, weight_decay=0.01)

    logger.info(f"[{tag}] training {NUM_STEPS} steps, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    t0 = time.time()
    step = 0
    loss_history = []
    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break
            gpu_batch = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                v = v.to("cuda")
                if k == "passage_pixel_values":
                    v = v.to(torch.bfloat16)
                gpu_batch[k] = v
            outputs = model(**gpu_batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append({"step": step, "loss": outputs.loss.item(), "rank_loss": outputs.rank_loss.item()})
            if step % 25 == 0:
                logger.info(f"[{tag}] step {step}: loss={outputs.loss.item():.4f} "
                            f"rank_loss={outputs.rank_loss.item():.4f} t={time.time()-t0:.1f}s")
            step += 1

    elapsed = time.time() - t0
    logger.info(f"[{tag}] training done in {elapsed:.1f}s ({elapsed/NUM_STEPS:.2f}s/step)")
    pd.DataFrame(loss_history).to_csv(run_dir / f"loss_history_{tag}.csv", index=False)
    model.eval()
    return model, processor


@torch.no_grad()
def evaluate_model(model, processor, tag, data, logger):
    """Encodes the eval subset with THIS model and scores in both global and
    local mode, returning a dict of {mode: metrics}."""
    image_token_id = model.encoder.image_token_id
    scale = model.hidden_size ** -0.25

    # ---- encode corpus: global (model.encode_passage) + local (manual, mirrors
    # _apply_sparse_head_local but without gradients / batch-agnostic padding
    # -- reuse the model's own method directly since it's already eval-safe). ----
    doc_global, doc_local = [], []
    batch_size = 8
    for start in range(0, len(data.corpus_images), batch_size):
        chunk = [im.convert("RGB") for im in data.corpus_images[start:start + batch_size]]
        p = processor.process_images(chunk)
        p = {k: (v.to("cuda").to(torch.bfloat16) if k == "pixel_values" else v.to("cuda"))
             for k, v in p.items()}
        w_global = model.encode_passage(**p)  # (B, V) -- public API, global pooled
        doc_global.append(w_global.float().cpu())

        w_local, tile_mask = model._encode_passage_sparse_local(**p)
        for i in range(w_local.shape[0]):
            real = tile_mask[i]
            doc_local.append(w_local[i][real].float().cpu() if real.any() else w_local[i][:0].float().cpu())
    doc_global_mat = torch.cat(doc_global, dim=0)

    # ---- encode queries (BOW, no transformer forward). ----
    query_vecs = []
    for start in range(0, len(data.query_texts), 64):
        chunk = data.query_texts[start:start + 64]
        q = processor.process_queries(chunk)
        q = {k: v.to("cuda") for k, v in q.items()}
        qv = model.encode_query(q["input_ids"], q["attention_mask"])
        query_vecs.append(qv.float().cpu())
    query_mat = torch.cat(query_vecs)

    global_scores = (query_mat @ doc_global_mat.T).numpy()

    D = len(doc_local)
    Q = query_mat.shape[0]
    local_scores = np.zeros((Q, D), dtype=np.float32)
    q_gpu = query_mat.to("cuda")
    for d, local in enumerate(doc_local):
        if local.shape[0] == 0:
            continue
        dots = q_gpu @ local.to("cuda").T
        local_scores[:, d] = dots.max(dim=1).values.cpu().numpy()

    m_global = evaluate_run(global_scores, data.query_ids, data.corpus_ids, data.qrels)
    m_local = evaluate_run(local_scores, data.query_ids, data.corpus_ids, data.qrels)
    logger.info(f"[{tag}] eval global-mode: {m_global}")
    logger.info(f"[{tag}] eval local-mode:  {m_local}")
    return {"global_mode": m_global, "local_mode": m_local}


def main():
    run_dir = make_run_dir("outputs", "vsplade_local_finetune_experiment")
    logger = get_run_logger(run_dir)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"base_model={BASE_MODEL} steps={NUM_STEPS} batch_size={BATCH_SIZE} "
                f"lr={LEARNING_RATE} seed={SEED} eval_subset={EVAL_SUBSET}")

    data = load_v1_beir_subset(EVAL_SUBSET)
    logger.info(f"Eval data: corpus={len(data.corpus_ids)} queries={len(data.query_ids)}")

    results = {}
    for local_supervision in (False, True):
        tag = "local" if local_supervision else "global"
        model, processor = build_and_train(local_supervision, logger, run_dir)
        results[tag] = evaluate_model(model, processor, tag, data, logger)
        del model
        torch.cuda.empty_cache()

    save_json(run_dir / "results.json", results)
    logger.info(f"FINAL RESULTS: {results}")
    print(pd.DataFrame({
        (tag, mode): metrics for tag, modes in results.items() for mode, metrics in modes.items()
    }).T)


if __name__ == "__main__":
    main()
