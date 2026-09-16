"""V-SPLADE (naver/v-splade-quality) wrapper: page-level (global) sparse vectors.

Passage/image side: `model.encode_passage(**inputs)` runs the BiModernVBert
encoder + SPLADE sparse head (log1p(relu(lm_head(hidden))) per position, then
max-pooled over the sequence) -- see outputs/vsplade_integration_notes.md for
the checkpoint-loading bugs found and fixed to get this far. This module only
extracts the already-pooled page-level vector (matching the official
`examples/quickstart.py` demo); no per-patch/local extraction here yet.

Query side: inference-free Li-LSR lookup (`model.encode_query`) -- no
transformer forward pass, just an Embedding(input_ids) + precomputed
per-vocab-id weight table.

Requires the naver/v-splade repo's `train/` package on sys.path (cloned to
/workspace/third_party/v-splade) -- its `models` package is a plain top-level
import (`from models.model import ...`), not `train.models`.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

V_SPLADE_REPO = Path("/workspace/third_party/v-splade")


def _ensure_train_dir_on_path():
    train_dir = str(V_SPLADE_REPO / "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)


def _local_snapshot_dir(model_name: str) -> str:
    """`AutoProcessor.from_pretrained('naver/v-splade-quality', trust_remote_code=True)`
    fails with a TypeError inside transformers 4.57.6's lazy chat-template
    resolution when given a bare repo id (see outputs/vsplade_integration_notes.md
    bug 1) -- but works fine once the repo is fully downloaded and loaded from a
    local directory. snapshot_download is a cheap no-op on repeat calls (cached)."""
    import os

    if os.path.isdir(model_name):
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name)


class VSpladeModel:
    def __init__(self, model_name: str = "naver/v-splade-quality", dtype: str = "bfloat16", device: str = "cuda"):
        _ensure_train_dir_on_path()
        from models import build_model
        from transformers import Idefics3Processor

        self.local_dir = _local_snapshot_dir(model_name)
        torch_dtype = getattr(torch, dtype)
        self.model = build_model(self.local_dir, mode="inference_only", dtype=torch_dtype).to(device).eval()
        self.model.query_encoder.to(device, dtype=torch_dtype)
        self.model.query_encoder.build_lookup_table()
        self.processor = Idefics3Processor.from_pretrained(self.local_dir)
        self.tokenizer = self.processor.tokenizer
        self.device = device
        self.dtype = torch_dtype
        self.vocab_size = self.model.vocab_size

        chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ""}]}]
        self._image_prompt = self.processor.apply_chat_template(chat, add_generation_prompt=True)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = 8, show_progress: bool = True) -> torch.Tensor:
        """Returns (N, vocab_size) page-level sparse vectors."""
        chunks = []
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="vsplade encode_images")
        for start in iterator:
            batch_images = [im.convert("RGB") for im in images[start : start + batch_size]]
            inputs = self.processor(
                images=batch_images,
                text=[self._image_prompt] * len(batch_images),
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            w = self.model.encode_passage(**inputs)
            chunks.append(w.float().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_images_local(
        self, images: list[Image.Image], batch_size: int = 8, show_progress: bool = True
    ) -> list[torch.Tensor]:
        """Returns a list of (K_i, vocab_size) tensors, one per image -- K_i
        local (per-tile) sparse vectors. K_i = the number of Idefics3
        image-splitting tiles for that image (varies with aspect ratio/size --
        unlike the fixed K=4/8/16 grid used for the earlier ColQwen setup,
        there is no need to force a fixed K here since scoring is max-based).

        Mirrors `model._apply_sparse_head` (train/models/model.py) but skips
        the final pooling step: calls `encoder.encode_passage(**inputs)`
        directly to get the pre-pooling (hidden, attention_mask), applies the
        same `lm_head(hidden) * scale -> log1p(relu(.))` transform per
        position, then instead of pooling over the whole sequence, pools only
        within each tile's own contiguous block of <image> placeholder tokens.

        Verified empirically (see outputs/vsplade_integration_notes.md): each
        tile contributes exactly `image_seq_len` (64) contiguous positions
        with input_id == image_token_id, in the same order as the tiles in
        `pixel_values` -- Idefics3's standard layout, not re-derived here.
        """
        image_token_id = self.model.encoder.image_token_id
        scale = self.model.hidden_size**-0.25
        results: list[torch.Tensor] = []
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="vsplade encode_images_local")
        for start in iterator:
            batch_images = [im.convert("RGB") for im in images[start : start + batch_size]]
            inputs = self.processor(
                images=batch_images,
                text=[self._image_prompt] * len(batch_images),
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            hidden, attn_mask = self.model.encoder.encode_passage(**inputs)  # (B, seq, hidden), (B, seq)
            h = self.model.head.lm_head(hidden) * scale
            w = torch.log1p(torch.relu(h))  # (B, seq, vocab), pre-pooling
            w = w * self.model.special_token_mask.to(w.dtype)

            input_ids = inputs["input_ids"]
            for i in range(w.shape[0]):
                real_len = int(attn_mask[i].sum().item())
                ids_i = input_ids[i, :real_len].cpu()
                w_i = w[i, :real_len].float().cpu()

                is_img = (ids_i == image_token_id).numpy().astype(int)
                padded = np.concatenate(([0], is_img, [0]))
                diff = np.diff(padded)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                if len(starts) == 0:
                    results.append(torch.zeros((0, w.shape[-1])))
                    continue
                local_vecs = [w_i[s:e].max(dim=0).values for s, e in zip(starts, ends)]
                results.append(torch.stack(local_vecs, dim=0))
        return results

    @torch.no_grad()
    def encode_queries(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> torch.Tensor:
        """Returns (N, vocab_size) sparse vectors via the inference-free Li-LSR lookup."""
        chunks = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="vsplade encode_queries")
        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            tok = self.tokenizer(batch_texts, return_tensors="pt", add_special_tokens=False, padding=True)
            q = self.model.encode_query(tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device))
            chunks.append(q.float().cpu())
        return torch.cat(chunks, dim=0)
