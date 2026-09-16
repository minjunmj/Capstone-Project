"""BiModernVBERT wrapper: frozen bi-encoder, single mean-pooled + L2-normalized
vector per image/query (ModernVBERT/bimodernvbert, ~250M params, ModernBERT
text tower + SigLIP2 vision tower). Replaces the ColQwen2.5 multi-vector
late-interaction model used earlier in this project (see
outputs/research_log.md) -- there is no per-patch multi-vector output here,
so scoring is a plain dot product between two (N, dim) matrices, not MaxSim.

Requires the `vbert` branch of illuin-tech/colpali (not on PyPI), installed
editable from /workspace/third_party/colpali. That branch's modernvbert
modeling code has three bugs relative to the published ModernVBERT/bimodernvbert
checkpoint that were patched locally (see the NOTE comments in
third_party/colpali/colpali_engine/models/modernvbert/modeling_modernvbert.py):
  1. init_language_model re-fetched the base text encoder's vocab_size (50368)
     instead of using the checkpoint's own text_config.vocab_size (50408, which
     includes the image placeholder token) -> embedding size mismatch on load.
  2. ModernVBertConnector wrapped modality_projection in an extra `.proj`
     submodule; the checkpoint stores a flat Linear -> weight silently unused,
     connector left randomly initialized.
  3. init_vision_model unwrapped `vision_model.vision_model` down to a single
     level; the checkpoint stores it doubly-nested (the full dual-tower
     Siglip2Model, saved with only its vision-tower weights) -> the entire
     vision encoder was silently left randomly initialized (confirmed via NaN
     similarity scores before the fix).
"""

import torch
from PIL import Image
from tqdm import tqdm


class BiModernVBertModel:
    def __init__(self, model_name: str = "ModernVBERT/bimodernvbert", dtype: str = "bfloat16", device: str = "cuda"):
        from colpali_engine.models import BiModernVBert, BiModernVBertProcessor

        torch_dtype = getattr(torch, dtype)
        # NOTE: ModernVBertConfig never populates config._name_or_path in this
        # transformers version, but the patched modeling code's checkpoint-layout
        # detection (vision nesting depth, connector shape, text embedding vocab
        # split -- these genuinely differ between checkpoints, e.g.
        # ModernVBERT/bimodernvbert vs. naver/v-splade-quality) needs it. Set it
        # explicitly before construction so the right layout is detected for
        # whatever model_name is passed here.
        config = BiModernVBert.config_class.from_pretrained(model_name)
        config._name_or_path = str(model_name)
        self.model = BiModernVBert.from_pretrained(model_name, config=config, dtype=torch_dtype, device_map=device).eval()
        self.processor = BiModernVBertProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = 16, show_progress: bool = True) -> torch.Tensor:
        return self._encode(images, is_query=False, batch_size=batch_size, show_progress=show_progress)

    @torch.no_grad()
    def encode_queries(self, queries: list[str], batch_size: int = 32, show_progress: bool = True) -> torch.Tensor:
        return self._encode(queries, is_query=True, batch_size=batch_size, show_progress=show_progress)

    @torch.no_grad()
    def _encode(self, items: list, is_query: bool, batch_size: int, show_progress: bool) -> torch.Tensor:
        """Returns a single (N, dim) L2-normalized embedding matrix (already
        pooled by the model -- unlike ColQwenModel, there is no per-item list
        of variable-length unpadded token tensors to build here)."""
        chunks = []
        iterator = range(0, len(items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="encode_queries" if is_query else "encode_images")
        for start in iterator:
            batch_items = items[start : start + batch_size]
            batch = self.processor.process_texts(batch_items) if is_query else self.processor.process_images(batch_items)
            batch = batch.to(self.device)
            out = self.model(**batch)  # (batch, dim)
            chunks.append(out.to(torch.float32).cpu())
        return torch.cat(chunks, dim=0)
