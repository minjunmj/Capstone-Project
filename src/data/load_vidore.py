"""ViDoRe v1 (BEIR format) and v3 (with bbox) dataset loading.

v1 subsets are hosted as `<name>_beir` with three configs: corpus (corpus-id,
image), queries (query-id, query), qrels (query-id, corpus-id, score).

v3 subsets are hosted as `vidore_v3_<domain>` with four configs: corpus,
queries, qrels (query_id, corpus_id, score, content_type, bounding_boxes),
documents_metadata. Field names use underscores instead of hyphens.
"""

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset
from PIL import Image

# name -> HF dataset id (v1, BEIR format)
VIDORE_V1_BEIR_SUBSETS: dict[str, str] = {
    "arxivqa": "vidore/arxivqa_test_subsampled_beir",
    "docvqa": "vidore/docvqa_test_subsampled_beir",
    "infovqa": "vidore/infovqa_test_subsampled_beir",
    "shiftproject": "vidore/shiftproject_test_beir",
    "tabfquad": "vidore/tabfquad_test_subsampled_beir",
    "syntheticDocQA_ai": "vidore/syntheticDocQA_artificial_intelligence_test_beir",
    "syntheticDocQA_energy": "vidore/syntheticDocQA_energy_test_beir",
    "syntheticDocQA_government_reports": "vidore/syntheticDocQA_government_reports_test_beir",
    "syntheticDocQA_healthcare_industry": "vidore/syntheticDocQA_healthcare_industry_test_beir",
    "tatdqa": "vidore/tatdqa_test_beir",  # numbers/table-heavy, used for RQ2
}

# name -> HF dataset id (v3, with bounding-box qrels)
VIDORE_V3_SUBSETS: dict[str, str] = {
    "computer_science": "vidore/vidore_v3_computer_science",
    "energy": "vidore/vidore_v3_energy",
    "finance_en": "vidore/vidore_v3_finance_en",
    "finance_fr": "vidore/vidore_v3_finance_fr",
    "hr": "vidore/vidore_v3_hr",
    "industrial": "vidore/vidore_v3_industrial",
    "pharmaceuticals": "vidore/vidore_v3_pharmaceuticals",
    "physics": "vidore/vidore_v3_physics",
}


@dataclass
class RetrievalDataset:
    """A retrieval-ready (corpus, queries, qrels) triple.

    corpus_ids / query_ids preserve dataset order; qrels maps query_id ->
    {corpus_id: relevance_score}. bounding_boxes (v3 only) maps
    (query_id, corpus_id) -> list of {annotator, x1, y1, x2, y2}.
    """

    name: str
    corpus_ids: list[Any]
    corpus_images: list[Image.Image]
    query_ids: list[Any]
    query_texts: list[str]
    qrels: dict[Any, dict[Any, int]]
    bounding_boxes: dict[tuple[Any, Any], list[dict[str, int]]] = field(default_factory=dict)


def _truncate(ds, limit: int | None):
    return ds if limit is None else ds.select(range(min(limit, len(ds))))


def load_v1_beir_subset(name: str, limit_corpus: int | None = None, limit_queries: int | None = None) -> RetrievalDataset:
    if name not in VIDORE_V1_BEIR_SUBSETS:
        raise ValueError(f"Unknown ViDoRe v1 subset '{name}'. Options: {sorted(VIDORE_V1_BEIR_SUBSETS)}")
    hf_id = VIDORE_V1_BEIR_SUBSETS[name]

    corpus = load_dataset(hf_id, "corpus")
    corpus = corpus[list(corpus.keys())[0]]
    corpus = _truncate(corpus, limit_corpus)

    queries = load_dataset(hf_id, "queries")
    queries = queries[list(queries.keys())[0]]
    queries = _truncate(queries, limit_queries)

    qrels_ds = load_dataset(hf_id, "qrels")
    qrels_ds = qrels_ds[list(qrels_ds.keys())[0]]

    corpus_ids = list(corpus["corpus-id"])
    corpus_id_set = set(corpus_ids)
    query_ids = list(queries["query-id"])
    query_id_set = set(query_ids)

    qrels: dict[Any, dict[Any, int]] = {}
    for row in qrels_ds:
        qid, cid, score = row["query-id"], row["corpus-id"], row["score"]
        if qid not in query_id_set or cid not in corpus_id_set:
            continue  # dropped by limit_corpus/limit_queries truncation
        qrels.setdefault(qid, {})[cid] = score

    return RetrievalDataset(
        name=name,
        corpus_ids=corpus_ids,
        corpus_images=list(corpus["image"]),
        query_ids=query_ids,
        query_texts=list(queries["query"]),
        qrels=qrels,
    )


def load_v3_subset(name: str, limit_corpus: int | None = None, limit_queries: int | None = None) -> RetrievalDataset:
    if name not in VIDORE_V3_SUBSETS:
        raise ValueError(f"Unknown ViDoRe v3 subset '{name}'. Options: {sorted(VIDORE_V3_SUBSETS)}")
    hf_id = VIDORE_V3_SUBSETS[name]

    corpus = load_dataset(hf_id, "corpus")
    corpus = corpus[list(corpus.keys())[0]]
    corpus = _truncate(corpus, limit_corpus)

    queries = load_dataset(hf_id, "queries")
    queries = queries[list(queries.keys())[0]]
    queries = _truncate(queries, limit_queries)

    qrels_ds = load_dataset(hf_id, "qrels")
    qrels_ds = qrels_ds[list(qrels_ds.keys())[0]]

    id_key = "corpus-id" if "corpus-id" in corpus.column_names else "corpus_id"
    corpus_ids = list(corpus[id_key])
    corpus_id_set = set(corpus_ids)
    qid_key = "query-id" if "query-id" in queries.column_names else "query_id"
    query_ids = list(queries[qid_key])
    query_id_set = set(query_ids)

    qrels: dict[Any, dict[Any, int]] = {}
    bounding_boxes: dict[tuple[Any, Any], list[dict[str, int]]] = {}
    for row in qrels_ds:
        qid, cid, score = row["query_id"], row["corpus_id"], row["score"]
        if qid not in query_id_set or cid not in corpus_id_set:
            continue
        qrels.setdefault(qid, {})[cid] = score
        bboxes = row.get("bounding_boxes")
        if bboxes:
            bounding_boxes[(qid, cid)] = bboxes

    return RetrievalDataset(
        name=name,
        corpus_ids=corpus_ids,
        corpus_images=list(corpus["image"]),
        query_ids=query_ids,
        query_texts=list(queries["query"]),
        qrels=qrels,
        bounding_boxes=bounding_boxes,
    )


def load_subset(benchmark: str, name: str, limit_corpus: int | None = None, limit_queries: int | None = None) -> RetrievalDataset:
    if benchmark == "vidore_v1":
        return load_v1_beir_subset(name, limit_corpus, limit_queries)
    if benchmark == "vidore_v3":
        return load_v3_subset(name, limit_corpus, limit_queries)
    raise ValueError(f"Unknown benchmark '{benchmark}' (expected 'vidore_v1' or 'vidore_v3')")
