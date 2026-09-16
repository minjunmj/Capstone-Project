"""Automatic lexical-cue heuristics for query classification (CLAUDE_RESEARCH.md
sec 9). This is a CANDIDATE labeler only -- sec 9.1/17 require a human-reviewable
sample file before these labels are trusted for analysis; never rely on the
automatic label alone.
"""

import re
from dataclasses import dataclass

DIGIT_RE = re.compile(r"\d")
PERCENT_MONEY_RE = re.compile(r"[%$₩]|\b(million|billion|thousand|percent)\b", re.IGNORECASE)
UPPER_ABBREV_RE = re.compile(r"\b[A-Z]{2,}\b")
CODE_TOKEN_RE = re.compile(r"\b[A-Za-z]+[-_/][0-9]+\b|\b[0-9]+[-_/][A-Za-z]+\b")
# Capitalized multi-word run, e.g. "JPMorgan Chase", "Qwen2.5" -- crude entity/proper-noun proxy.
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,3}\b")
# Common sentence-initial words that are capitalized purely by English convention, not because
# they're proper nouns -- without this, "What is...", "Explain the..." get false-positive
# "entity" hits on the first word alone, and every question gets nudged toward lexical/mixed.
_SENTENCE_STARTERS = {
    "what", "did", "explain", "summarize", "how", "why", "is", "are", "does", "do", "can",
    "where", "when", "who", "which", "list", "give", "describe", "compare", "calculate",
    "find", "show", "according", "based", "in", "the", "please",
}


def _strip_sentence_starter(query: str) -> str:
    words = query.split()
    if words and words[0].strip("?.,!\"'").lower() in _SENTENCE_STARTERS:
        return query[len(words[0]):]
    return query


# Found via manual review of 45 candidate labels (outputs/lexical_subset_candidates.csv,
# see research_log.md sec 9.1 sample check): financial-statement line-item words are
# capitalized by table/document convention (e.g. "Income", "Other", "Net cash"), not
# because they're proper nouns. ENTITY_RE alone flagged these as "entities", which
# inflated lexical-heavy/mixed counts for queries that are really about generic
# accounting concepts, not a specific hard-to-guess lexical string. Only filters
# single-word entity matches; multi-word capitalized runs (e.g. "North America",
# "VMware Inc") are left alone since generic false positives there were rare.
_GENERIC_LINE_ITEM_WORDS = {
    "income", "property", "other", "trade", "share", "shares", "devices", "accrued",
    "net", "common", "total", "expenses", "expense", "revenue", "revenues", "assets",
    "liabilities", "cash", "cost", "costs", "value", "balance", "change", "increase",
    "decrease", "amount", "number", "table", "items", "convertible",
}


def _filter_generic_single_word_entities(matches: list[str]) -> list[str]:
    return [m for m in matches if " " in m or m.lower() not in _GENERIC_LINE_ITEM_WORDS]


@dataclass
class LexicalCueMatch:
    digit: bool
    percent_money: bool
    upper_abbrev: bool
    code_token: bool
    entity: bool
    matched_terms: list[str]

    @property
    def any_cue(self) -> bool:
        return self.digit or self.percent_money or self.upper_abbrev or self.code_token or self.entity


def detect_lexical_cues(query: str) -> LexicalCueMatch:
    entity_text = _strip_sentence_starter(query)
    entity_matches = _filter_generic_single_word_entities(ENTITY_RE.findall(entity_text))
    terms = []
    for pattern in (PERCENT_MONEY_RE, UPPER_ABBREV_RE, CODE_TOKEN_RE):
        terms.extend(pattern.findall(query))
    terms.extend(entity_matches)
    # DIGIT_RE.findall would return individual digit chars; grab full digit runs instead.
    terms.extend(re.findall(r"\d[\d,.]*", query))
    terms = [t for t in terms if t]  # PERCENT_MONEY_RE's capturing group yields '' on the bare-symbol branch
    return LexicalCueMatch(
        digit=bool(DIGIT_RE.search(query)),
        percent_money=bool(PERCENT_MONEY_RE.search(query)),
        upper_abbrev=bool(UPPER_ABBREV_RE.search(query)),
        code_token=bool(CODE_TOKEN_RE.search(query)),
        entity=bool(entity_matches),
        matched_terms=list(dict.fromkeys(terms)),  # dedupe, preserve order
    )


def classify_query(query: str) -> str:
    """Heuristic 3-way label: lexical-heavy / non-lexical / mixed.

    Rule of thumb (candidate only, sec 9.1): >=2 distinct cue *categories*
    firing -> lexical-heavy (multiple hard-to-guess tokens likely decisive);
    exactly 1 category -> mixed (some lexical grounding, but semantic/layout
    understanding still likely needed); 0 -> non-lexical.
    """
    cues = detect_lexical_cues(query)
    num_categories = sum([cues.digit, cues.percent_money, cues.upper_abbrev, cues.code_token, cues.entity])
    if num_categories >= 2:
        return "lexical-heavy"
    if num_categories == 1:
        return "mixed"
    return "non-lexical"


def build_candidate_table(query_ids: list, query_texts: list[str]) -> list[dict]:
    """One row per query -- sec 9.2's required human-reviewable candidate file."""
    rows = []
    for qid, text in zip(query_ids, query_texts):
        cues = detect_lexical_cues(text)
        rows.append({
            "query_id": qid,
            "query_text": text,
            "auto_label": classify_query(text),
            "digit": cues.digit,
            "percent_money": cues.percent_money,
            "upper_abbrev": cues.upper_abbrev,
            "code_token": cues.code_token,
            "entity": cues.entity,
            "matched_terms": "; ".join(cues.matched_terms),
            "human_reviewed": "",   # left blank for manual annotation pass
            "human_label": "",      # left blank for manual annotation pass
        })
    return rows
