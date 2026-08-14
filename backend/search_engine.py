"""
Job search engine: BM25 ranking with Jaccard similarity as a comparison metric.

BM25 is the ranking function used by Elasticsearch and Lucene. It improves on
TF-IDF with term-frequency saturation (repeated terms give diminishing returns)
and document-length normalisation (verbose listings don't win by default).

Background corpus
-----------------
BM25's IDF term measures how rare a word is across the collection. The live
corpus fetched from the Jooble API is small (~500 listings), which makes those
statistics noisy: there isn't enough evidence to tell a distinctive skill apart
from a generic one.

To fix that, IDF is estimated from a much larger archived corpus (~32k listings)
and those values are used when ranking the live set. This is standard practice
in information retrieval — using a large background collection to obtain
reliable term statistics for a smaller target collection.

Search runs in two stages, mirroring how production recommenders separate
candidate generation from re-ranking:
    1. BM25 scores the whole corpus and selects a candidate pool.
    2. Jaccard similarity re-scores only that pool.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.data_sources import (
    ARCHIVE_PATH,
    DataLoadError,
    load_live_jobs,
    read_json_list,
)

logger = logging.getLogger(__name__)

# Keep +, # and . so tokens like "c++", "c#" and ".net" survive tokenisation.
_TOKEN_SPLIT = re.compile(r"[^a-z0-9+#.]+")

# User input may separate skills with commas, newlines, or natural conjunctions.
_SKILL_SPLIT = re.compile(r"[,\n;]|\band\b|&")

# How many BM25 candidates to re-rank per requested result.
CANDIDATE_MULTIPLIER = 5


def tokenize(text: str) -> list[str]:
    """Lowercase text and split into search tokens."""
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


def split_skills(query: str) -> list[str]:
    """Split a free-form query into individual skill phrases."""
    return [s.strip() for s in _SKILL_SPLIT.split(query) if s.strip()]


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Set overlap ratio: |A n B| / |A u B|. Returns 0.0 for empty input."""
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def document_text(job: dict) -> str:
    """
    Build the indexable text for a job record.

    Live records (Jooble) carry a prepared `search_text` field. Archived records
    carry a structured `skills` list instead.
    """
    if job.get("search_text"):
        return job["search_text"]
    return " ".join(job.get("skills") or [])


class JobSearchEngine:
    """Loads the job corpus, builds a BM25 index, and answers search queries."""

    def __init__(
        self,
        data_path: Path | None = None,
        background_path: Path | None = ARCHIVE_PATH,
    ) -> None:
        # data_path=None means "resolve the live corpus via data_sources",
        # which prefers Supabase. Passing an explicit path (as tests do) reads
        # that file directly instead.
        self.data_path = data_path
        self.background_path = background_path

        self.jobs: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.background_size: int = 0
        self.source: str = "none"

        # Precomputed at load time so searches never re-tokenise job text.
        self._job_tokens: list[set[str]] = []

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None and bool(self.jobs)

    # -- background corpus ---------------------------------------------------

    def _background_idf(self) -> dict[str, float] | None:
        """
        Estimate IDF from the large archived corpus.

        Returns None when no background corpus is configured or readable, in
        which case BM25 falls back to statistics from the live corpus alone.
        """
        if self.background_path is None:
            return None

        try:
            archive = read_json_list(self.background_path)
        except DataLoadError:
            logger.warning("Background corpus unavailable; using live-corpus IDF.")
            return None

        doc_freq: dict[str, int] = {}
        total_docs = 0

        for job in archive:
            tokens = set(tokenize(document_text(job)))
            if not tokens:
                continue
            total_docs += 1
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        if total_docs == 0:
            return None

        # Must match rank_bm25's BM25Okapi._calc_idf exactly, or the background
        # values would sit on a different scale to the ones they replace.
        idf = {
            term: math.log(total_docs - freq + 0.5) - math.log(freq + 0.5)
            for term, freq in doc_freq.items()
        }

        self.background_size = total_docs
        logger.info("Estimated IDF from %d background documents.", total_docs)
        return idf

    # -- loading -------------------------------------------------------------

    def load(self) -> None:
        """Read the corpus and build the BM25 index. Called once at startup."""
        if self.data_path is None:
            jobs, self.source = load_live_jobs()
        else:
            jobs, self.source = read_json_list(self.data_path), "local"

        corpus: list[list[str]] = []
        valid_jobs: list[dict] = []
        job_tokens: list[set[str]] = []

        for job in jobs:
            tokens = tokenize(document_text(job))
            if not tokens:
                continue
            corpus.append(tokens)
            job_tokens.append(set(tokens))
            valid_jobs.append(job)

        if not corpus:
            raise DataLoadError("No jobs with usable text were found.")

        self.jobs = valid_jobs
        self._job_tokens = job_tokens
        self.bm25 = BM25Okapi(corpus)

        # Override the small-corpus IDF with background estimates where available.
        background = self._background_idf()
        if background:
            positive = [v for v in background.values() if v > 0]
            # Terms unseen in the background corpus are treated as maximally rare.
            unseen = max(positive) if positive else 0.0

            merged = {
                term: background.get(term, unseen) for term in self.bm25.idf
            }
            average = sum(merged.values()) / len(merged) if merged else 0.0

            # rank_bm25 floors non-positive IDF at epsilon * average_idf so that
            # very common terms never subtract from a document's score.
            floor = self.bm25.epsilon * average
            self.bm25.idf = {
                term: (value if value > 0 else floor) for term, value in merged.items()
            }
            self.bm25.average_idf = average

        logger.info("Indexed %d jobs with BM25 (source: %s).", len(self.jobs), self.source)

    # -- search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return up to `top_k` jobs ranked by BM25, each with a Jaccard score."""
        if not self.is_ready:
            raise DataLoadError("Search engine is not initialised.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Stage 1 - candidate generation over the full corpus.
        pool_size = min(top_k * CANDIDATE_MULTIPLIER, len(scores))
        candidates = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:pool_size]

        # Stage 2 - Jaccard re-scoring on the candidate pool only.
        query_token_set = set(tokenize(" ".join(split_skills(query))))

        results: list[dict] = []
        for idx in candidates:
            if scores[idx] <= 0:
                continue
            results.append(
                {
                    **self.jobs[idx],
                    "bm25_score": round(float(scores[idx]), 3),
                    "jaccard_score": round(
                        jaccard_similarity(self._job_tokens[idx], query_token_set), 3
                    ),
                }
            )
            if len(results) >= top_k:
                break

        return results


engine = JobSearchEngine()