"""
Job search engine: BM25 ranking with Jaccard similarity as a comparison metric.

BM25 is the ranking function used by Elasticsearch and Lucene. It improves on
TF-IDF with term-frequency saturation (repeated terms give diminishing returns)
and document-length normalization (verbose listings don't win by default).

The search runs in two stages, mirroring how production recommenders separate
candidate generation from re-ranking:
    1. BM25 scores the entire corpus and selects a candidate pool.
    2. Jaccard similarity re-scores only that pool.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

DATA_PATH = Path("data") / "jobs_clean.json"

# Keep +, # and . so tokens like "c++", "c#" and ".net" survive tokenization.
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
    """Set overlap ratio: |A ∩ B| / |A ∪ B|. Returns 0.0 for empty input."""
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class DataLoadError(RuntimeError):
    """Raised when the job corpus cannot be loaded or is unusable."""


class JobSearchEngine:
    """Loads the job corpus, builds a BM25 index, and answers search queries."""

    def __init__(self, data_path: Path = DATA_PATH) -> None:
        self.data_path = data_path
        self.jobs: list[dict] = []
        self.bm25: BM25Okapi | None = None
        # Precomputed at load time so searches never re-tokenize job skills.
        self._job_tokens: list[set[str]] = []

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None and bool(self.jobs)

    def load(self) -> None:
        """Read the corpus and build the BM25 index. Called once at startup."""
        try:
            with self.data_path.open(encoding="utf-8") as f:
                jobs = json.load(f)
        except FileNotFoundError as exc:
            raise DataLoadError(
                f"Job data not found at {self.data_path}. "
                "Run `python backend/clean_data.py` to generate it."
            ) from exc
        except json.JSONDecodeError as exc:
            raise DataLoadError(f"Job data at {self.data_path} is not valid JSON.") from exc

        if not isinstance(jobs, list) or not jobs:
            raise DataLoadError(f"Job data at {self.data_path} is empty or malformed.")

        corpus: list[list[str]] = []
        valid_jobs: list[dict] = []
        job_tokens: list[set[str]] = []

        for job in jobs:
            skills = job.get("skills") or []
            if not skills:
                continue
            tokens = tokenize(" ".join(skills))
            if not tokens:
                continue
            corpus.append(tokens)
            job_tokens.append(set(tokens))
            valid_jobs.append(job)

        if not corpus:
            raise DataLoadError("No jobs with usable skill data were found.")

        self.jobs = valid_jobs
        self._job_tokens = job_tokens
        self.bm25 = BM25Okapi(corpus)
        logger.info("Indexed %d jobs with BM25.", len(self.jobs))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return up to `top_k` jobs ranked by BM25, each with a Jaccard score."""
        if not self.is_ready:
            raise DataLoadError("Search engine is not initialised.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Stage 1 — candidate generation over the full corpus.
        pool_size = min(top_k * CANDIDATE_MULTIPLIER, len(scores))
        candidates = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:pool_size]

        # Stage 2 — Jaccard re-scoring on the candidate pool only.
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