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
from collections import Counter
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

# The two job APIs report contract types with different vocabularies and
# casing - Adzuna sends "full_time", Jooble sends "Full-time". Mapping both to
# a canonical label stops the same concept appearing twice in the filter list.
CONTRACT_ALIASES = {
    "full_time": "Full-time",
    "full-time": "Full-time",
    "fulltime": "Full-time",
    "permanent": "Full-time",
    "part_time": "Part-time",
    "part-time": "Part-time",
    "parttime": "Part-time",
    "contract": "Contract",
    "contractor": "Contract",
    "temporary": "Contract",
    "temp": "Contract",
    "internship": "Internship",
    "intern": "Internship",
}

# Country-level values match everything, so they are not useful as filters.
LOCATION_STOPWORDS = {"india", "remote", "anywhere"}

# Location strings arrive at inconsistent granularity: some records carry
# "Pune, Maharashtra" (state) while others carry a bare "Bangalore" (city) or
# a district like "Gautam Buddha Nagar". Mapping known cities and districts to
# their state keeps the filter list at one consistent level.
CITY_TO_STATE = {
    "bangalore": "Karnataka", "bengaluru": "Karnataka", "mysore": "Karnataka",
    "mysuru": "Karnataka", "mangalore": "Karnataka", "hubli": "Karnataka",
    "mumbai": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra",
    "thane": "Maharashtra", "nashik": "Maharashtra", "navi mumbai": "Maharashtra",
    "aurangabad": "Maharashtra",
    "hyderabad": "Telangana", "secunderabad": "Telangana", "warangal": "Telangana",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu",
    "salem": "Tamil Nadu", "tiruchirappalli": "Tamil Nadu",
    "delhi": "Delhi", "new delhi": "Delhi",
    "noida": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh",
    "gautam buddha nagar": "Uttar Pradesh", "lucknow": "Uttar Pradesh",
    "kanpur": "Uttar Pradesh", "varanasi": "Uttar Pradesh", "agra": "Uttar Pradesh",
    "gurgaon": "Haryana", "gurugram": "Haryana", "faridabad": "Haryana",
    "panchkula": "Haryana",
    "kolkata": "West Bengal", "howrah": "West Bengal", "siliguri": "West Bengal",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat",
    "rajkot": "Gujarat", "gandhinagar": "Gujarat",
    "jaipur": "Rajasthan", "jodhpur": "Rajasthan", "udaipur": "Rajasthan",
    "kochi": "Kerala", "ernakulam": "Kerala", "thiruvananthapuram": "Kerala",
    "trivandrum": "Kerala", "kozhikode": "Kerala", "calicut": "Kerala",
    "thrissur": "Kerala",
    "indore": "Madhya Pradesh", "bhopal": "Madhya Pradesh",
    "jabalpur": "Madhya Pradesh", "gwalior": "Madhya Pradesh",
    "mohali": "Punjab", "ludhiana": "Punjab", "amritsar": "Punjab",
    "jalandhar": "Punjab",
    "bhubaneswar": "Odisha", "khordha": "Odisha", "cuttack": "Odisha",
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "guntur": "Andhra Pradesh", "tirupati": "Andhra Pradesh",
    "patna": "Bihar", "ranchi": "Jharkhand", "raipur": "Chhattisgarh",
    "dehradun": "Uttarakhand", "guwahati": "Assam",
    "chandigarh": "Chandigarh", "goa": "Goa", "panaji": "Goa",
}

# How many BM25 candidates to re-rank per requested result.
CANDIDATE_MULTIPLIER = 5

# Field weights for the combined score.
#
# A flat index treats "Penetration Tester" in a job title the same as "works
# alongside our penetration testing team" in paragraph four of an AI role's
# description. Scoring the title separately and weighting it heavily fixes
# that: incidental mentions still rank, but never above jobs actually about
# the thing being searched for.
#
# This is the idea behind BM25F, the fielded variant of BM25.
TITLE_WEIGHT = 2.5
BODY_WEIGHT = 1.0

# Ceiling on how many result slots diversification may reassign. Relevance
# stays primary: only this many places can be given to an under-represented
# source, and only when that source has a genuinely scoring result available.
MAX_DIVERSITY_SWAPS = 3


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


def normalise_location(value: str | None) -> str:
    """
    Reduce a location string to a single state or union territory.

    Tries the broadest component first, then falls back to any component that
    maps to a known state. "Pune, Maharashtra" and a bare "Pune" both resolve
    to Maharashtra.
    """
    if not value:
        return ""

    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return ""

    # Prefer the last component - usually already the state.
    for candidate in (parts[-1], *reversed(parts[:-1])):
        key = candidate.casefold()
        if key in LOCATION_STOPWORDS:
            continue
        if key in CITY_TO_STATE:
            return CITY_TO_STATE[key]

    # Nothing recognised: keep the broadest component unless it is a stopword.
    last = parts[-1]
    return "" if last.casefold() in LOCATION_STOPWORDS else last


def normalise_contract(value: str | None) -> str:
    """Map a source-specific contract label onto a canonical one."""
    if not value:
        return ""
    key = value.strip().casefold().replace(" ", "_")
    return CONTRACT_ALIASES.get(key, value.strip().title())


def document_text(job: dict) -> str:
    """
    Build the full indexable text for a job record.

    Live records carry a prepared `search_text` field. Archived records carry a
    structured `skills` list instead.
    """
    if job.get("search_text"):
        return job["search_text"]
    return " ".join(job.get("skills") or [])


def title_text(job: dict) -> str:
    """
    The high-signal field: job title plus the search category it came from.

    Kept separate from the body so a match here can be weighted far more
    heavily than the same term appearing anywhere in a long description.
    """
    return f"{job.get('title', '')} {job.get('category', '')}".strip()


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
        self.bm25_title: BM25Okapi | None = None
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
        title_corpus: list[list[str]] = []
        valid_jobs: list[dict] = []
        job_tokens: list[set[str]] = []

        for job in jobs:
            tokens = tokenize(document_text(job))
            if not tokens:
                continue
            corpus.append(tokens)
            # May be empty for archived records, which have no title field.
            title_corpus.append(tokenize(title_text(job)) or ["__notitle__"])
            job_tokens.append(set(tokens))
            valid_jobs.append(job)

        if not corpus:
            raise DataLoadError("No jobs with usable text were found.")

        self.jobs = valid_jobs
        self._job_tokens = job_tokens
        self.bm25 = BM25Okapi(corpus)
        self.bm25_title = BM25Okapi(title_corpus)

        # Override the small-corpus IDF with background estimates where available.
        background = self._background_idf()
        if background:
            positive = [v for v in background.values() if v > 0]
            # Terms unseen in the background corpus are treated as maximally rare.
            unseen = max(positive) if positive else 0.0

            # Both indexes get the same term statistics, so their scores stay
            # on a comparable scale when combined.
            for index in (self.bm25, self.bm25_title):
                merged = {
                    term: background.get(term, unseen) for term in index.idf
                }
                average = sum(merged.values()) / len(merged) if merged else 0.0

                # rank_bm25 floors non-positive IDF at epsilon * average_idf so
                # very common terms never subtract from a document's score.
                floor = index.epsilon * average
                index.idf = {
                    term: (value if value > 0 else floor)
                    for term, value in merged.items()
                }
                index.average_idf = average

        logger.info("Indexed %d jobs with BM25 (source: %s).", len(self.jobs), self.source)

    # -- search --------------------------------------------------------------

    def _promote_missing_sources(
        self,
        head: list[dict],
        scores,
        query_token_set: set[str],
        top_k: int,
    ) -> list[dict]:
        """
        Guarantee under-represented sources a foothold in the results.

        Listings are aggregated from several job APIs that return different
        amounts of text. Sources with fuller descriptions have more terms
        available to match, so they can dominate on relevance alone even when a
        sparser source holds a comparable listing.

        This looks across the *entire* scored corpus - not just the candidate
        pool, which the dominant source may fill completely - and reassigns at
        most MAX_DIVERSITY_SWAPS of the weakest slots to the best-scoring
        result from each absent source. The cap keeps relevance primary.
        """
        present = {job.get("source") for job in head}
        chosen_urls = {job["url"] for job in head}

        # Best-scoring index per source that isn't already represented.
        best_by_source: dict[str, int] = {}
        for idx, job in enumerate(self.jobs):
            source = job.get("source")
            if not source or source in present or scores[idx] <= 0:
                continue
            if job.get("url") in chosen_urls:
                continue
            current = best_by_source.get(source)
            if current is None or scores[idx] > scores[current]:
                best_by_source[source] = idx

        if not best_by_source:
            return head

        # Strongest absent sources first, capped.
        ordered = sorted(best_by_source.items(), key=lambda kv: scores[kv[1]], reverse=True)
        swaps = min(len(ordered), MAX_DIVERSITY_SWAPS, max(0, top_k - 1))

        promotions = [
            {
                **self.jobs[idx],
                "bm25_score": round(float(scores[idx]), 3),
                "jaccard_score": round(
                    jaccard_similarity(self._job_tokens[idx], query_token_set), 3
                ),
                "promoted": True,
            }
            for _, idx in ordered[:swaps]
        ]

        kept = head[: max(0, top_k - len(promotions))]
        merged = kept + promotions
        merged.sort(key=lambda job: job["bm25_score"], reverse=True)
        return merged

    @staticmethod
    def _matches_filters(
        job: dict,
        source: str | None,
        location: str | None,
        contract: str | None,
    ) -> bool:
        """Check a job against the active filters. Absent filters always pass."""
        if source and job.get("source") != source:
            return False

        if contract and normalise_contract(job.get("experience")) != normalise_contract(contract):
            return False

        if location:
            needle = location.casefold()
            places = job.get("location") or []
            # Match either the raw string or its resolved state, so filtering
            # on "Karnataka" catches a record stored only as "Bangalore".
            matched = any(
                needle in str(place).casefold()
                or needle == normalise_location(place).casefold()
                for place in places
            )
            if not matched:
                return False

        return True

    def filter_options(self, min_listings: int = 3) -> dict[str, list[str]]:
        """
        Distinct values available for each filter, for populating the UI.

        Two cleanups happen here, both driven by what the source APIs actually
        return:

        Contract labels are normalised. Adzuna reports "full_time" where Jooble
        reports "Full-time"; without mapping, the filter lists the same concept
        twice and neither option covers both sources.

        Locations are resolved to a single state. The source data mixes
        granularity - "Pune, Maharashtra" alongside a bare "Bangalore" - so
        cities and districts are mapped to their state to keep the list at one
        consistent level. Country-level values are dropped, since "India"
        matches nearly every listing.
        """
        sources: set[str] = set()
        contracts: set[str] = set()
        regions: Counter = Counter()

        for job in self.jobs:
            if job.get("source"):
                sources.add(job["source"])

            canonical = normalise_contract(job.get("experience"))
            if canonical:
                contracts.add(canonical)

            for place in job.get("location") or []:
                state = normalise_location(place)
                if state:
                    regions[state] += 1

        return {
            "sources": sorted(sources),
            "contracts": sorted(contracts),
            # Only regions with enough listings to be worth filtering on.
            "locations": [
                name for name, count in regions.most_common(25) if count >= min_listings
            ],
        }

    def search(
        self,
        query: str,
        top_k: int = 20,
        diversify_sources: bool = False,
        source: str | None = None,
        location: str | None = None,
        contract: str | None = None,
    ) -> list[dict]:
        """
        Return up to `top_k` jobs ranked by BM25, each with a Jaccard score.

        Filters are applied to the whole corpus before `top_k` is taken, so a
        filtered search still returns a full page of results rather than
        whatever survived from an unfiltered top 20.
        """
        if not self.is_ready:
            raise DataLoadError("Search engine is not initialised.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        body_scores = self.bm25.get_scores(query_tokens)
        title_scores = self.bm25_title.get_scores(query_tokens)
        scores = BODY_WEIGHT * body_scores + TITLE_WEIGHT * title_scores

        filtering = any((source, location, contract))

        # Stage 1 - candidate generation over the full corpus. When filtering,
        # widen the pool: the best matches overall may not survive the filter.
        multiplier = CANDIDATE_MULTIPLIER * (6 if filtering else 1)
        pool_size = min(top_k * multiplier, len(scores))
        candidates = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:pool_size]

        # Stage 2 - Jaccard re-scoring on the candidate pool only.
        query_token_set = set(tokenize(" ".join(split_skills(query))))

        # Score the whole candidate pool. Diversification needs to look past
        # the first `top_k` to find results from under-represented sources.
        pool: list[dict] = []
        for idx in candidates:
            if scores[idx] <= 0:
                continue
            if filtering and not self._matches_filters(
                self.jobs[idx], source, location, contract
            ):
                continue
            pool.append(
                {
                    **self.jobs[idx],
                    "bm25_score": round(float(scores[idx]), 3),
                    "jaccard_score": round(
                        jaccard_similarity(self._job_tokens[idx], query_token_set), 3
                    ),
                }
            )

        head = pool[:top_k]
        # Skip diversification when a source filter is active - promoting other
        # sources would directly contradict what the user asked for.
        if diversify_sources and not source:
            return self._promote_missing_sources(head, scores, query_token_set, top_k)
        return head


engine = JobSearchEngine()