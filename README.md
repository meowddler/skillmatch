# 🎯 SkillMatch — Job Recommendation Engine

A job recommendation engine that ranks 31,970 real job listings against a user's skills using **BM25** — the ranking algorithm behind Elasticsearch and Lucene — with **Jaccard similarity** as a secondary comparison metric.

Built to explore how different information-retrieval approaches change what "relevant" actually means.

**Live demo:** _coming soon_

## The Problem

Naive skill-matching treats every skill as equally important. Search "Python" against a job corpus and you'll find that a generic tag like `"IT Skills"` — which appears in **6,323 of 31,970 listings** — carries the same weight as something specific like `"Kubernetes"`. That's not useful.

BM25 solves this through inverse document frequency: terms appearing everywhere get weighted near zero, while rare, distinctive skills dominate the score.

## Why BM25 over TF-IDF

BM25 refines TF-IDF in two ways that matter here:

- **Diminishing returns** — a listing mentioning "Python" ten times isn't ten times more relevant than one mentioning it once. TF-IDF scales linearly; BM25 saturates.
- **Length normalization** — a listing with 25 skills shouldn't outrank a focused one with 5 purely because it has more surface area to match against.

## Why keep Jaccard?

The original approach to this problem used Jaccard similarity (set overlap). It's kept as a toggle because the comparison is genuinely instructive:

Searching `java spring boot`, two listings with skills `["Java", "Spring Boot"]` score a **perfect 1.0 Jaccard** — total overlap. But BM25 ranks fuller, more specific listings above them. Jaccard rewards *brevity*, not *relevance*: a short skill list is mathematically easier to fully overlap with.

Both scores are computed and returned by the API so the difference is observable rather than theoretical.

## Architecture

```
jobs_raw.json (53,160 records)
    ↓  clean_data.py — dedupe by URL, strip scraper artifacts
jobs_clean.json (31,970 unique jobs)
    ↓  BM25 index built once at server startup
FastAPI /api/search
    ↓  Stage 1: BM25 scores the full corpus → top candidates
    ↓  Stage 2: Jaccard re-scores only those candidates
Frontend (vanilla JS, no build step)
```


The two-stage design — cheap ranking across everything, then more expensive scoring on a small candidate set — mirrors how production recommender systems separate candidate generation from re-ranking.

## Data Cleaning

The scraped source had real quality problems worth documenting:

| Issue | Scale | Handling |
|---|---|---|
| Duplicate listings (same job scraped under multiple search terms) | 21,190 records (~40%) | Deduplicated by URL |
| `>` prefix artifacts in skill strings | 43 records | Stripped |
| Case-inconsistent duplicate skills within a job | Widespread | Normalized, first casing kept |

Deduplication matters beyond tidiness: BM25's IDF component measures how *rare* a term is across the corpus. Leaving 21k duplicates in would have systematically distorted those statistics.

**Final corpus:** 31,970 jobs · 26,122 unique skills · ~7.25 skills per listing.

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI |
| Ranking | rank-bm25 (BM25Okapi), custom Jaccard implementation |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Data | JSON, loaded and indexed in memory at startup |

## Setup

```bash
git clone https://github.com/meowddler/skillmatch.git
cd skillmatch

python -m venv venv
venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open **http://127.0.0.1:8000**

> The app must run through uvicorn — FastAPI serves both the API and the frontend. Opening `index.html` directly (or via a static server) will fail, since `/api/search` won't exist.

To regenerate the cleaned dataset from raw source data:
```bash
python backend/clean_data.py
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/search?q=<skills>&top_k=20` | Ranked results with both BM25 and Jaccard scores |
| `GET /api/health` | Status and indexed job count |
| `GET /docs` | Interactive API documentation (auto-generated) |

## Known Limitations

- **Exact-token matching.** "ML" won't match "Machine Learning" — BM25 operates on tokens, not meaning. Adding a semantic embedding layer would address this.
- **No formal evaluation yet.** Ranking quality is assessed qualitatively. A labeled query set with Precision@K would make this rigorous.
- **Static dataset.** Listings are a point-in-time scrape, not live.
- **In-memory index.** Fine at this scale; a persistent index (e.g. Elasticsearch) would be needed to scale meaningfully beyond it.

## Attribution
Job listing data originates from a publicly available scraped dataset. The search engine, ranking pipeline, API, and interface in this repository were built independently.
