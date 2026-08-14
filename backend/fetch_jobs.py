"""
Fetches live job listings from multiple job APIs and merges them into a single
corpus at data/jobs_live.json.

Sources
-------
Jooble  : aggregator, ~20 results/page, returns short snippets.
Adzuna  : aggregator, up to 50 results/page, returns fuller descriptions,
          structured salary ranges, and direct redirect URLs.

Both return overlapping listings, so records are deduplicated on a normalised
(title, company) key as well as URL.

Run:
    python backend/fetch_jobs.py

Requires JOOBLE_API_KEY, ADZUNA_APP_ID and ADZUNA_APP_KEY in a .env file at the
project root. Either source may be omitted; the fetcher skips what it can't
authenticate and reports what it used.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

JOOBLE_KEY = os.getenv("JOOBLE_API_KEY")
ADZUNA_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_KEY = os.getenv("ADZUNA_APP_KEY")

JOOBLE_URL = "https://jooble.org/api/"
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

OUTPUT_PATH = Path("data") / "jobs_live.json"

COUNTRY = "in"          # Adzuna country code
LOCATION = "India"      # Jooble location string
PAGES_PER_KEYWORD = 2
ADZUNA_PER_PAGE = 50
DELAY_SECONDS = 0.4

KEYWORDS = [
    "python developer", "java developer", "javascript developer",
    "react developer", "angular developer", "node js developer",
    "full stack developer", "backend developer", "frontend developer",
    "data scientist", "data analyst", "data engineer",
    "machine learning engineer", "ai engineer", "nlp engineer",
    "devops engineer", "cloud engineer", "aws engineer", "azure engineer",
    "kubernetes engineer", "site reliability engineer",
    "cyber security analyst", "penetration tester", "security engineer",
    "soc analyst", "network engineer", "system administrator",
    "database administrator", "sql developer", "android developer",
    "ios developer", "flutter developer", "qa engineer",
    "automation tester", "business analyst", "product manager",
    "ui ux designer", "salesforce developer", "sap consultant",
    "php developer", "dotnet developer", "golang developer",
]

_WHITESPACE = re.compile(r"\s+")


def clean_text(value: str | None) -> str:
    """Collapse whitespace and strip. Handles None safely."""
    return _WHITESPACE.sub(" ", (value or "")).strip()


def dedupe_key(title: str, company: str) -> str:
    """A loose identity for a listing, used to catch cross-source duplicates."""
    return f"{title.lower()}|{company.lower()}"


def build_record(
    *,
    title: str,
    url: str,
    company: str,
    location: str,
    salary: str,
    contract: str,
    description: str,
    keyword: str,
    source: str,
) -> dict | None:
    """Normalise a listing from any source into the shape the engine indexes."""
    title = clean_text(title)
    url = clean_text(url)
    if not title or not url:
        return None

    company = clean_text(company)
    description = clean_text(description)

    return {
        "title": title,
        "url": url,
        "company": company,
        "experience": clean_text(contract),
        "salary": clean_text(salary),
        "location": [clean_text(location)] if clean_text(location) else [],
        "snippet": description,
        "category": keyword,
        "source": source,
        # Text BM25 indexes. Title is repeated so it carries more weight than
        # an incidental mention buried in the description body.
        "search_text": f"{title} {title} {company} {description} {keyword}",
    }


# --- Jooble -----------------------------------------------------------------

def fetch_jooble(keyword: str, page: int) -> list[dict]:
    payload = {"keywords": keyword, "location": LOCATION, "page": str(page)}
    try:
        response = requests.post(JOOBLE_URL + JOOBLE_KEY, json=payload, timeout=20)
        response.raise_for_status()
        raw = response.json().get("jobs", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"    ! jooble '{keyword}' p{page}: {exc}")
        return []

    records = []
    for job in raw:
        record = build_record(
            title=job.get("title"),
            url=job.get("link"),
            company=job.get("company"),
            location=job.get("location"),
            salary=job.get("salary"),
            contract=job.get("type"),
            description=job.get("snippet"),
            keyword=keyword,
            source="jooble",
        )
        if record:
            records.append(record)
    return records


# --- Adzuna -----------------------------------------------------------------

def format_salary(job: dict) -> str:
    """Adzuna gives numeric min/max. Render as a readable range, or blank."""
    low, high = job.get("salary_min"), job.get("salary_max")
    if not low and not high:
        return ""
    if low and high and low != high:
        return f"{int(low):,} - {int(high):,} PA."
    return f"{int(low or high):,} PA."


def fetch_adzuna(keyword: str, page: int) -> list[dict]:
    url = ADZUNA_URL.format(country=COUNTRY, page=page)
    params = {
        "app_id": ADZUNA_ID,
        "app_key": ADZUNA_KEY,
        "what": keyword,
        "results_per_page": ADZUNA_PER_PAGE,
        "content-type": "application/json",
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        raw = response.json().get("results", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"    ! adzuna '{keyword}' p{page}: {exc}")
        return []

    records = []
    for job in raw:
        record = build_record(
            title=job.get("title"),
            url=job.get("redirect_url"),
            company=(job.get("company") or {}).get("display_name"),
            location=(job.get("location") or {}).get("display_name"),
            salary=format_salary(job),
            contract=job.get("contract_time") or job.get("contract_type") or "",
            description=job.get("description"),
            keyword=keyword,
            source="adzuna",
        )
        if record:
            records.append(record)
    return records


# --- orchestration ----------------------------------------------------------

def fetch_all() -> list[dict]:
    sources = []
    if JOOBLE_KEY:
        sources.append(("jooble", fetch_jooble))
    else:
        print("! JOOBLE_API_KEY missing - skipping Jooble.")

    if ADZUNA_ID and ADZUNA_KEY:
        sources.append(("adzuna", fetch_adzuna))
    else:
        print("! ADZUNA_APP_ID/ADZUNA_APP_KEY missing - skipping Adzuna.")

    if not sources:
        raise SystemExit("No API credentials found. Add them to .env at the project root.")

    seen_urls: set[str] = set()
    seen_listings: set[str] = set()
    jobs: list[dict] = []
    per_source: dict[str, int] = {name: 0 for name, _ in sources}

    for i, keyword in enumerate(KEYWORDS, 1):
        added = 0
        for name, fetch in sources:
            for page in range(1, PAGES_PER_KEYWORD + 1):
                for record in fetch(keyword, page):
                    key = dedupe_key(record["title"], record["company"])
                    if record["url"] in seen_urls or key in seen_listings:
                        continue
                    seen_urls.add(record["url"])
                    seen_listings.add(key)
                    jobs.append(record)
                    per_source[name] += 1
                    added += 1
                time.sleep(DELAY_SECONDS)

        print(f"[{i:>2}/{len(KEYWORDS)}] {keyword:<28} +{added:<4} (total {len(jobs)})")

    print("\nPer source:")
    for name, count in per_source.items():
        print(f"  {name:<8} {count}")

    return jobs


if __name__ == "__main__":
    jobs = fetch_all()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=1)

    print(f"\nSaved {len(jobs)} unique jobs to {OUTPUT_PATH}")