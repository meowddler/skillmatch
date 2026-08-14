"""
Uploads the fetched job corpus to a Supabase table.

Run after fetch_jobs.py:
    python backend/upload_to_supabase.py

Records are inserted in batches and upserted on `url`, so re-running after a
fresh fetch updates existing listings rather than failing on duplicates.

Requires SUPABASE_URL and SUPABASE_KEY in .env at the project root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

DATA_PATH = Path("data") / "jobs_live.json"
TABLE = "jobs"
BATCH_SIZE = 500

# Columns that exist in the table. Anything else in the JSON is dropped so a
# stray field from a future API change can't break the insert.
COLUMNS = (
    "title",
    "url",
    "company",
    "experience",
    "salary",
    "location",
    "snippet",
    "category",
    "source",
    "search_text",
)


def to_row(job: dict) -> dict:
    """Project a job record onto the table's columns."""
    return {column: job.get(column, "") for column in COLUMNS} | {
        "location": job.get("location") or []
    }


def main() -> None:
    if not (SUPABASE_URL and SUPABASE_KEY):
        raise SystemExit("SUPABASE_URL / SUPABASE_KEY missing from .env")

    with DATA_PATH.open(encoding="utf-8") as f:
        jobs = json.load(f)

    # The table has a unique constraint on url; drop in-file duplicates first
    # so a single batch can't conflict with itself.
    seen: set[str] = set()
    rows = []
    for job in jobs:
        url = (job.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        rows.append(to_row(job))

    print(f"Prepared {len(rows)} rows from {len(jobs)} records.")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    uploaded = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        try:
            client.table(TABLE).upsert(batch, on_conflict="url").execute()
            uploaded += len(batch)
            print(f"  uploaded {uploaded}/{len(rows)}")
        except Exception as exc:
            print(f"  ! batch starting at {start} failed: {exc}")

    result = client.table(TABLE).select("*", count="exact").limit(1).execute()
    print(f"\nDone. Table now holds {result.count} rows.")


if __name__ == "__main__":
    main()