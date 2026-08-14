"""
Job data sources.

Live listings are stored in Supabase (Postgres). The archived corpus used for
background IDF estimation stays on disk: it is static reference data that never
changes, so putting it behind a network call would add latency and a failure
mode for no benefit.

If Supabase is unreachable or unconfigured, loading falls back to the local
JSON snapshot so the app still starts and serves results.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
LIVE_SNAPSHOT = DATA_DIR / "jobs_live.json"
ARCHIVE_PATH = DATA_DIR / "jobs_clean.json"

TABLE = "jobs"

# Supabase caps a single select at 1000 rows, so pages are fetched explicitly.
PAGE_SIZE = 1000

COLUMNS = "title,url,company,experience,salary,location,snippet,category,source,search_text"


class DataLoadError(RuntimeError):
    """Raised when a job corpus cannot be loaded or is unusable."""


def read_json_list(path: Path) -> list[dict]:
    """Read a JSON array of records from disk."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Job data not found at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Job data at {path} is not valid JSON.") from exc

    if not isinstance(data, list) or not data:
        raise DataLoadError(f"Job data at {path} is empty or malformed.")
    return data


def _supabase_client():
    """Build a Supabase client, or return None if it isn't configured."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not (url and key):
        logger.info("Supabase not configured; using local snapshot.")
        return None

    try:
        from supabase import create_client
    except ImportError:
        logger.warning("supabase package not installed; using local snapshot.")
        return None

    try:
        return create_client(url, key)
    except Exception:
        logger.exception("Could not create Supabase client; using local snapshot.")
        return None


def fetch_from_supabase() -> list[dict]:
    """
    Page through the jobs table.

    Returns an empty list on any failure, letting the caller fall back rather
    than taking the whole application down.
    """
    client = _supabase_client()
    if client is None:
        return []

    rows: list[dict] = []
    start = 0

    while True:
        try:
            response = (
                client.table(TABLE)
                .select(COLUMNS)
                .range(start, start + PAGE_SIZE - 1)
                .execute()
            )
        except Exception:
            logger.exception("Supabase query failed at offset %d.", start)
            return []

        page = response.data or []
        rows.extend(page)

        # A short page means we've reached the end.
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    logger.info("Loaded %d live jobs from Supabase.", len(rows))
    return rows


def load_live_jobs() -> tuple[list[dict], str]:
    """
    Load live listings, preferring Supabase and falling back to the snapshot.

    Returns (records, source_name) so the health endpoint can report which
    path was actually used.
    """
    rows = fetch_from_supabase()
    if rows:
        return rows, "supabase"

    logger.info("Falling back to local snapshot at %s.", LIVE_SNAPSHOT)
    return read_json_list(LIVE_SNAPSHOT), "local"