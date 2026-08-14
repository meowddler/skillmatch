"""SkillMatch API — BM25-ranked job recommendations."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.models import HealthResponse, SearchResponse
from backend.search_engine import DataLoadError, engine

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path("frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the BM25 index once at startup, before the first request."""
    try:
        engine.load()
    except DataLoadError:
        # Log and continue: the app still starts so /api/health can report the
        # failure, rather than crashing with an opaque traceback.
        logger.exception("Failed to load job data")
    yield


app = FastAPI(
    title="SkillMatch API",
    description="Job recommendations ranked with BM25, cross-scored with Jaccard similarity.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=500, description="Skills, comma-separated or free text"),
    top_k: int = Query(20, ge=1, le=50, description="Number of results to return"),
) -> SearchResponse:
    """Rank job listings against the given skills."""
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Search index is unavailable.")

    try:
        results = engine.search(q, top_k=top_k)
    except DataLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SearchResponse(query=q, count=len(results), results=results)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report index status. Useful as a deployment health check."""
    return HealthResponse(
        status="ok" if engine.is_ready else "degraded",
        jobs_indexed=len(engine.jobs),
        background_corpus_size=engine.background_size,
        source=engine.source,
    )


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")