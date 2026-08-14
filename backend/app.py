from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.search_engine import engine

app = FastAPI(title="SkillMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Load data and build the BM25 index once, when the server boots."""
    engine.load()


@app.get("/api/search")
def search(
    q: str = Query(..., description="Skills, comma-separated or natural language"),
    top_k: int = Query(20, ge=1, le=50),
):
    results = engine.search(q, top_k=top_k)
    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "jobs_indexed": len(engine.jobs)}


# Serve the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def index():
    return FileResponse("frontend/index.html")