"""Pydantic models describing the API's response shapes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JobResult(BaseModel):
    """A single ranked job listing."""

    title: str
    url: str
    company: str = ""
    experience: str = ""
    salary: str = ""
    location: list[str] = Field(default_factory=list)
    snippet: str = ""
    category: str = ""
    skills: list[str] = Field(default_factory=list)
    bm25_score: float = Field(description="BM25 relevance score. Higher is more relevant.")
    jaccard_score: float = Field(ge=0.0, le=1.0, description="Token overlap ratio.")


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[JobResult]


class HealthResponse(BaseModel):
    status: str
    jobs_indexed: int
    background_corpus_size: int = 0
    source: str = "none"