"""Integration tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health_reports_indexed_jobs():
    with TestClient(app) as c:
        response = c.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["jobs_indexed"] > 0


def test_search_returns_results():
    with TestClient(app) as c:
        response = c.get("/api/search", params={"q": "python"})
        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "python"
        assert body["count"] > 0
        assert len(body["results"]) == body["count"]


def test_search_respects_top_k():
    with TestClient(app) as c:
        response = c.get("/api/search", params={"q": "python", "top_k": 5})
        assert len(response.json()["results"]) <= 5


def test_search_rejects_empty_query():
    """min_length=1 on the query parameter should trigger a 422."""
    with TestClient(app) as c:
        assert c.get("/api/search", params={"q": ""}).status_code == 422


def test_search_rejects_out_of_range_top_k():
    with TestClient(app) as c:
        assert c.get("/api/search", params={"q": "python", "top_k": 0}).status_code == 422
        assert c.get("/api/search", params={"q": "python", "top_k": 999}).status_code == 422


def test_search_result_shape():
    with TestClient(app) as c:
        result = c.get("/api/search", params={"q": "java", "top_k": 1}).json()["results"][0]
        for field in ("title", "url", "skills", "bm25_score", "jaccard_score"):
            assert field in result