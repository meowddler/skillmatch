"""Unit tests for tokenization, similarity scoring, and ranking behaviour."""

import pytest

from backend.search_engine import (
    JobSearchEngine,
    jaccard_similarity,
    split_skills,
    tokenize,
)


# --- tokenize ---------------------------------------------------------------

def test_tokenize_lowercases():
    assert tokenize("Python SQL") == ["python", "sql"]


def test_tokenize_preserves_symbol_languages():
    """C++, C# and .NET must survive tokenization as single tokens."""
    tokens = tokenize("C++ and C# and .NET")
    assert "c++" in tokens
    assert "c#" in tokens
    assert ".net" in tokens


def test_tokenize_empty_and_punctuation():
    assert tokenize("") == []
    assert tokenize("!!! ???") == []


# --- split_skills -----------------------------------------------------------

def test_split_skills_handles_mixed_separators():
    assert split_skills("python and java, c++ & sql") == ["python", "java", "c++", "sql"]


def test_split_skills_ignores_blanks():
    assert split_skills(",,  ,") == []


# --- jaccard ----------------------------------------------------------------

def test_jaccard_identical_sets_is_one():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard_similarity({"a"}, {"b"}) == 0.0


def test_jaccard_partial_overlap():
    # {a} shared, {a,b,c} union -> 1/3
    assert jaccard_similarity({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)


def test_jaccard_empty_input_is_zero():
    assert jaccard_similarity(set(), {"a"}) == 0.0
    assert jaccard_similarity(set(), set()) == 0.0


# --- engine -----------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    """A small engine instance backed by a temporary fixture corpus."""
    import json

    data = [
        {"title": "Python Developer", "url": "u1", "experience": "2-5 Yrs",
         "salary": "", "location": ["Pune"], "skills": ["Python", "Django", "SQL"]},
        {"title": "Java Engineer", "url": "u2", "experience": "3-7 Yrs",
         "salary": "", "location": ["Delhi"], "skills": ["Java", "Spring Boot"]},
        {"title": "Data Scientist", "url": "u3", "experience": "1-4 Yrs",
         "salary": "", "location": ["Remote"], "skills": ["Python", "Machine Learning"]},
        {"title": "No Skills Listed", "url": "u4", "experience": "",
         "salary": "", "location": [], "skills": []},
    ]
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    eng = JobSearchEngine(data_path=path)
    eng.load()
    return eng


def test_engine_skips_jobs_without_skills(engine):
    """The 4th fixture job has no skills and must be excluded from the index."""
    assert len(engine.jobs) == 3


def test_engine_is_ready_after_load(engine):
    assert engine.is_ready


def test_search_returns_relevant_job_first(engine):
    results = engine.search("python django", top_k=3)
    assert results[0]["title"] == "Python Developer"


def test_search_excludes_zero_score_matches(engine):
    """Jobs with no term overlap should not appear at all."""
    results = engine.search("python", top_k=10)
    assert all("Java" not in r["title"] for r in results)


def test_search_respects_top_k(engine):
    assert len(engine.search("python", top_k=1)) == 1


def test_search_empty_query_returns_nothing(engine):
    assert engine.search("") == []
    assert engine.search("!!!") == []


def test_search_unknown_terms_return_nothing(engine):
    assert engine.search("zzzqqq nonexistent") == []


def test_search_results_include_both_scores(engine):
    result = engine.search("python", top_k=1)[0]
    assert "bm25_score" in result
    assert 0.0 <= result["jaccard_score"] <= 1.0


def test_missing_data_file_raises(tmp_path):
    from backend.search_engine import DataLoadError

    eng = JobSearchEngine(data_path=tmp_path / "nope.json")
    with pytest.raises(DataLoadError, match="not found"):
        eng.load()


def test_malformed_json_raises(tmp_path):
    from backend.search_engine import DataLoadError

    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    eng = JobSearchEngine(data_path=path)
    with pytest.raises(DataLoadError, match="valid JSON"):
        eng.load()


def test_empty_corpus_raises(tmp_path):
    from backend.search_engine import DataLoadError

    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    eng = JobSearchEngine(data_path=path)
    with pytest.raises(DataLoadError, match="empty or malformed"):
        eng.load()