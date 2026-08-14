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


# --- live records & background corpus ---------------------------------------

def test_document_text_prefers_search_text():
    from backend.search_engine import document_text

    live = {"search_text": "python django api", "skills": ["ignored"]}
    assert document_text(live) == "python django api"


def test_document_text_falls_back_to_skills():
    from backend.search_engine import document_text

    archived = {"skills": ["Python", "Django"]}
    assert document_text(archived) == "Python Django"


def test_document_text_handles_missing_fields():
    from backend.search_engine import document_text

    assert document_text({}) == ""


@pytest.fixture
def live_engine(tmp_path):
    """An engine over live-shaped records, with a larger background corpus."""
    import json

    live = [
        {"title": "Python Developer", "url": "l1", "company": "Acme",
         "snippet": "Django REST APIs", "category": "python developer",
         "search_text": "Python Developer Acme Django REST APIs python developer"},
        {"title": "Java Developer", "url": "l2", "company": "Globex",
         "snippet": "Spring Boot services", "category": "java developer",
         "search_text": "Java Developer Globex Spring Boot services java developer"},
    ]
    # "developer" is near-ubiquitous; "django" is rare. A realistic background
    # corpus needs enough documents for that difference to register.
    background = [{"skills": ["Developer", f"Filler{i}"]} for i in range(10)]
    background[0]["skills"].append("Django")

    live_path = tmp_path / "live.json"
    bg_path = tmp_path / "bg.json"
    live_path.write_text(json.dumps(live), encoding="utf-8")
    bg_path.write_text(json.dumps(background), encoding="utf-8")

    eng = JobSearchEngine(data_path=live_path, background_path=bg_path)
    eng.load()
    return eng


def test_live_engine_indexes_search_text(live_engine):
    assert len(live_engine.jobs) == 2
    assert live_engine.search("django", top_k=1)[0]["title"] == "Python Developer"


def test_background_corpus_is_used(live_engine):
    assert live_engine.background_size == 10


def test_rare_terms_outrank_common_terms(live_engine):
    """
    'Django' appears in one background document; 'developer' appears in the
    live corpus generally. Background IDF should rank the rarer term higher.

    Note: bm25.idf only holds terms present in the live corpus, so background
    terms absent from it (e.g. "common") are not keys here.
    """
    idf = live_engine.bm25.idf
    assert "django" in idf
    assert idf["django"] > idf["developer"], (
        f'django={idf["django"]:.3f} should exceed developer={idf["developer"]:.3f}'
    )


def test_engine_works_without_background(tmp_path):
    """
    Without a background corpus, BM25 falls back to live-corpus statistics.

    Two documents are used rather than one: a term appearing in *every*
    document of a corpus gets an IDF at or near zero, so a single-document
    corpus scores everything 0 and returns nothing. That is BM25 behaving
    correctly, and it is precisely the small-corpus problem the background
    corpus exists to solve.
    """
    import json

    path = tmp_path / "live.json"
    path.write_text(
        json.dumps([
            {"title": "Solo", "url": "s1", "search_text": "python django"},
            {"title": "Other", "url": "s2", "search_text": "java spring"},
            {"title": "Third", "url": "s3", "search_text": "ruby rails"},
        ]),
        encoding="utf-8",
    )
    eng = JobSearchEngine(data_path=path, background_path=None)
    eng.load()

    assert eng.background_size == 0
    assert len(eng.search("python", top_k=1)) == 1


def test_single_document_corpus_scores_zero(tmp_path):
    """Documented edge case: one document means zero IDF, so no results."""
    import json

    path = tmp_path / "one.json"
    path.write_text(
        json.dumps([{"title": "Solo", "url": "s1", "search_text": "python only"}]),
        encoding="utf-8",
    )
    eng = JobSearchEngine(data_path=path, background_path=None)
    eng.load()

    assert eng.is_ready
    assert eng.search("python") == []


def test_missing_background_falls_back_gracefully(tmp_path):
    """A missing background file should warn, not crash."""
    import json

    path = tmp_path / "live.json"
    path.write_text(
        json.dumps([{"title": "Solo", "url": "s1", "search_text": "python only"}]),
        encoding="utf-8",
    )
    eng = JobSearchEngine(data_path=path, background_path=tmp_path / "nope.json")
    eng.load()

    assert eng.is_ready
    assert eng.background_size == 0


# --- data sources -----------------------------------------------------------

def test_load_live_jobs_falls_back_when_supabase_unconfigured(monkeypatch, tmp_path):
    """With no credentials set, loading should use the local snapshot."""
    import backend.data_sources as ds

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    snapshot = tmp_path / "snap.json"
    snapshot.write_text(
        '[{"title": "Local", "url": "u1", "search_text": "python local"}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(ds, "LIVE_SNAPSHOT", snapshot)

    jobs, source = ds.load_live_jobs()
    assert source == "local"
    assert jobs[0]["title"] == "Local"


def test_load_live_jobs_prefers_supabase(monkeypatch, tmp_path):
    """When Supabase returns rows, the snapshot is not read."""
    import backend.data_sources as ds

    monkeypatch.setattr(
        ds, "fetch_from_supabase",
        lambda: [{"title": "Remote", "url": "r1", "search_text": "python remote"}],
    )
    monkeypatch.setattr(ds, "LIVE_SNAPSHOT", tmp_path / "does-not-exist.json")

    jobs, source = ds.load_live_jobs()
    assert source == "supabase"
    assert jobs[0]["title"] == "Remote"


def test_fetch_from_supabase_returns_empty_without_credentials(monkeypatch):
    import backend.data_sources as ds

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert ds.fetch_from_supabase() == []


def test_engine_reports_source(tmp_path):
    """An explicit data_path bypasses Supabase and reports 'local'."""
    import json

    path = tmp_path / "live.json"
    path.write_text(
        json.dumps([
            {"title": "A", "url": "a", "search_text": "python django"},
            {"title": "B", "url": "b", "search_text": "java spring"},
            {"title": "C", "url": "c", "search_text": "ruby rails"},
        ]),
        encoding="utf-8",
    )
    eng = JobSearchEngine(data_path=path, background_path=None)
    eng.load()
    assert eng.source == "local"