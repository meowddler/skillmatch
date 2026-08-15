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


# --- source diversification -------------------------------------------------

@pytest.fixture
def lopsided_engine(tmp_path):
    """
    A corpus where one source has far richer text than the other.

    This reproduces the real imbalance between job APIs that return full
    descriptions and those that return short snippets.
    """
    import json

    jobs = [
        {
            "title": "Python Developer",
            "url": f"https://rich/{i}",
            "company": f"Corp{i}",
            "source": "rich_source",
            "search_text": "Python Developer django flask fastapi postgresql celery pytest " * 4,
        }
        for i in range(80)
    ]
    jobs += [
        {
            "title": "Backend Engineer",
            "url": f"https://sparse/{i}",
            "company": f"Startup{i}",
            "source": "sparse_source",
            "search_text": "Backend Engineer django",
        }
        for i in range(10)
    ]
    # Unrelated filler so IDF isn't degenerate.
    jobs += [
        {
            "title": "Java Developer",
            "url": f"https://rich/j{i}",
            "company": "C",
            "source": "rich_source",
            "search_text": "Java Developer spring hibernate maven kafka junit " * 4,
        }
        for i in range(60)
    ]

    path = tmp_path / "live.json"
    path.write_text(json.dumps(jobs), encoding="utf-8")

    eng = JobSearchEngine(data_path=path, background_path=None)
    eng.load()
    return eng


def test_relevance_ranking_lets_one_source_dominate(lopsided_engine):
    """Baseline: without diversification the richer source takes every slot."""
    results = lopsided_engine.search("django", top_k=20)
    assert {r["source"] for r in results} == {"rich_source"}


def test_diversification_surfaces_missing_source(lopsided_engine):
    results = lopsided_engine.search("django", top_k=20, diversify_sources=True)
    assert "sparse_source" in {r["source"] for r in results}


def test_diversification_marks_promoted_results(lopsided_engine):
    results = lopsided_engine.search("django", top_k=20, diversify_sources=True)
    promoted = [r for r in results if r.get("promoted")]
    assert promoted
    assert all(r["source"] == "sparse_source" for r in promoted)


def test_diversification_respects_swap_cap(lopsided_engine):
    from backend.search_engine import MAX_DIVERSITY_SWAPS

    results = lopsided_engine.search("django", top_k=20, diversify_sources=True)
    assert sum(1 for r in results if r.get("promoted")) <= MAX_DIVERSITY_SWAPS


def test_diversification_respects_top_k(lopsided_engine):
    assert len(lopsided_engine.search("django", top_k=5, diversify_sources=True)) <= 5


def test_diversification_is_noop_when_all_sources_present(lopsided_engine):
    """Nothing should be promoted if every source already appears."""
    results = lopsided_engine.search("backend engineer django", top_k=20, diversify_sources=True)
    sources = {r["source"] for r in results}
    if len(sources) > 1:
        assert not any(r.get("promoted") for r in results)


def test_diversification_never_promotes_zero_score_results(lopsided_engine):
    results = lopsided_engine.search("django", top_k=20, diversify_sources=True)
    assert all(r["bm25_score"] > 0 for r in results)


def test_diversification_keeps_results_score_ordered(lopsided_engine):
    results = lopsided_engine.search("django", top_k=20, diversify_sources=True)
    scores = [r["bm25_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# --- filtering --------------------------------------------------------------

@pytest.fixture
def filter_engine(tmp_path):
    import json

    jobs = [
        {"title": "Python Developer", "url": "f1", "source": "adzuna",
         "experience": "full_time", "location": ["Pune, Maharashtra"],
         "search_text": "Python Developer django flask postgresql"},
        {"title": "Python Engineer", "url": "f2", "source": "jooble",
         "experience": "part_time", "location": ["Bangalore, Karnataka"],
         "search_text": "Python Engineer django celery redis"},
        {"title": "Python Analyst", "url": "f3", "source": "adzuna",
         "experience": "full_time", "location": ["Bangalore, Karnataka"],
         "search_text": "Python Analyst pandas numpy reporting"},
        {"title": "Java Developer", "url": "f4", "source": "adzuna",
         "experience": "full_time", "location": ["Pune, Maharashtra"],
         "search_text": "Java Developer spring hibernate maven"},
    ]
    path = tmp_path / "live.json"
    path.write_text(json.dumps(jobs), encoding="utf-8")

    eng = JobSearchEngine(data_path=path, background_path=None)
    eng.load()
    return eng


def test_filter_by_source(filter_engine):
    results = filter_engine.search("python", source="jooble")
    assert {r["source"] for r in results} == {"jooble"}


def test_filter_by_contract(filter_engine):
    results = filter_engine.search("python", contract="part_time")
    assert all(r["experience"] == "part_time" for r in results)


def test_filter_by_location_matches_substring(filter_engine):
    """Stored as 'Bangalore, Karnataka' - filtering on the city should hit."""
    results = filter_engine.search("python", location="Bangalore")
    assert results
    assert all("Bangalore" in r["location"][0] for r in results)


def test_filter_by_location_is_case_insensitive(filter_engine):
    assert filter_engine.search("python", location="bangalore")


def test_filters_combine(filter_engine):
    results = filter_engine.search("python", source="adzuna", location="Bangalore")
    assert all(r["source"] == "adzuna" for r in results)
    assert all("Bangalore" in r["location"][0] for r in results)


def test_filter_excluding_everything_returns_empty(filter_engine):
    assert filter_engine.search("python", location="Antarctica") == []


def test_no_filters_returns_everything_relevant(filter_engine):
    assert len(filter_engine.search("python")) == 3


def test_source_filter_suppresses_diversification(filter_engine):
    """Promoting other sources would contradict an explicit source filter."""
    results = filter_engine.search("python", source="adzuna", diversify_sources=True)
    assert {r["source"] for r in results} == {"adzuna"}


def test_filter_options_lists_distinct_values(filter_engine):
    options = filter_engine.filter_options()
    assert set(options["sources"]) == {"adzuna", "jooble"}
    # Raw values are "full_time"/"part_time"; the filter list shows canonical labels.
    assert set(options["contracts"]) == {"Full-time", "Part-time"}


def test_filter_options_uses_region_not_full_location(filter_engine):
    """Locations collapse to their last component so the list stays usable."""
    options = filter_engine.filter_options(min_listings=1)
    assert set(options["locations"]) == {"Maharashtra", "Karnataka"}


def test_filter_options_drops_thin_regions(filter_engine):
    """Regions with too few listings aren't worth offering as a filter."""
    assert filter_engine.filter_options(min_listings=3)["locations"] == []


# --- contract & location normalisation --------------------------------------

def test_normalise_contract_unifies_source_vocabularies():
    """Adzuna sends 'full_time', Jooble sends 'Full-time'. Same concept."""
    from backend.search_engine import normalise_contract

    assert normalise_contract("full_time") == "Full-time"
    assert normalise_contract("Full-time") == "Full-time"
    assert normalise_contract("permanent") == "Full-time"


def test_normalise_contract_groups_temporary_with_contract():
    from backend.search_engine import normalise_contract

    assert normalise_contract("Temporary") == "Contract"
    assert normalise_contract("contract") == "Contract"


def test_normalise_contract_handles_empty():
    from backend.search_engine import normalise_contract

    assert normalise_contract("") == ""
    assert normalise_contract(None) == ""


def test_normalise_contract_passes_through_unknown_values():
    """An unmapped label is title-cased rather than discarded."""
    from backend.search_engine import normalise_contract

    assert normalise_contract("freelance") == "Freelance"


def test_contract_filter_matches_across_source_vocabularies(tmp_path):
    """Filtering on 'Full-time' must catch Adzuna's 'full_time' too."""
    import json

    # Filler documents without "python" so the term keeps a positive IDF -
    # a term present in every document scores zero and returns nothing.
    records = [
        {"title": "A", "url": "a", "experience": "full_time", "search_text": "python django"},
        {"title": "B", "url": "b", "experience": "Full-time", "search_text": "python flask"},
        {"title": "C", "url": "c", "experience": "part_time", "search_text": "python celery"},
    ] + [
        {"title": f"F{i}", "url": f"f{i}", "experience": "full_time",
         "search_text": "java spring hibernate"}
        for i in range(8)
    ]

    path = tmp_path / "live.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    eng = JobSearchEngine(data_path=path, background_path=None)
    eng.load()

    results = eng.search("python", contract="Full-time")
    assert {r["url"] for r in results} == {"a", "b"}


def test_filter_options_drops_country_level_locations(tmp_path):
    """'India' matches nearly everything, so it isn't offered as a filter."""
    import json

    path = tmp_path / "live.json"
    path.write_text(
        json.dumps([
            {"title": "A", "url": "a", "location": ["India"],
             "search_text": "python django"},
            {"title": "B", "url": "b", "location": ["Pune, Maharashtra"],
             "search_text": "python flask"},
            {"title": "C", "url": "c", "location": ["Remote"],
             "search_text": "python celery"},
        ]),
        encoding="utf-8",
    )
    eng = JobSearchEngine(data_path=path, background_path=None)
    eng.load()

    locations = eng.filter_options(min_listings=1)["locations"]
    assert "India" not in locations
    assert "Remote" not in locations
    assert "Maharashtra" in locations