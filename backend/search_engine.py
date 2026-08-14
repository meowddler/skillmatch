"""
Search engine: BM25 ranking over job skill data.

BM25 is the ranking algorithm used by Elasticsearch/Lucene. It scores how well
each job's skill list matches a user's query, with two refinements over plain
TF-IDF: diminishing returns on repeated terms, and normalization for document
length so skill-heavy listings don't automatically win.
"""
import json
import os
import re
from rank_bm25 import BM25Okapi

CLEAN_PATH = os.path.join("data", "jobs_clean.json")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics. '.NET', 'C++' etc. become tokens too."""
    return [t for t in re.split(r"[^a-z0-9+#.]+", text.lower()) if t]


class JobSearchEngine:
    def __init__(self):
        self.jobs = []
        self.bm25 = None
        self.job_skill_sets = []   # precomputed lowercase skill sets, for Jaccard

    def load(self):
        with open(CLEAN_PATH, encoding="utf-8") as f:
            self.jobs = json.load(f)

        # Each job becomes one "document" made of its skill strings
        corpus = []
        for job in self.jobs:
            skills_text = " ".join(job["skills"])
            corpus.append(tokenize(skills_text))
            self.job_skill_sets.append({s.lower() for s in job["skills"]})

        # Build the BM25 index once at startup — queries after this are fast
        self.bm25 = BM25Okapi(corpus)
        print(f"Indexed {len(self.jobs)} jobs with BM25.")

    def jaccard(self, query_skills: set, job_index: int) -> float:
        """Set overlap ratio: |intersection| / |union|. The original project's method."""
        job_skills = self.job_skill_sets[job_index]
        if not job_skills or not query_skills:
            return 0.0
        # Compare on tokens so 'python' matches 'Python Developer'
        job_tokens = set(tokenize(" ".join(job_skills)))
        query_tokens = set(tokenize(" ".join(query_skills)))
        union = job_tokens | query_tokens
        return len(job_tokens & query_tokens) / len(union) if union else 0.0

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Candidate generation: take a wider pool than we need, then re-rank
        candidate_count = min(top_k * 5, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_count]

        query_skills = {q.strip() for q in re.split(r"[,\n]| and | & ", query) if q.strip()}

        results = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            job = self.jobs[i]
            results.append({
                **job,
                "bm25_score": round(float(scores[i]), 3),
                "jaccard_score": round(self.jaccard(query_skills, i), 3),
            })

        return results[:top_k]


engine = JobSearchEngine()