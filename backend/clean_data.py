"""
One-time data cleaning script.
Deduplicates jobs by URL, strips scraping artifacts from skill strings,
and writes a clean JSON file the app loads at startup.
"""
import json
import os

RAW_PATH = os.path.join("data", "jobs_raw.json")
CLEAN_PATH = os.path.join("data", "jobs_clean.json")


def clean_skill(skill: str) -> str:
    """Strip whitespace and the '>' prefix artifact left by the scraper."""
    return skill.strip().lstrip(">").strip()


def clean_dataset():
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    seen_urls = set()
    cleaned = []

    for job in raw:
        url = job.get("url")
        # Deduplicate: the scraper caught many jobs under multiple search terms
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # Clean skills and remove case-insensitive duplicates within a job
        skills, seen_skills = [], set()
        for s in job.get("skills", []):
            s = clean_skill(s)
            if s and s.lower() not in seen_skills:
                seen_skills.add(s.lower())
                skills.append(s)

        if not skills:
            continue

        cleaned.append({
            "title": (job.get("title") or "").strip(),
            "url": url,
            "experience": (job.get("experience") or "").strip(),
            "salary": (job.get("salary") or "").strip(),
            "location": job.get("location") or [],
            "skills": skills,
        })

    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False)

    print(f"Raw records:     {len(raw)}")
    print(f"After cleaning:  {len(cleaned)}")
    print(f"Removed:         {len(raw) - len(cleaned)} duplicates/empties")


if __name__ == "__main__":
    clean_dataset()