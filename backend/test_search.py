from backend.search_engine import engine

engine.load()

for query in ["python, machine learning", "java spring boot", "kubernetes docker aws"]:
    print(f"\n=== {query} ===")
    for r in engine.search(query, top_k=5):
        print(f"  [{r['bm25_score']:6.2f} | J={r['jaccard_score']:.2f}] {r['title'][:60]}")
        print(f"      {', '.join(r['skills'][:6])}")