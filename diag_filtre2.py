"""Diagnostic v2 : valide le retrieve() filtrer-puis-classer."""
import sys, time
sys.path.insert(0, "app")
from scripts.rag_chain import get_chain
from scripts.query_filter import parse_constraints, event_matches

chain = get_chain()
q = "Quels concerts de jazz à Paris début juillet 2026 ?"
c = parse_constraints(q, known_cities=chain.known_cities)
print("CONTRAINTES :", c)

# Test 1 : le filter natif de LangChain accepte-t-il un callable sur metadata ?
t0 = time.perf_counter()
try:
    res = chain.vectorstore.similarity_search_with_score(
        q, k=20,
        filter=lambda meta: event_matches(meta, c),
        fetch_k=chain.vectorstore.index.ntotal,
    )
    dt = time.perf_counter() - t0
    print(f"\n[OK] filter callable supporté — {len(res)} chunks en {dt:.2f}s")
    for doc, score in res[:6]:
        m = doc.metadata
        print(f"   [{score:.3f}] {m['title'][:45]} | {m['city']} | {m['date_begin'][:10]}")
except Exception as e:
    print(f"\n[ÉCHEC] filter callable : {type(e).__name__}: {e}")

# Test 2 : via retrieve() complet
print("\n── retrieve() complet ──")
r = chain.retrieve(q)
print(f"{len(r)} événements retournés :")
for item in r:
    e = item["event"]
    print(f"   [{item['score']:.3f}] {e['title'][:45]} | {e['city']} | {e['date_begin'][:10]} | {e['price'][:12]}")
