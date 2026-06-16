"""Diagnostic ciblé : le uid de l'index correspond-il aux clés events_by_uid ?"""
import sys
sys.path.insert(0, "app")
from scripts.rag_chain import get_chain
from scripts.query_filter import parse_constraints, event_matches

chain = get_chain()
q = "Quels concerts de jazz à Paris début juillet 2026 ?"
c = parse_constraints(q, known_cities=chain.known_cities)
res = chain.vectorstore.similarity_search_with_score(
    q, k=5, filter=lambda meta: event_matches(meta, c),
    fetch_k=chain.vectorstore.index.ntotal,
)

print("── uid côté MÉTADONNÉES de l'index ──")
for doc, _ in res[:3]:
    u = doc.metadata.get("uid")
    print(f"   uid={u!r}  type={type(u).__name__}  présent dans events_by_uid ? {u in chain.events_by_uid}")

print("\n── clés côté events_by_uid (3 exemples) ──")
for k in list(chain.events_by_uid)[:3]:
    print(f"   clé={k!r}  type={type(k).__name__}")

# Test croisé str/int
doc0 = res[0][0]
u = doc0.metadata.get("uid")
print(f"\n   str(uid) dans events_by_uid ? {str(u) in chain.events_by_uid}")
print(f"   int(uid) dans events_by_uid ? ", end="")
try:
    print(int(u) in chain.events_by_uid)
except Exception as e:
    print(f"(uid non convertible en int: {e})")
