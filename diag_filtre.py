"""Diagnostic : que contient l'index et que voit le filtre ?"""
import sys
sys.path.insert(0, "app")
from scripts.rag_chain import get_chain
from scripts.query_filter import parse_constraints, event_matches

chain = get_chain()
q = "Quels concerts de jazz à Paris début juillet 2026 ?"

# 1. Contraintes extraites
c = parse_constraints(q, known_cities=chain.known_cities)
print("CONTRAINTES :", c)
print("known_cities contient 'Paris' ?", "Paris" in chain.known_cities)
print()

# 2. Recherche brute (sans filtre) : que renvoie FAISS et quelles métadonnées ?
raw = chain.vectorstore.similarity_search_with_score(q, k=5)
print(f"FAISS renvoie {len(raw)} chunks. Métadonnées du 1er :")
doc, score = raw[0]
for k, v in doc.metadata.items():
    print(f"   {k:12} = {v!r}")
print()

# 3. Le filtre matche-t-il ces métadonnées ?
nb_ok = sum(1 for doc, s in raw if event_matches(doc.metadata, c))
print(f"Sur ces 5 chunks, {nb_ok} passent le filtre")

# 4. Recherche large + filtre, comme dans retrieve()
big = chain.vectorstore.similarity_search_with_score(q, k=400)
ok = [(d, s) for d, s in big if event_matches(d.metadata, c)]
print(f"Sur 400 chunks : {len(ok)} passent le filtre")
if ok:
    print("Exemple retenu :", ok[0][0].metadata.get("title"), "|", ok[0][0].metadata.get("date_begin"))
