from embeddings.search_by_text import search_by_text

results = search_by_text("casual white sneakers")
for r in results:
    print(r)
