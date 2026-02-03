import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "indices", "text.index")
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

_model = None
_index = None
_df = None

def _load_resources():
    global _model, _index, _df

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    if _df is None:
        _df = pd.read_csv(CSV_PATH)

    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError("Text index not found. Run build_text_index.py")
        _index = faiss.read_index(INDEX_PATH)

def search_by_text(query: str, k: int = 5):
    _load_resources()

    q_emb = _model.encode([query], normalize_embeddings=True)
    scores, indices = _index.search(q_emb, k)

    results = []
    for i, idx in enumerate(indices[0]):
        row = _df.iloc[idx]
        results.append({
            "filename": row["filename"],
            "brand": row["brand"],
            "category": row["category"],
            "material": row["material"],
            "style_hint": row["style_hint"],
            "score": float(scores[0][i]),
        })

    return results
