import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="D:/hf_cache")

df = pd.read_csv(CSV_PATH)

texts = [
    f"{row['brand']} {row['category']} made of {row['material']} for {row['style_hint']} use"
    for _, row in df.iterrows()
]

embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

query = "casual adidas sneaker"
q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

scores, idxs = index.search(q_emb, k=3)

print("Query:", query)
for i in idxs[0]:
    print(df.iloc[i]["filename"], df.iloc[i]["brand"])
