import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")
INDEX_PATH = os.path.join(BASE_DIR, "indices", "text.index")

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(CSV_PATH)

texts = (
    df["brand"].fillna("") + " "
    + df["category"].fillna("") + " "
    + df["material"].fillna("") + " "
    + df["style_hint"].fillna("")
).tolist()

embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
print("Text index built:", INDEX_PATH)

print("Text index size:", index.ntotal)