import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_text_index():
    df = pd.read_csv(CSV_PATH)

    texts = []
    filenames = []

    for _, row in df.iterrows():
        text = (
            f"{row['brand']} {row['category']} made of "
            f"{row['material']} material for {row['style_hint']} use"
        )
        texts.append(text)
        filenames.append(row["filename"])

    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, filenames

if __name__ == "__main__":
    index, filenames = build_text_index()
    print(f"Text index size: {index.ntotal}")
    print("Filenames:", filenames)
