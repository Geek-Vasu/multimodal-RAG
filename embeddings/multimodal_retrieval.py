import os
import pandas as pd
import numpy as np
import torch
import clip
import faiss
from PIL import Image
from embeddings.image_retriever import search_similar

def search_by_image(image, top_k=5):
    return search_similar(image, top_k)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")


df = pd.read_csv(CSV_PATH)


def encode_image(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

embeddings = []
filenames = []

for _, row in df.iterrows():
    path = os.path.join(IMAGE_DIR, row["filename"])
    embeddings.append(encode_image(path))
    filenames.append(row["filename"])

embeddings = np.array(embeddings).astype("float32")


index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


query_image = os.path.join(IMAGE_DIR, filenames[0])
q_emb = encode_image(query_image).reshape(1, -1)

scores, idxs = index.search(q_emb, k=3)

print("Query image:", filenames[0])
print("\nRetrieved products:\n")

for i in idxs[0]:
    row = df[df["filename"] == filenames[i]].iloc[0]
    print(f"- {row['brand']} {row['category']} | {row['material']} | {row['style_hint']}")

from llm.reasoner import reason_over_products

retrieved = []

for i in idxs[0]:
    row = df[df["filename"] == filenames[i]].iloc[0]
    retrieved.append({
        "brand": row["brand"],
        "category": row["category"],
        "material": row["material"],
        "style_hint": row["style_hint"]
    })


