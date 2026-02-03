import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "indices", "image.index")
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.read_index(INDEX_PATH)
metadata = pd.read_csv(CSV_PATH)

def search_similar(image: Image.Image, top_k: int = 5):
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    emb = emb.cpu().numpy().astype("float32")
    scores, indices = index.search(emb, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        row = metadata.iloc[idx]
        results.append({
    "filename": row["filename"],
    "brand": row["brand"],
    "category": row["category"],
    "material": row["material"],          
    "style_hint": row["style_hint"],       
    "score": float(score)
})


    return results
