import os
import faiss
import pickle
import torch
import clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "indices")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

# Load FAISS index
index = faiss.read_index(os.path.join(INDEX_DIR, "image.index"))

# Load filename mapping
with open(os.path.join(INDEX_DIR, "filenames.pkl"), "rb") as f:
    filenames = pickle.load(f)

def encode_image(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def search_similar(image_path, k=5):
    query_emb = encode_image(image_path)
    scores, indices = index.search(query_emb, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "filename": filenames[idx],
            "score": float(score)
        })
    return results

if __name__ == "__main__":
    test_image = os.path.join(IMAGE_DIR, filenames[0])
    results = search_similar(test_image)

    for r in results:
        print(r)

print("INDEX PATH:", os.path.join(INDEX_DIR, "image.index"))
