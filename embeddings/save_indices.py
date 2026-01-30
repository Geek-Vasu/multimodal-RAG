import os
import faiss
import pickle
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")
INDEX_DIR = os.path.join(BASE_DIR, "indices")

os.makedirs(INDEX_DIR, exist_ok=True)

def encode_image(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

df = pd.read_csv(CSV_PATH)

embeddings = []
filenames = []

for _, row in df.iterrows():
    path = os.path.join(IMAGE_DIR, row["filename"])
    embeddings.append(encode_image(path))
    filenames.append(row["filename"])

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(INDEX_DIR, "image.index"))

with open(os.path.join(INDEX_DIR, "filenames.pkl"), "wb") as f:
    pickle.dump(filenames, f)

print("Image index saved")
print("INDEX PATH:", os.path.join(INDEX_DIR, "image.index"))
