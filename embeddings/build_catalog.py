import os
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

def encode_image(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def build_catalog():
    df = pd.read_csv(CSV_PATH)

    catalog = []

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row["filename"])

        if not os.path.exists(img_path):
            print(f"Skipping missing image: {row['filename']}")
            continue

        embedding = encode_image(img_path)

        catalog.append({
            "filename": row["filename"],
            "category": row["category"],
            "brand": row["brand"],
            "material": row["material"],
            "style_hint": row["style_hint"],
            "embedding": embedding
        })

    return catalog

if __name__ == "__main__":
    catalog = build_catalog()
    print(f"Catalog size: {len(catalog)}")

    print("\nSample entry:")
    print(catalog[0])
