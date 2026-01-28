import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

def encode_image(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

if __name__ == "__main__":
    image_files = sorted(os.listdir(IMAGE_DIR))

    embeddings = []
    filenames = []

    print("Encoding images...")
    for img in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, img)
        emb = encode_image(img_path)
        embeddings.append(emb[0])
        filenames.append(img)

    embeddings = np.vstack(embeddings).astype("float32")

    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Indexed {index.ntotal} images")

    
    query_index = 0
    query_emb = embeddings[query_index].reshape(1, -1)

    scores, indices = index.search(query_emb, k=6)

    plt.figure(figsize=(10, 4))

    
    plt.subplot(1, 6, 1)
    query_img = Image.open(os.path.join(IMAGE_DIR, filenames[query_index]))
    plt.imshow(query_img)
    plt.title("Query")
    plt.axis("off")

   
    for i, idx in enumerate(indices[0][1:]):
        plt.subplot(1, 6, i + 2)
        img = Image.open(os.path.join(IMAGE_DIR, filenames[idx]))
        plt.imshow(img)
        plt.title(f"{scores[0][i+1]:.2f}")
        plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "visual_check.png")
    plt.savefig(output_path)
    print(f"Visual result saved to {output_path}")
