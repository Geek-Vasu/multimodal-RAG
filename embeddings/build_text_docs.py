import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

def build_text_documents():
    df = pd.read_csv(CSV_PATH)
    documents = []

    for _, row in df.iterrows():
        text = (
            f"This product is a {row['brand']} {row['category']} made of "
            f"{row['material']} material, suitable for {row['style_hint']} use."
        )
        documents.append({
            "filename": row["filename"],
            "text": text
        })

    return documents

if __name__ == "__main__":
    docs = build_text_documents()
    print(f"Text documents created: {len(docs)}\n")
    for doc in docs:
        print(doc)
