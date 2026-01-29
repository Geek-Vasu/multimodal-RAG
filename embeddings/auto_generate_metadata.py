import os
import csv
import base64
import json
import re
import time
from openai import RateLimitError
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

FIELDS = ["filename", "category", "brand", "material", "style_hint"]

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image(image_path):
    image_b64 = encode_image(image_path)

    prompt = """
Analyze the product image objectively.
Return ONLY valid JSON with these keys:
category, brand, material, style_hint

Rules:
- If unsure, use "unknown"
- Do NOT hallucinate brand
- No markdown, no explanations
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a product attribute extractor."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    
    content = re.sub(r"^```json|```$", "", content, flags=re.MULTILINE).strip()

    return json.loads(content)

    

def load_existing():
    if not os.path.exists(CSV_PATH):
        return {}

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row for row in reader}

def main():
    existing = load_existing()
    rows = []

    for img in os.listdir(IMAGE_DIR):
        if img in existing:
            rows.append(existing[img])
            continue

        print(f"Processing {img}...")
        while True:
            try:
              meta = analyze_image(os.path.join(IMAGE_DIR, img))
              break
            except RateLimitError:
                print("Rate limit hit. Waiting 1 second...")
                time.sleep(1)

        row = {
            "filename": img,
            "category": meta["category"],
            "brand": meta["brand"],
            "material": meta["material"],
            "style_hint": meta["style_hint"],
        }
        rows.append(row)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata written for {len(rows)} products")

if __name__ == "__main__":
    main()
