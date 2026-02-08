from dotenv import load_dotenv
load_dotenv()

import os
import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------------
# CONFIG — adjust only if your image folder lives elsewhere
# ------------------------------------------------------------------
IMAGE_DIR = Path("data/images")  # <-- MUST contain sneaker images


def image_to_base64_from_filename(filename: str) -> str:
    """
    Load image from disk using filename and convert to base64.
    Frontend-safe. No URLs. No guessing.
    """
    img_path = IMAGE_DIR / filename

    if not img_path.exists():
        return ""

    img = Image.open(img_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def reason_over_products(query: str, retrieved_products: list[dict]):
    """
    LLM ONLY explains.
    Retrieval decides relevance & confidence.
    """

    # --------------------------------------------------------------
    # STEP 1 — STRICT retrieval grounding (REAL scores only)
    # --------------------------------------------------------------
    # STEP 1: ALWAYS take top 5 retrieved products
    strong = sorted(
     retrieved_products,
     key=lambda x: x.get("final_score", x.get("score", 0)),
     reverse=True
       )[:5]
    
    

    # --------------------------------------------------------------
    # STEP 2 — Build explanation-only context (NO scores)
    # --------------------------------------------------------------
    context_lines = []
    for p in strong:
        context_lines.append(
            f"- Brand: {p.get('brand','unknown')}, "
            f"Category: {p.get('category','unknown')}, "
            f"Style: {p.get('style_hint','unknown')}, "
            f"Material: {p.get('material','unknown')}"
        )

    context = "\n".join(context_lines) or "No strong matches found."

    prompt = f"""
You are a fashion expert.

User intent:
{query}

High-confidence retrieved products:
{context}

Explain WHY these products are good matches.
Do NOT invent products.
Do NOT mention scores.
Return a short professional explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    explanation = response.choices[0].message.content.strip()

    # --------------------------------------------------------------
    # FINAL OUTPUT — fully grounded, frontend-ready
    # --------------------------------------------------------------
    return {
        "summary": explanation,
        "recommended": [
            {
                "filename": p["filename"],
                "brand": p.get("brand", "unknown"),
                "confidence": round(p.get("final_score", p.get("score", 0)), 2),  # ← REAL retrieval score
               # <-- already frontend-safe URL),
            }
            for p in strong
        ]
    }