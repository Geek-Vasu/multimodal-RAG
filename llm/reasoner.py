from dotenv import load_dotenv
import json

load_dotenv()

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def reason_over_products(query, retrieved_products):
    context_lines = []

    for p in retrieved_products:
        context_lines.append(
            f"- filename: {p.get('filename','unknown')}, "
            f"brand: {p.get('brand','unknown')}, "
            f"category: {p.get('category','unknown')}, "
            f"material: {p.get('material','unknown')}, "
            f"style: {p.get('style_hint','unknown')}, "
            f"score: {p.get('score', 0.0)}"
        )

    context = "\n".join(context_lines)


    prompt = f"""
You are a strict JSON-only reasoning engine.

User intent:
{query}

Retrieved candidates:
{context}

Return ONLY valid JSON with this schema:
{{
  "recommended": [{{"filename": str, "reason": str, "confidence": float}}],
  "rejected": [{{"filename": str, "reason": str}}],
  "summary": str
}}

Rules:
- confidence must be between 0 and 1
- recommend only strong matches
- reject weak or ambiguous items
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return json.loads(response.choices[0].message.content)