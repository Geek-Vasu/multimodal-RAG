from dotenv import load_dotenv
import os

load_dotenv()

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv(""))

def reason_over_products(query, retrieved_products):
    """
    query: user intent (string)
    retrieved_products: list of dicts with metadata
    """

    context = "\n".join([
        f"- Brand: {p['brand']}, Category: {p['category']}, "
        f"Material: {p['material']}, Style: {p['style_hint']}"
        for p in retrieved_products
    ])

    prompt = f"""
You are an AI product analyst.

User request:
{query}

Retrieved product candidates:
{context}

Your tasks:
1. Decide which products best satisfy the request
2. Explain WHY they match
3. Reject weak matches explicitly

Be objective. Do not hallucinate missing facts.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You reason strictly from provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
