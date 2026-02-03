from dotenv import load_dotenv
load_dotenv()
import re

from openai import OpenAI
from PIL import Image
import base64
import io
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def outfit_planner(image: Image.Image) -> dict:
    # Convert image → base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = """
You are a fashion reasoning assistant.

Task:
1. Identify key attributes of the clothing item.
2. Decide what kind of shoes generally pair well with it.
3. Generate a concise shoe search query.

Return STRICT JSON:
{
  "attributes": {
    "color": "...",
    "style": "...",
    "formality": "..."
  },
  "generated_query": "..."
}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ],
            }
        ],
        temperature=0.3,
    )
    print("RAW RESPONSE:", response)

    output_text=""
    for message in response.output:
        for content in message.content:
            if content.type=="output_text":
                output_text += content.text
    print("OUTPUT TEXT:", output_text)


    if "```" in output_text:
       output_text = output_text.split("```")[1]

    cleaned = output_text.strip()

    print("\nCLEANED JSON:\n", cleaned)

    return json.loads(cleaned)
