from PIL import Image
from agent.graph import agent


img = Image.open("data/images/your_shirt_image.jpg").convert("RGB")

state = {
    "input_type": "outfit",   # 👈 important
    "image": img,
    "query": None,
    "filters": None,

    "image_results": [],
    "text_results": [],
    "metadata_results": [],
    "merged_results": [],
    "retry_used": False
}

result = agent.invoke(state)

print("\nFINAL OUTPUT:\n")
print(result["llm_output"])
