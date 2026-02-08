from PIL import Image
from backend.app_service import run_fashion_agent

img = Image.open("test_images/shoe.jpg")

out = run_fashion_agent(
    mode="find_similar",
    image=img,
)

print(out["merged_results"][:3])
print(out["llm_output"])