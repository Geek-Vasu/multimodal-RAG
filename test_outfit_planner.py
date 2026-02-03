from PIL import Image
from agent.outfit_planner import outfit_planner

img=Image.open("data/images/107999.585.jpg")

result=outfit_planner(img)
print(result)