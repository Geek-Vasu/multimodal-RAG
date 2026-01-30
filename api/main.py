
from api.models import TextQuery
from fastapi import FastAPI, UploadFile, File
from embeddings.image_retriever import search_similar
from PIL import Image
import io


app = FastAPI()
@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/search/text")
def search_text(payload: TextQuery):
    return {
        "received_query": payload.query,
        "top_k": payload.top_k
    }

@app.post("/search/image")
def search_image(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = search_similar(image)

    return {
        "query_image": file.filename,
        "results": results
    }
