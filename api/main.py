from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any
from PIL import Image
import io

from agent.graph import agent

app = FastAPI(title="Agentic Multimodal RAG System")




class TextSearchRequest(BaseModel):
    query: str


class MetadataSearchRequest(BaseModel):
    filters: Dict[str, Any]



@app.get("/")
def root():
    return {"status": "API running"}




@app.post("/search/image")
def search_image(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    state = {
        "input_type": "image",
        "image": image,
        "retry_used": False
    }

    final_state = agent.invoke(state)

    return {
        "results": final_state.get("results"),
        "analysis": final_state.get("llm_output"),
        "retry_used": final_state.get("retry_used", False)
    }



@app.post("/search/text")
def search_text(query: str):
    state = {
        "input_type": "text",
        "query": query,
        "image": None,
        "filters": None,

        "image_results": [],
        "text_results": [],
        "metadata_results": [],

        "merged_results": [],
        "llm_output": "",
        "retry_used": False
    }

    final_state = agent.invoke(state)

    return {
        "query": query,
        "results": final_state["merged_results"],
        "explanation": final_state["llm_output"],
        "retry_used": final_state["retry_used"]
    }







@app.post("/search/metadata")
def search_metadata(request: MetadataSearchRequest):
    state = {
        "input_type": "metadata",
        "filters": request.filters,
        "retry_used": False
    }

    final_state = agent.invoke(state)

    return {
        "results": final_state.get("results"),
        "analysis": final_state.get("llm_output"),
        "retry_used": final_state.get("retry_used", False)
    }
