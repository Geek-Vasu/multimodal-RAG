from pydantic import BaseModel

class TextQuery(BaseModel):
    query: str
    top_k: int = 5
