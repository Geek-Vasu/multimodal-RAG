from PIL import Image
from typing import Optional, Dict, Any

from agent.graph import agent   # your LangGraph agent


def run_fashion_agent(
    mode: str,
    image: Optional[Image.Image] = None,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single backend entry point for Streamlit UI
    """

    if mode not in {"find_similar", "match_outfit"}:
        raise ValueError(f"Invalid mode: {mode}")

    state = {
        "input_type": "image" if image else "text",
        "image": image,
        "query": text,
        "filters": None,

        "generated_query": None,

        "image_results": [],
        "text_results": [],
        "metadata_results": [],

        "merged_results": [],
        "llm_output": "",
        "retry_used": False,
    }

    # Special routing for outfit mode
    if mode == "match_outfit":
        state["input_type"] = "outfit"

    result = agent.invoke(state)
    return result
