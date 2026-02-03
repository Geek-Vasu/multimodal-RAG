from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from embeddings.multimodal_retrieval import search_by_image
from embeddings.search_by_text import search_by_text
from embeddings.search_by_metadata import search_by_metadata

from llm.reasoner import reason_over_products
from agent.outfit_planner import outfit_planner


# ---------------- STATE ---------------- #

class AgentState(TypedDict):
    input_type: str

    image: Any | None
    query: str | None
    filters: Dict[str, Any] | None
    generated_query: str | None

    image_results: List[Dict]
    text_results: List[Dict]
    metadata_results: List[Dict]

    merged_results: List[Dict]
    llm_output: Dict
    retry_used: bool


# ---------------- ROUTER ---------------- #

def route_input(state: AgentState):
    return state["input_type"]

def router_node(state: AgentState):
    return state


# ---------------- SEARCH NODES ---------------- #

def image_search_node(state: AgentState):
    state["image_results"] = search_by_image(state["image"])
    return state


def text_search_node(state: AgentState):
    state["text_results"] = search_by_text(state["query"])
    return state


def metadata_search_node(state: AgentState):
    state["metadata_results"] = search_by_metadata(state["filters"])
    return state


# ---------------- OUTFIT PLANNER ---------------- #

def outfit_planner_node(state: AgentState):
    outfit = outfit_planner(state["image"])
    state["generated_query"] = outfit["generated_query"]
    state["query"] = outfit["generated_query"]
    return state


# ---------------- MERGE ---------------- #

def merge_results_node(state: AgentState):
    merged = {}

    for source, weight in [
        ("image_results", 0.5),
        ("text_results", 0.3),
        ("metadata_results", 0.2),
    ]:
        for r in state.get(source, []):
            key = r["filename"]

            if key not in merged:
                merged[key] = r.copy()
                merged[key]["final_score"] = 0.0
                merged[key]["sources"] = []

            merged[key]["final_score"] += weight * r.get("score", 1.0)
            merged[key]["sources"].append(source)

    state["merged_results"] = sorted(
        merged.values(),
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return state


# ---------------- REASONING ---------------- #

def reasoning_node(state: AgentState):
    state["llm_output"] = reason_over_products(
        query=state.get("query", "Analyze retrieved products"),
        retrieved_products=state["merged_results"],
    )
    return state


def confidence_check(state: AgentState):
    strong = [
        r for r in state["llm_output"]["recommended"]
        if r.get("confidence", 0) >= 0.7
    ]
    return "accept" if strong else "retry"


def relaxed_reasoning_node(state: AgentState):
    state["llm_output"] = reason_over_products(
        query="Relax constraints and focus on broader compatibility",
        retrieved_products=state["merged_results"],
    )
    state["retry_used"] = True
    return state


# ---------------- GRAPH ---------------- #

try:
    graph = StateGraph(AgentState)

    # nodes
    graph.add_node("router", router_node)
    graph.add_node("image_search", image_search_node)
    graph.add_node("text_search", text_search_node)
    graph.add_node("metadata_search", metadata_search_node)
    graph.add_node("outfit_planner", outfit_planner_node)
    graph.add_node("merge", merge_results_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("relaxed_reasoning", relaxed_reasoning_node)

    # entry
    graph.set_entry_point("router")

    # routing
    graph.add_conditional_edges(
        "router",
        route_input,
        {
            "image": "image_search",
            "text": "text_search",
            "metadata": "metadata_search",
            "outfit": "outfit_planner",
        },
    )

    # flows
    graph.add_edge("image_search", "merge")
    graph.add_edge("text_search", "merge")
    graph.add_edge("metadata_search", "merge")
    graph.add_edge("outfit_planner", "text_search")

    graph.add_edge("merge", "reason")

    graph.add_conditional_edges(
        "reason",
        confidence_check,
        {
            "accept": END,
            "retry": "relaxed_reasoning",
        },
    )

    graph.add_edge("relaxed_reasoning", END)

    agent = graph.compile()

except Exception:
    import warnings
    warnings.warn("langgraph not available — using fallback agent")

    class _FallbackAgent:
        def invoke(self, state: AgentState):
            t = state["input_type"]

            if t == "outfit":
                state = outfit_planner_node(state)
                state = text_search_node(state)
            elif t == "image":
                state = image_search_node(state)
            elif t == "text":
                state = text_search_node(state)
            elif t == "metadata":
                state = metadata_search_node(state)
            else:
                return state

            state = merge_results_node(state)
            state = reasoning_node(state)

            if confidence_check(state) == "retry":
                state = relaxed_reasoning_node(state)

            return state

    agent = _FallbackAgent()
