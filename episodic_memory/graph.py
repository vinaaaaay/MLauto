"""
LangGraph StateGraph for the Episodic Memory Module.

Single-node graph:
  START → rerank_tutorials → END

Expects tutorial_retrieval to be populated by the Semantic Memory module.
Can be invoked standalone or composed into a larger pipeline.
"""

from langgraph.graph import StateGraph, START, END

from .state import EpisodicMemoryState
from .nodes import rerank_tutorials


def build_episodic_memory_graph():
    """
    Build and compile the Episodic Memory StateGraph.

    Returns:
        A compiled LangGraph that accepts EpisodicMemoryState (with
        tutorial_retrieval populated) and returns the state enriched
        with tutorial_prompt.
    """
    graph = StateGraph(EpisodicMemoryState)

    graph.add_node("rerank_tutorials", rerank_tutorials)

    graph.add_edge(START, "rerank_tutorials")
    graph.add_edge("rerank_tutorials", END)

    return graph.compile()
