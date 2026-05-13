"""
LangGraph StateGraph for the Semantic Memory Module.

Single-node graph:
  START → retrieve_tutorials → END

Can be invoked standalone or composed into a larger pipeline.
"""

from langgraph.graph import StateGraph, START, END

from shared.state import MLAutoState
from .nodes import retrieve_tutorials


def build_semantic_memory_graph():
    """
    Build and compile the Semantic Memory StateGraph.

    Returns:
        A compiled LangGraph that accepts MLAutoState and returns
        the state enriched with tutorial_retrieval (list of TutorialInfo).
    """
    graph = StateGraph(MLAutoState)

    graph.add_node("retrieve_tutorials", retrieve_tutorials)

    graph.add_edge(START, "retrieve_tutorials")
    graph.add_edge("retrieve_tutorials", END)

    return graph.compile()
