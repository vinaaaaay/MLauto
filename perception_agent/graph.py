"""
LangGraph StateGraph for the Perception Module.

Linear chain:
  scan_data → find_description_files → generate_task_description
  → select_tools → retrieve_tutorials (semantic memory) → rerank_tutorials (episodic memory) → END
"""

from langgraph.graph import StateGraph, START, END

from shared.state import MLAutoState
from .nodes import (
    scan_data,
    find_description_files,
    generate_task_description,
    select_tools,
)
from semantic_memory.nodes import retrieve_tutorials
from episodic_memory.nodes import rerank_tutorials


def build_perception_graph():
    """
    Build and compile the Perception StateGraph.

    Returns:
        A compiled LangGraph that accepts MLAutoState and returns
        the state enriched with data_prompt, description_files,
        task_description, selected_tools, current_tool, tool_prompt,
        tutorial_retrieval, and tutorial_prompt.
    """
    graph = StateGraph(MLAutoState)

    # Perception nodes
    graph.add_node("scan_data", scan_data)
    graph.add_node("find_description_files", find_description_files)
    graph.add_node("generate_task_description", generate_task_description)
    graph.add_node("select_tools", select_tools)

    # Memory nodes (from their own modules)
    graph.add_node("retrieve_tutorials", retrieve_tutorials)
    graph.add_node("rerank_tutorials", rerank_tutorials)

    # Wire edges (linear chain)
    graph.add_edge(START, "scan_data")
    graph.add_edge("scan_data", "find_description_files")
    graph.add_edge("find_description_files", "generate_task_description")
    graph.add_edge("generate_task_description", "select_tools")
    graph.add_edge("select_tools", "retrieve_tutorials")
    graph.add_edge("retrieve_tutorials", "rerank_tutorials")
    graph.add_edge("rerank_tutorials", END)

    return graph.compile()
