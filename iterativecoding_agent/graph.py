"""
LangGraph StateGraph for the Iterative Coding Module.

Loop: generate_python_code → generate_bash_script → execute_and_evaluate
                                                          │
                                              ┌───────────┴───────────┐
                                          SUCCESS → END          FIX → analyze_error
                                                                       │
                                                               loop back to generate_python_code
"""

from langgraph.graph import StateGraph, START, END

from shared.state import MLAutoState
from .nodes import (
    generate_python_code,
    generate_bash_script,
    execute_and_evaluate,
    analyze_error,
)


def should_continue(state: MLAutoState) -> str:
    """
    Routing function after execute_and_evaluate.

    Returns:
        "done" if SUCCESS or max iterations reached,
        "analyze_error" if FIX and iterations remain.
    """
    if state.get("decision") == "SUCCESS":
        return "done"

    max_iters = state.get("max_iterations", 5)
    if state.get("iteration", 0) >= max_iters:
        return "done"

    return "analyze_error"


def build_iterative_coding_graph():
    """
    Build and compile the Iterative Coding StateGraph.

    Returns:
        A compiled LangGraph with a conditional loop for debugging.
    """
    graph = StateGraph(MLAutoState)

    # Add nodes
    graph.add_node("generate_python_code", generate_python_code)
    graph.add_node("generate_bash_script", generate_bash_script)
    graph.add_node("execute_and_evaluate", execute_and_evaluate)
    graph.add_node("analyze_error", analyze_error)

    # Wire edges
    graph.add_edge(START, "generate_python_code")
    graph.add_edge("generate_python_code", "generate_bash_script")
    graph.add_edge("generate_bash_script", "execute_and_evaluate")

    # Conditional: SUCCESS/max_iters → END, FIX → analyze_error → loop
    graph.add_conditional_edges(
        "execute_and_evaluate",
        should_continue,
        {
            "done": END,
            "analyze_error": "analyze_error",
        },
    )
    graph.add_edge("analyze_error", "generate_python_code")  # loop back

    return graph.compile()
