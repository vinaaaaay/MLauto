"""
LangGraph StateGraph for the Iterative Coding Module (MCTS).

Graph structure:

  START → select_node → (complete? → END)
       → expand_node → retrieve_node_tutorials → rerank_node_tutorials
       → generate_python_code → generate_bash_script → execute_and_evaluate
       → (SUCCESS → backpropagate → should_continue?)
       → (FIX → analyze_error → backpropagate → should_continue?)
       → should_continue → (select_node or END)
"""

from langgraph.graph import StateGraph, START, END

from shared.state import MLAutoState
from .nodes import (
    select_node,
    expand_node,
    retrieve_node_tutorials,
    rerank_node_tutorials,
    generate_python_code,
    generate_bash_script,
    execute_and_evaluate,
    analyze_error,
    backpropagate,
)


def _route_after_select(state: MLAutoState) -> str:
    """Route after select_node: if complete, end; otherwise expand."""
    if state.get("is_complete"):
        return END
    return "expand_node"


def _route_after_evaluate(state: MLAutoState) -> str:
    """Route after execute_and_evaluate: if FIX, go to analyze_error."""
    if state.get("decision") == "SUCCESS":
        return "backpropagate"
    return "analyze_error"


def _should_continue(state: MLAutoState) -> str:
    """Decide whether to continue the MCTS search or stop."""
    if state.get("is_complete"):
        return END

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    if iteration >= max_iterations:
        return END

    return "select_node"


def build_iterative_coding_graph():
    """
    Build and compile the MCTS-based Iterative Coding StateGraph.

    The graph implements a full MCTS loop:
    1. Select a node to expand (UCT)
    2. Expand with a new child (evolve/debug)
    3. Retrieve and rerank tutorials for this node
    4. Generate Python code + bash script
    5. Execute and evaluate
    6. If failed: analyze the error
    7. Backpropagate results up the tree
    8. Continue or stop

    Returns:
        A compiled LangGraph ready for invocation.
    """
    graph = StateGraph(MLAutoState)

    # Add nodes
    graph.add_node("select_node", select_node)
    graph.add_node("expand_node", expand_node)
    graph.add_node("retrieve_node_tutorials", retrieve_node_tutorials)
    graph.add_node("rerank_node_tutorials", rerank_node_tutorials)
    graph.add_node("generate_python_code", generate_python_code)
    graph.add_node("generate_bash_script", generate_bash_script)
    graph.add_node("execute_and_evaluate", execute_and_evaluate)
    graph.add_node("analyze_error", analyze_error)
    graph.add_node("backpropagate", backpropagate)

    # Wire edges
    graph.add_edge(START, "select_node")

    # After select: check if complete
    graph.add_conditional_edges("select_node", _route_after_select, {
        END: END,
        "expand_node": "expand_node",
    })

    # After expand: retrieve tutorials
    graph.add_edge("expand_node", "retrieve_node_tutorials")
    graph.add_edge("retrieve_node_tutorials", "rerank_node_tutorials")
    graph.add_edge("rerank_node_tutorials", "generate_python_code")
    graph.add_edge("generate_python_code", "generate_bash_script")
    graph.add_edge("generate_bash_script", "execute_and_evaluate")

    # After evaluate: SUCCESS → backpropagate, FIX → analyze_error
    graph.add_conditional_edges("execute_and_evaluate", _route_after_evaluate, {
        "backpropagate": "backpropagate",
        "analyze_error": "analyze_error",
    })

    # analyze_error always → backpropagate
    graph.add_edge("analyze_error", "backpropagate")

    # After backpropagate: continue or end
    graph.add_conditional_edges("backpropagate", _should_continue, {
        "select_node": "select_node",
        END: END,
    })

    return graph.compile()
