import os
import json
import logging
import uuid
from datetime import datetime
from tree_store import TreeStore

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RUNS_DIR = os.environ.get("RUNS_DIR", "/runs")

def _get_tree(event):
    tree_input = event.get("mcts_tree")
    if not tree_input:
        if "nodes" in event or "run_id" in event:
            tree_input = event
        else:
            raise ValueError("Missing required parameter: 'mcts_tree'")

    run_id = event.get("run_id") or (tree_input.get("run_id") if isinstance(tree_input, dict) else None)
    
    if run_id:
        tree_path = os.path.join(RUNS_DIR, run_id, "mcts_tree.json")
        if os.path.exists(tree_path):
            logger.info(f"Loading MCTS tree from local storage: {tree_path}")
            with open(tree_path, "r", encoding="utf-8") as f:
                return TreeStore.normalize_tree(json.load(f))

    if not isinstance(tree_input, dict):
        raise ValueError(f"Invalid mcts_tree format: {tree_input}")
    return TreeStore.normalize_tree(tree_input)

def _save_tree(tree, run_id=None):
    if not run_id:
        run_id = tree.get("run_id")
    if run_id:
        tree_path = os.path.join(RUNS_DIR, run_id, "mcts_tree.json")
        logger.info(f"Saving MCTS tree to local storage: {tree_path}")
        os.makedirs(os.path.dirname(tree_path), exist_ok=True)
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2)

def _make_lightweight_tree(tree):
    if "run_id" in tree:
        return {
            "run_id": tree["run_id"],
            "iteration": tree.get("iteration", 0),
            "max_iterations": tree.get("max_iterations", 10),
            "all_error_analyses": tree.get("all_error_analyses", []),
            "best_score": tree.get("best_score"),
            "best_code": tree.get("best_code", ""),
            "best_node_id": tree.get("best_node_id"),
            "best_validation_score": tree.get("best_validation_score"),
            "worst_validation_score": tree.get("worst_validation_score")
        }
    return tree

def handle_request(event: dict) -> dict:
    """
    Unified entry point for the stateless MCTS tree operations.
    Inspects the incoming payload for an 'action' parameter and routes accordingly.
    """
    action = event.get("action")
    if not action:
        if "perception_results" in event or "selected_tools" in event:
            action = "init"
        else:
            raise ValueError("Missing required parameter: 'action'")

    logger.info(f"MCTS Handler Invoked with Action: {action}")
    run_id = event.get("run_id")

    if action == "init":
        perception_results = event.get("perception_results", {})
        selected_tools = perception_results.get("selected_tools") or event.get("selected_tools", ["machine learning"])
        mcts_config = event.get("config", {}).get("mcts", {})
        
        tree = TreeStore.initialize(mcts_config, selected_tools)
        tree["iteration"] = 0
        tree["max_iterations"] = event.get("max_iterations") or mcts_config.get("max_iterations", 10)
        tree["all_error_analyses"] = []
        tree["best_score"] = None
        tree["best_code"] = ""
        tree["best_node_id"] = None
        
        if run_id:
            tree["run_id"] = run_id
            _save_tree(tree, run_id)
            return _make_lightweight_tree(tree)
        
        return tree

    elif action == "select":
        tree = _get_tree(event)
        
        node_id = TreeStore.select_node(tree)
        if node_id is None:
            logger.info("No expandable nodes. Finalizing.")
            return {
                "node_id": None,
                "stage": "root",
                "depth": 0,
                "is_complete": True
            }
            
        node = tree["nodes"][node_id]
        logger.info(f"Selected node {node_id} (stage={node.get('stage')}, depth={node.get('depth')})")
        
        return {
            "node_id": node_id,
            "stage": node.get("stage", "root"),
            "depth": node.get("depth", 0),
            "is_complete": False,
        }

    elif action == "expand":
        tree = _get_tree(event)
        selection = event.get("current_selection", {})
        
        if not selection:
            selection = {
                "node_id": event.get("node_id"),
                "stage": event.get("stage", "evolve"),
                "depth": event.get("depth", 0),
                "is_complete": event.get("is_complete", False)
            }
            
        parent_id = selection.get("node_id")
        if parent_id is not None:
            parent_id = int(parent_id)
        
        if selection.get("is_complete") or parent_id is None:
            return {"mcts_tree": _make_lightweight_tree(tree), "current_selection": selection}
            
        new_id = TreeStore.expand_node(tree, parent_id)
        child = tree["nodes"][new_id]
        logger.info(f"Created child node {new_id} (stage={child['stage']}, tool={child['tool_used']})")
        
        parent_context = TreeStore.get_parent_context(tree, new_id)
        
        selection.update({
            "node_id": new_id,
            "stage": child["stage"],
            "depth": child["depth"],
            "current_tool": child["tool_used"],
            "parent_context": parent_context
        })
        
        _save_tree(tree, run_id)
        
        return {
            "mcts_tree": _make_lightweight_tree(tree),
            "current_selection": selection
        }

    elif action == "update":
        tree = _get_tree(event)
        coding_results = event.get("coding_results", {})
        selection = event.get("current_selection", {})
        
        if not coding_results:
            coding_results = {
                "python_code": event.get("python_code", ""),
                "bash_script": event.get("bash_script", ""),
                "stdout": event.get("stdout", ""),
                "stderr": event.get("stderr", ""),
                "decision": event.get("decision", "FIX"),
                "validation_score": event.get("validation_score"),
                "error_analysis": event.get("error_analysis", ""),
                "error_message": event.get("error_message", ""),
                "execution_time": event.get("execution_time", 0.0),
                "processing_time": event.get("processing_time", 0.0),
                "ai_call_time": event.get("ai_call_time", 0.0)
            }
        if not selection:
            selection = {
                "node_id": event.get("node_id"),
                "stage": event.get("stage"),
                "depth": event.get("depth", 0),
                "is_complete": event.get("is_complete", False)
            }
            
        node_id = selection.get("node_id")
        if node_id is not None:
            node_id = int(node_id)
        
        if node_id is None:
            return {"mcts_tree": _make_lightweight_tree(tree)}
            
        TreeStore.update_node(tree, node_id, coding_results)
        
        all_analyses = tree.get("all_error_analyses", [])
        node = tree["nodes"].get(node_id, {})
        if not node.get("is_successful") and coding_results.get("error_analysis"):
            tool = node.get("tool_used", "")
            analysis_str = str(coding_results.get("error_analysis"))[:1000]
            all_analyses.append(f"[Node {node_id} ({tool})] {analysis_str}")
            
        tree["all_error_analyses"] = all_analyses[-20:]
        
        tree["best_score"] = tree.get("best_validation_score")
        best_node_id = tree.get("best_node_id")
        if best_node_id is not None and best_node_id in tree["nodes"]:
            tree["best_code"] = tree["nodes"][best_node_id].get("python_code", "")
            tree["best_node_id"] = best_node_id
        
        _save_tree(tree, run_id)
        
        return {
            "mcts_tree": _make_lightweight_tree(tree),
            "current_selection": selection
        }

    elif action == "backpropagate":
        tree = _get_tree(event)
        selection = event.get("current_selection", {})
        
        if not selection:
            selection = {
                "node_id": event.get("node_id"),
                "stage": event.get("stage"),
                "depth": event.get("depth", 0),
                "is_complete": event.get("is_complete", False)
            }
            
        node_id = selection.get("node_id")
        if node_id is not None:
            node_id = int(node_id)
            TreeStore.backpropagate(tree, node_id)
            
        tree["iteration"] = tree.get("iteration", 0) + 1
        
        _save_tree(tree, run_id)
        
        return _make_lightweight_tree(tree)

    elif action == "finalize":
        tree = _get_tree(event)
        
        tree_viz = TreeStore.visualize_tree(tree)
        logger.info(f"Final Tree Visualization:\n{tree_viz}")
        
        status = TreeStore.get_status(tree)
        logger.info(f"Tree status: {status}")
        
        status["best_score"] = tree.get("best_validation_score")
        best_node_id = tree.get("best_node_id")
        if best_node_id is not None and best_node_id in tree["nodes"]:
            status["best_code"] = tree["nodes"][best_node_id].get("python_code", "")
            status["best_node_id"] = best_node_id
        else:
            status["best_code"] = ""
            status["best_node_id"] = None
        
        return {"status": status, "tree_visualization": tree_viz}

    else:
        raise ValueError(f"Unknown action: '{action}'")
