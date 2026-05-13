"""
MLauto — Entry point for the MCTS-based AutoML pipeline.

Replicates autogluon-assistant's coding_agent.run_agent():
  1. Perception phase: scan data, identify task, select tools, retrieve tutorials
  2. Iterative coding phase: MCTS tree search (select → expand → code → execute → backpropagate)
"""

import argparse
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.logging_config import configure_logging
from shared.node_manager import NodeManager
from perception_agent.graph import build_perception_graph
from iterativecoding_agent.graph import build_iterative_coding_graph

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    default_path = PROJECT_ROOT / "config.yaml"

    config = {}
    if default_path.exists():
        with open(default_path, "r") as f:
            config = yaml.safe_load(f) or {}

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        # Merge (user overrides default)
        _deep_merge(config, user_config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def run_agent(
    input_data_folder: str,
    output_folder: str = None,
    config_path: str = None,
    user_input: str = "",
    max_iterations: int = None,
    verbosity: int = 2,
):
    """
    Run the MLauto pipeline.

    Args:
        input_data_folder: Path to the input data directory
        output_folder: Path to the output directory (auto-generated if None)
        config_path: Optional override config YAML path
        user_input: User instructions / problem statement
        max_iterations: Override for mcts.max_iterations
    """
    # ── Load config ──
    config = load_config(config_path)

    mcts_config = config.get("mcts", {})
    if max_iterations is not None:
        mcts_config["max_iterations"] = max_iterations
    config["mcts"] = mcts_config

    effective_max_iter = mcts_config.get("max_iterations", 10)

    # ── Output directory ──
    if not output_folder:
        runs_dir = PROJECT_ROOT / "runs"
        runs_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4()
        output_folder = str(runs_dir / f"mlauto-mcts-{ts}-{uid}")

    output_path = Path(output_folder).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir=str(output_path), verbosity=verbosity)
    logger.info(f"MLauto MCTS pipeline starting")
    logger.info(f"  Input:  {input_data_folder}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Max iterations: {effective_max_iter}")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1: Perception
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  PHASE 1: PERCEPTION")
    logger.info("=" * 60)

    perception_graph = build_perception_graph()

    initial_state = {
        "input_data_folder": str(Path(input_data_folder).resolve()),
        "output_folder": str(output_path),
        "user_input": user_input,
        "config": config,
        "all_error_analyses": [],
    }

    perception_result = perception_graph.invoke(initial_state)

    logger.info(f"  Task: {perception_result.get('task_description', '')[:200]}...")
    logger.info(f"  Tools: {perception_result.get('selected_tools', [])}")
    logger.info(f"  Current tool: {perception_result.get('current_tool', '')}")
    logger.info(f"  Tutorial prompt length: {len(perception_result.get('tutorial_prompt', ''))} chars")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2: Iterative Coding (MCTS)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  PHASE 2: ITERATIVE CODING (MCTS)")
    logger.info("=" * 60)

    # Create NodeManager for MCTS
    node_manager = NodeManager(
        config=config,
        output_folder=str(output_path),
        selected_tools=perception_result.get("selected_tools", ["machine learning"]),
    )

    coding_graph = build_iterative_coding_graph()

    coding_state = {
        # Carry forward from perception
        "input_data_folder": perception_result.get("input_data_folder", ""),
        "output_folder": str(output_path),
        "user_input": user_input,
        "config": config,
        "data_prompt": perception_result.get("data_prompt", ""),
        "task_description": perception_result.get("task_description", ""),
        "selected_tools": perception_result.get("selected_tools", []),
        "current_tool": perception_result.get("current_tool", ""),
        "tool_prompt": perception_result.get("tool_prompt", ""),
        "tutorial_retrieval": perception_result.get("tutorial_retrieval", []),
        "tutorial_prompt": perception_result.get("tutorial_prompt", ""),

        # MCTS control
        "_node_manager": node_manager,
        "iteration": 0,
        "max_iterations": effective_max_iter,
        "all_error_analyses": [],
        "is_complete": False,
    }

    start_time = time.time()
    coding_result = coding_graph.invoke(coding_state)
    elapsed = time.time() - start_time

    # ══════════════════════════════════════════════════════════════════════
    #  Results Summary
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 60)

    best_score = coding_result.get("best_score")
    best_node_id = coding_result.get("best_node_id")
    total_iterations = coding_result.get("iteration", 0)

    logger.info(f"  MCTS search completed in {elapsed:.2f} seconds")
    logger.info(f"  Total nodes explored: {node_manager.time_step + 1}")
    logger.info(f"  Iterations used: {total_iterations}/{effective_max_iter}")
    logger.info(f"  Best validation score: {best_score}")
    logger.info(f"  Best node: {best_node_id}")
    logger.info(f"  Tools used: {', '.join(node_manager.used_tools)}")

    # Create best run copy
    node_manager.create_best_run_copy()

    # Save MCTS tree visualization
    tree_viz = node_manager.visualize_tree()
    tree_path = os.path.join(str(output_path), "mcts_tree.txt")
    with open(tree_path, "w") as f:
        f.write(tree_viz)
    logger.info(f"  MCTS tree saved to: {tree_path}")
    logger.info(f"\n{tree_viz}")

    logger.info(f"  Output saved in: {output_path}")

    return {
        "best_score": best_score,
        "best_node_id": best_node_id,
        "total_iterations": total_iterations,
        "elapsed_time": elapsed,
        "output_folder": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="MLauto — AutoGluon-Assistant replica in LangGraph")
    parser.add_argument("input_data_folder", help="Path to input data directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--config", "-c", default=None, help="Config YAML path")
    parser.add_argument("--user-input", "-u", default="", help="User instructions")
    parser.add_argument("--max-iterations", "-n", type=int, default=None, help="Max MCTS iterations")
    parser.add_argument("--verbosity", "-v", type=int, default=2, choices=[0, 1, 2, 3, 4], help="Set verbosity level (0-4)")

    args = parser.parse_args()

    run_agent(
        input_data_folder=args.input_data_folder,
        output_folder=args.output,
        config_path=args.config,
        user_input=args.user_input,
        max_iterations=args.max_iterations,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()
