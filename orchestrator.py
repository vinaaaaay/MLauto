"""
Orchestrator for the MLauto pipeline.

Coordinates the transition between the Perception graph and the Iterative Coding (MCTS) graph.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
import yaml

from shared.logging_config import configure_logging
from shared.node_manager import NodeManager
from perception_agent.graph import build_perception_graph
from iterativecoding_agent.graph import build_iterative_coding_graph

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent


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


class Orchestrator:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str = None,
        config_path: str = None,
        user_input: str = "",
        max_iterations: int = None,
        verbosity: int = 2,
    ):
        self.input_data_folder = input_data_folder
        self.user_input = user_input
        self.verbosity = verbosity
        
        # ── Load config ──
        self.config = load_config(config_path)

        mcts_config = self.config.get("mcts", {})
        if max_iterations is not None:
            mcts_config["max_iterations"] = max_iterations
        self.config["mcts"] = mcts_config

        self.effective_max_iter = mcts_config.get("max_iterations", 10)

        # ── Output directory ──
        if not output_folder:
            runs_dir = PROJECT_ROOT / "runs"
            runs_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4()
            output_folder = str(runs_dir / f"mlauto-mcts-{ts}-{uid}")

        self.output_path = Path(output_folder).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

        configure_logging(output_dir=str(self.output_path), verbosity=self.verbosity)

    def run(self):
        """Run the end-to-end MLauto pipeline."""
        logger.info(f"MLauto MCTS pipeline starting")
        logger.info(f"  Input:  {self.input_data_folder}")
        logger.info(f"  Output: {self.output_path}")
        logger.info(f"  Max iterations: {self.effective_max_iter}")

        # ══════════════════════════════════════════════════════════════════════
        #  Phase 1: Perception
        # ══════════════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("  PHASE 1: PERCEPTION")
        logger.info("=" * 60)

        perception_graph = build_perception_graph()

        initial_state = {
            "input_data_folder": str(Path(self.input_data_folder).resolve()),
            "output_folder": str(self.output_path),
            "user_input": self.user_input,
            "config": self.config,
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
            config=self.config,
            output_folder=str(self.output_path),
            selected_tools=perception_result.get("selected_tools", ["machine learning"]),
        )

        coding_graph = build_iterative_coding_graph()

        coding_state = {
            # Carry forward from perception
            "input_data_folder": perception_result.get("input_data_folder", ""),
            "output_folder": str(self.output_path),
            "user_input": self.user_input,
            "config": self.config,
            "data_prompt": perception_result.get("data_prompt", ""),
            "task_description": perception_result.get("task_description", ""),
            "selected_tools": perception_result.get("selected_tools", []),
            "current_tool": perception_result.get("current_tool", ""),
            "tool_prompt": perception_result.get("tool_prompt", ""),
            "tutorial_retrieval": perception_result.get("tutorial_retrieval", []),
            "tutorial_prompt": perception_result.get("tutorial_prompt", ""),

            # MCTS control
            "node_manager": node_manager,
            "iteration": 0,
            "max_iterations": self.effective_max_iter,
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
        logger.info(f"  Iterations used: {total_iterations}/{self.effective_max_iter}")
        logger.info(f"  Best validation score: {best_score}")
        logger.info(f"  Best node: {best_node_id}")
        logger.info(f"  Tools used: {', '.join(node_manager.used_tools)}")

        # Create best run copy
        node_manager.create_best_run_copy()

        # Save MCTS tree visualization
        tree_viz = node_manager.visualize_tree()
        tree_path = os.path.join(str(self.output_path), "mcts_tree.txt")
        with open(tree_path, "w") as f:
            f.write(tree_viz)
        logger.info(f"  MCTS tree saved to: {tree_path}")
        logger.info(f"\n{tree_viz}")

        logger.info(f"  Output saved in: {self.output_path}")

        return {
            "best_score": best_score,
            "best_node_id": best_node_id,
            "total_iterations": total_iterations,
            "elapsed_time": elapsed,
            "output_folder": str(self.output_path),
        }
