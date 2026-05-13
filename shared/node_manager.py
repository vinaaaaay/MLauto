"""
NodeManager — central orchestrator for the MCTS tree search.

Ported from autogluon-assistant's NodeManager. Manages the tree of Node
objects, handles selection (UCT), expansion, and terminal-node tracking.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from shared.node import Node
from shared.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class NodeManager:
    """
    Manages the MCTS tree for solution search.

    This is NOT a LangGraph node — it sits alongside the state and is
    stored at state["_node_manager"]. LangGraph node functions call its
    methods to perform MCTS selection/expansion/backpropagation.
    """

    def __init__(self, config: dict, output_folder: str, selected_tools: List[str]):
        self.config = config
        self.output_folder = output_folder
        self.selected_tools = selected_tools

        mcts_cfg = config.get("mcts", {})
        self.exploration_constant = mcts_cfg.get("exploration_constant", 1.414)
        self.max_debug_depth = mcts_cfg.get("max_debug_depth", 3)
        self.failure_offset = mcts_cfg.get("failure_offset", 0)
        self.failure_penalty_weight = mcts_cfg.get("failure_penalty_weight", 0.5)
        self.initial_root_children = mcts_cfg.get("initial_root_children", 3)
        self.max_debug_children = mcts_cfg.get("max_debug_children", 2)
        self.max_evolve_children = mcts_cfg.get("max_evolve_children", 2)
        self.continuous_improvement = mcts_cfg.get("continuous_improvement", True)

        # Tree state
        self.root = Node(time_step=-1, stage="root")
        self.root.is_successful = True  # Root is a virtual node
        self.all_nodes: List[Node] = [self.root]
        self.time_step = -1

        # Current node being processed
        self.current_node: Optional[Node] = None

        # Score tracking
        self._best_node: Optional[Node] = None
        self._best_validation_score: Optional[float] = None
        self._worst_validation_score: Optional[float] = None

        # Tools rotation
        self._tool_index = 0

        # Registry for tool info
        registry_path = config.get("tool_registry_path")
        self.registry = ToolRegistry(registry_path)

    @property
    def best_validation_score(self) -> Optional[float]:
        return self._best_validation_score

    @property
    def best_node(self) -> Optional[Node]:
        return self._best_node

    @property
    def used_tools(self) -> List[str]:
        return list(set(n.tool_used for n in self.all_nodes if n.tool_used))

    def _next_time_step(self) -> int:
        self.time_step += 1
        return self.time_step

    def _get_next_tool(self) -> str:
        """Rotate through available tools."""
        if not self.selected_tools:
            return "machine learning"
        tool = self.selected_tools[self._tool_index % len(self.selected_tools)]
        self._tool_index += 1
        return tool

    # ─── MCTS Selection ──────────────────────────────────────────────────

    def select_node(self) -> Optional[Node]:
        """Select a non-terminal leaf node using UCT."""
        # Phase 1: Expand root children first
        if self.root.num_children < self.initial_root_children:
            return self.root

        # Phase 2: UCT selection among leaves
        leaves = self._get_expandable_leaves()
        if not leaves:
            return None

        best = max(leaves, key=lambda n: n.uct_value(
            exploration_constant=self.exploration_constant,
            best_score=self._best_validation_score,
            worst_score=self._worst_validation_score,
            failure_offset=self.failure_offset,
            failure_penalty_weight=self.failure_penalty_weight,
        ))

        return best

    def _get_expandable_leaves(self) -> List[Node]:
        """Get all leaf nodes that can still be expanded."""
        leaves = []
        for node in self.all_nodes:
            if node.is_terminal:
                continue
            if node == self.root:
                continue
            if node.is_leaf:
                leaves.append(node)
            elif node.is_successful and node.num_children < self.max_evolve_children:
                leaves.append(node)
            elif not node.is_successful and node.num_children < self.max_debug_children:
                leaves.append(node)
        return leaves

    # ─── MCTS Expansion ─────────────────────────────────────────────────

    def _create_evolve_node_only(self) -> Node:
        """Create an evolve child without generating code."""
        parent = self.current_node
        tool = self._get_next_tool()

        child = Node(
            parent=parent,
            time_step=self._next_time_step(),
            stage="evolve",
        )
        child.tool_used = tool
        child.tools_available = self.selected_tools

        self.all_nodes.append(child)
        self.current_node = child

        logger.info(f"Created evolve node {child.id} (parent={parent.id}, tool={tool})")
        return child

    def _create_debug_node_only(self) -> Node:
        """Create a debug child without generating code."""
        parent = self.current_node
        tool = parent.tool_used  # Use same tool as parent for debugging

        child = Node(
            parent=parent,
            time_step=self._next_time_step(),
            stage="debug",
        )
        child.tool_used = tool
        child.tools_available = self.selected_tools

        self.all_nodes.append(child)
        self.current_node = child

        logger.info(f"Created debug node {child.id} (parent={parent.id}, tool={tool})")
        return child

    def _find_debug_origin(self, node: Node) -> Optional[Node]:
        """Find the original failed node that started a debug chain."""
        current = node.parent
        while current and current.stage == "debug":
            current = current.parent
        return current if current and current != self.root else None

    # ─── Terminal marking ────────────────────────────────────────────────

    def mark_node_terminal(self, node: Node) -> None:
        """Mark a node and all its descendants as terminal."""
        node.is_terminal = True
        for child in node.children:
            self.mark_node_terminal(child)

    # ─── Results ─────────────────────────────────────────────────────────

    def create_best_run_copy(self) -> None:
        """Copy the best node's output to best_run/ folder."""
        if self._best_node is None:
            return

        best_node_dir = os.path.join(self.output_folder, f"node_{self._best_node.id}")
        best_run_dir = os.path.join(self.output_folder, "best_run")

        if os.path.exists(best_node_dir):
            if os.path.exists(best_run_dir):
                shutil.rmtree(best_run_dir)
            shutil.copytree(best_node_dir, best_run_dir)
            logger.info(f"Best run copied from node_{self._best_node.id} to best_run/")

    def visualize_tree(self) -> str:
        """Generate a text visualization of the MCTS tree."""
        lines = []
        self._visualize_subtree(self.root, "", True, lines)
        return "\n".join(lines)

    def _visualize_subtree(self, node: Node, prefix: str, is_last: bool, lines: list):
        connector = "└── " if is_last else "├── "
        label = f"Node {node.id}"
        if node == self.root:
            label = "Root"
        else:
            status = "✓" if node.is_successful else "✗"
            score = f"score={node.validation_score:.4f}" if node.validation_score is not None else "no score"
            terminal = " [TERMINAL]" if node.is_terminal else ""
            label += f" [{node.stage}|{node.tool_used}|{status}|v={node.visits}|{score}]{terminal}"

        lines.append(f"{prefix}{connector}{label}")
        children = list(node.children)
        for i, child in enumerate(children):
            extension = "    " if is_last else "│   "
            self._visualize_subtree(child, prefix + extension, i == len(children) - 1, lines)
