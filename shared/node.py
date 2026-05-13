"""
Node dataclass for MCTS tree search.

Ported from autogluon-assistant/src/autogluon/assistant/managers/node_manager.py.
Each Node represents one iteration attempt in the solution tree.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """
    A node in the solution tree representing a single iteration.
    Stores code, execution results, and evaluation information.
    """

    # Node creation time
    ctime: float = field(default_factory=lambda: time.time())

    # Tree structure
    parent: Optional["Node"] = None
    children: Set["Node"] = field(default_factory=set)

    # Node position in tree
    time_step: int = None       # Global time step when created
    depth: int = 0              # Depth in tree (root=0)

    # Solution stage
    stage: Literal["root", "debug", "evolve"] = "root"

    # MCTS statistics
    visits: int = 0
    validated_visits: int = 0
    failure_visits: int = 0
    unvalidated_visits: int = 0
    validated_reward: float = 0.0

    # Node state tracking
    is_successful: bool = False
    is_debug_successful: bool = False
    is_terminal: bool = False
    debug_attempts: int = 0

    # Solution artifacts
    python_code: str = ""
    bash_script: str = ""
    tool_used: str = ""
    tools_available: List[str] = field(default_factory=list)
    tutorial_retrieval: str = ""
    tutorial_prompt: str = ""

    # Execution results
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    error_analysis: str = ""

    # Evaluation metrics
    validation_score: Optional[float] = None

    # Locking for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    expected_child_count: int = 0

    @property
    def id(self):
        return self.time_step

    def __post_init__(self):
        if self.parent is not None:
            self.parent.add_child(self)
            self.depth = self.parent.depth + 1

    def add_child(self, child: "Node") -> None:
        logger.debug(f"Node {child.id} added to children of Node {self.id}.")
        self.children.add(child)

    def remove_child(self, child: "Node") -> None:
        logger.debug(f"Node {child.id} removed from children of Node {self.id}.")
        self.children.remove(child)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def num_children(self) -> int:
        return len(self.children)

    @property
    def prev_tutorial_prompt(self) -> str:
        if self.parent and self.parent.tutorial_prompt:
            return self.parent.tutorial_prompt
        return ""

    def update(self, reward: float, is_validated: bool = False, is_failure: bool = False) -> None:
        """Update MCTS statistics with a new reward."""
        with self._lock:
            self.visits += 1
            if is_failure:
                self.failure_visits += 1
            elif is_validated and reward is not None:
                self.validated_visits += 1
                self.validated_reward += reward
            else:
                self.unvalidated_visits += 1

    def uct_value(
        self,
        exploration_constant: float = 1.414,
        best_score: Optional[float] = None,
        worst_score: Optional[float] = None,
        failure_offset: float = 0,
        failure_penalty_weight: float = 0.5,
    ) -> float:
        """Calculate the UCT (Upper Confidence Bound for Trees) value."""
        if self.visits == 0:
            return float("inf")

        parent_visits = max(1, self.parent.visits) if self.parent else 1

        # Failure penalty
        normalized_failure_visit = max(0, self.failure_visits - failure_offset)
        failure_penalty = -failure_penalty_weight * normalized_failure_visit / self.visits

        # Validated rewards
        if self.validated_visits > 0:
            if best_score is not None and worst_score is not None and best_score > worst_score:
                avg_raw_score = self.validated_reward / self.validated_visits
                normalized_score = (avg_raw_score - worst_score) / (best_score - worst_score)
                validated_weight = self.validated_visits / self.visits
                validated_contribution = validated_weight * normalized_score
            else:
                validated_contribution = 1.0
        else:
            validated_contribution = 0.0

        exploitation = validated_contribution + failure_penalty
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)

        return exploitation + exploration

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
