"""
Central state definition for the MLauto pipeline.

This TypedDict is used by all LangGraph nodes. It carries both the
perception-phase outputs and the per-node MCTS artifacts through the
iterative coding graph.
"""

from typing import Optional, TypedDict


class MLAutoState(TypedDict, total=False):
    # ── Inputs (set once at startup) ──
    input_data_folder: str
    output_folder: str
    user_input: str
    config: dict

    # ── Perception outputs ──
    data_prompt: str                # from scan_data
    description_files: list[str]    # from find_description_files
    task_description: str           # from generate_task_description
    selected_tools: list[str]       # ranked tools from select_tools
    current_tool: str               # tool being used for current node
    tool_prompt: str                # tool-specific prompt snippet

    # ── Semantic memory (tutorial retrieval + reranking) ──
    tutorial_retrieval: list        # raw retrieved TutorialInfo list
    tutorial_prompt: str            # formatted tutorial prompt for coder

    # ── MCTS tree state ──
    node_id: int                    # current node's unique id
    parent_node_id: Optional[int]   # parent node id (-1 for root children)
    time_step: int                  # global step counter
    depth: int                      # depth in tree (root=0)
    stage: str                      # "root" | "evolve" | "debug"

    # ── Iterative coding state ──
    python_code: str                # from generate_python_code
    bash_script: str                # from generate_bash_script
    python_file_path: str           # path where python code is saved
    stdout: str                     # from execute_and_evaluate
    stderr: str                     # from execute_and_evaluate
    decision: str                   # SUCCESS or FIX
    error_summary: Optional[str]    # from execute_and_evaluate
    validation_score: Optional[float]
    error_analysis: str             # from analyze_error
    error_message: str              # combined error context for debugging
    all_error_analyses: list[str]   # accumulated across iterations
    previous_python_code: str       # for debug/improve context
    previous_bash_script: str       # for debug/improve context

    # ── Control ──
    iteration: int
    max_iterations: int
    best_score: Optional[float]
    best_code: str
    best_node_id: Optional[int]
    is_complete: bool
