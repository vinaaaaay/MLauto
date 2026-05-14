from typing import Any, Optional, TypedDict

class IterativeCodingState(TypedDict, total=False):
    # ── Inherited from Perception ──
    input_data_folder: str
    output_folder: str
    user_input: str
    config: dict
    data_prompt: str
    task_description: str
    selected_tools: list[str]
    current_tool: str
    tool_prompt: str
    tutorial_retrieval: list
    tutorial_prompt: str

    # ── MCTS control ──
    node_manager: Any
    iteration: int
    max_iterations: int
    all_error_analyses: list[str]
    is_complete: bool

    # ── MCTS tree state ──
    node_id: int
    parent_node_id: Optional[int]
    time_step: int
    depth: int
    stage: str

    # ── Iterative coding state ──
    python_code: str
    bash_script: str
    python_file_path: str
    stdout: str
    stderr: str
    decision: str
    error_summary: Optional[str]
    validation_score: Optional[float]
    error_analysis: str
    error_message: str
    previous_python_code: str
    previous_bash_script: str

    # ── Results ──
    best_score: Optional[float]
    best_code: str
    best_node_id: Optional[int]
