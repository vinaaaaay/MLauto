"""
Central state definition for the MLauto pipeline.

This single TypedDict replaces the entire NodeManager + Node dataclass
hierarchy from autogluon-assistant. All LangGraph nodes read from and
write to this state.
"""

from typing import Optional, TypedDict


class MLAutoState(TypedDict, total=False):
    # ── Inputs (set once at startup) ──
    input_data_folder: str
    output_folder: str
    user_input: str
    config: dict

    # ── Perception outputs ──
    data_prompt: str                # from scan_data (DataPerceptionAgent)
    description_files: list[str]    # from find_description_files (DescriptionFileRetrieverAgent)
    task_description: str           # from generate_task_description (TaskDescriptorAgent)
    selected_tools: list[str]       # from select_tools (ToolSelectorAgent)
    current_tool: str               # tool being used in current iteration
    tool_prompt: str                # tool-specific prompt snippet

    # ── Iterative coding state ──
    python_code: str                # from generate_python_code (CoderAgent)
    bash_script: str                # from generate_bash_script (CoderAgent)
    python_file_path: str           # path where python code is saved
    stdout: str                     # from execute_and_evaluate (ExecuterAgent)
    stderr: str                     # from execute_and_evaluate (ExecuterAgent)
    decision: str                   # SUCCESS or FIX
    error_summary: Optional[str]    # from execute_and_evaluate
    validation_score: Optional[float]
    error_analysis: str             # from analyze_error (ErrorAnalyzerAgent)
    error_message: str              # combined error context for debugging
    all_error_analyses: list[str]   # accumulated across iterations
    previous_python_code: str       # for debug/improve context
    previous_bash_script: str       # for debug/improve context

    # ── Control ──
    iteration: int
    max_iterations: int
    best_score: Optional[float]
    best_code: str
    is_complete: bool
