from typing import TypedDict, Optional

class PerceptionState(TypedDict, total=False):
    # ── Inputs ──
    input_data_folder: str
    output_folder: str
    user_input: str
    config: dict

    # ── Perception outputs ──
    data_prompt: str
    description_files: list[str]
    task_description: str
    selected_tools: list[str]
    current_tool: str
    tool_prompt: str

    # ── Memory (delegated) ──
    tutorial_retrieval: list
    tutorial_prompt: str

    # ── Additional ──
    all_error_analyses: list[str]
