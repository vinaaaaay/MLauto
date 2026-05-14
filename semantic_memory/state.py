from typing import TypedDict

class SemanticMemoryState(TypedDict, total=False):
    config: dict
    output_folder: str
    task_description: str
    data_prompt: str
    user_input: str
    current_tool: str
    all_error_analyses: list[str]
    tutorial_retrieval: list
