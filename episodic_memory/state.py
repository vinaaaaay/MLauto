from typing import TypedDict

class EpisodicMemoryState(TypedDict, total=False):
    config: dict
    output_folder: str
    tutorial_retrieval: list
    task_description: str
    data_prompt: str
    user_input: str
    current_tool: str
    tutorial_prompt: str
