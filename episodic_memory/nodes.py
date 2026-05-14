"""
LangGraph node function for the Episodic Memory Module.

Maps to: autogluon-assistant's RerankerAgent
  - Takes retrieved tutorial candidates from semantic memory
  - Uses LLM to select the most relevant ones
  - Formats selected tutorials into a tutorial_prompt for the coder
"""

import logging

from .state import EpisodicMemoryState
from shared.llm import get_llm
from shared.logging_config import LLMCallLogger

from .prompts import RERANKER_PROMPT

logger = logging.getLogger(__name__)


def _get_call_logger(state: EpisodicMemoryState) -> LLMCallLogger:
    output_folder = state.get("output_folder", "./output")
    return LLMCallLogger(output_folder)


def rerank_tutorials(state: EpisodicMemoryState) -> dict:
    """
    Episodic Memory: select the most relevant tutorials from retrieved
    candidates, then format them into a tutorial prompt.

    Maps to: RerankerAgent.__call__()

    Input from state:
        tutorial_retrieval (list[TutorialInfo]), task_description, data_prompt,
        user_input, all_error_analyses, config

    Returns:
        {"tutorial_prompt": str}
    """
    logger.info("─── [Episodic Memory] rerank_tutorials: selecting top tutorials ───")

    config = state.get("config", {})
    tutorials_config = config.get("tutorials", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    tutorials = state.get("tutorial_retrieval", [])
    max_num = tutorials_config.get("max_num_tutorials", 3)
    max_length = tutorials_config.get("max_tutorial_length", 30000)
    use_summary = tutorials_config.get("use_tutorial_summary", True)

    if not tutorials:
        logger.warning("  No tutorials to rerank")
        return {"tutorial_prompt": ""}

    # ── Format tutorials info for the LLM ──
    tutorials_info_lines = []
    for i, tutorial in enumerate(tutorials):
        summary_text = tutorial.summary if use_summary and tutorial.summary else "(No summary available)"
        tutorials_info_lines.append(f"{i+1}. Title: {tutorial.title}\n   Summary: {summary_text}")
    tutorials_info = "\n".join(tutorials_info_lines)

    all_errors = "\n\n".join(state.get("all_error_analyses", [])) or "None"

    prompt = RERANKER_PROMPT.format(
        task_description=state.get("task_description", ""),
        data_prompt=state.get("data_prompt", ""),
        user_input=state.get("user_input", ""),
        all_previous_error_analyses=all_errors,
        tutorials_info=tutorials_info,
        max_num_tutorials=max_num,
    )

    response = call_logger.call(llm, prompt, node_name="episodic_memory/rerank_tutorials")

    # ── Parse comma-separated indices from response ──
    content_line = response.strip().split("\n")[0]
    content_clean = "".join(c for c in content_line if c.isdigit() or c == ",")

    selected_tutorials = []
    if content_clean:
        try:
            indices = [int(idx.strip()) - 1 for idx in content_clean.split(",") if idx.strip()]
            for idx in indices:
                if 0 <= idx < len(tutorials):
                    selected_tutorials.append(tutorials[idx])
        except ValueError as e:
            logger.warning(f"  Error parsing tutorial indices: {e}")

    # ── Fallback: top tutorials by retrieval score ──
    if not selected_tutorials:
        logger.warning("  Reranking failed; falling back to top tutorials by score")
        sorted_tutorials = sorted(tutorials, key=lambda t: t.score or 0.0, reverse=True)
        selected_tutorials = sorted_tutorials[:max_num]
    else:
        selected_tutorials = selected_tutorials[:max_num]

    # ── Format selected tutorials into the tutorial prompt ──
    per_tutorial_length = max_length // max(1, len(selected_tutorials))
    formatted_parts = []

    for tutorial in selected_tutorials:
        try:
            with open(tutorial.path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > per_tutorial_length:
                content = content[:per_tutorial_length] + "\n...(truncated)"
            formatted_parts.append(f"### {tutorial.title}\n{content}")
        except Exception as e:
            logger.warning(f"  Error reading tutorial {tutorial.path}: {e}")

    tutorial_prompt = "\n\n".join(formatted_parts) if formatted_parts else ""

    logger.info(f"  Selected {len(selected_tutorials)} tutorials, "
                f"prompt length: {len(tutorial_prompt)} chars")

    return {"tutorial_prompt": tutorial_prompt}
