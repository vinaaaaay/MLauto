"""
LangGraph node function for the Semantic Memory Module.

Maps to: autogluon-assistant's RetrieverAgent
  - Generates a search query via LLM
  - Performs FAISS semantic search over tool tutorials
  - Returns list of TutorialInfo candidates
"""

import logging
import os

from .state import SemanticMemoryState
from shared.llm import get_llm
from shared.logging_config import LLMCallLogger
from shared.tool_registry import TutorialInfo
from shared.tutorial_indexer import TutorialIndexer

from .prompts import RETRIEVER_PROMPT

logger = logging.getLogger(__name__)


def _get_call_logger(state: SemanticMemoryState) -> LLMCallLogger:
    output_folder = state.get("output_folder", "./output")
    return LLMCallLogger(output_folder)


def retrieve_tutorials(state: SemanticMemoryState) -> dict:
    """
    Semantic Memory: generate a search query via LLM, then perform
    FAISS + BGE semantic search over tool tutorials.

    Maps to: RetrieverAgent.__call__()

    Input from state:
        task_description, data_prompt, user_input, current_tool,
        all_error_analyses, config

    Returns:
        {"tutorial_retrieval": list[TutorialInfo]}
    """
    logger.info("─── [Semantic Memory] retrieve_tutorials: generating search query ───")

    config = state.get("config", {})
    tutorials_config = config.get("tutorials", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    current_tool = state.get("current_tool", "")
    top_k = tutorials_config.get("num_tutorial_retrievals", 10)
    condense = tutorials_config.get("condense_tutorials", True)

    # ── Initialize tutorial indexer ──
    registry_path = config.get("tool_registry_path")
    indexer = TutorialIndexer(registry_path=registry_path)

    try:
        loaded = indexer.load_indices()
        if not loaded:
            logger.info("  Building tutorial indices for the first time...")
            indexer.build_indices()
            indexer.save_indices()
            logger.info("  Tutorial indices built and saved.")
    except Exception as e:
        logger.error(f"  Error initializing tutorial indexer: {e}")
        return {"tutorial_retrieval": []}

    # ── Generate search query via LLM ──
    all_errors = "\n\n".join(state.get("all_error_analyses", [])) or "None"

    prompt = RETRIEVER_PROMPT.format(
        task_description=state.get("task_description", ""),
        data_prompt=state.get("data_prompt", ""),
        user_input=state.get("user_input", ""),
        all_previous_error_analyses=all_errors,
        selected_tool=current_tool,
    )

    response = call_logger.call(llm, prompt, node_name="semantic_memory/retrieve_tutorials")

    # ── Parse search query from LLM response ──
    search_query = response.strip().split("\n")[0].strip().strip("\"'")

    # Remove unwanted prefixes (mirrors RetrieverPrompt.parse())
    for prefix in ["search query:", "query:", "the search query is:"]:
        if search_query.lower().startswith(prefix):
            search_query = search_query[len(prefix):].strip()
            break

    if not search_query:
        search_query = state.get("task_description", current_tool)[:256]
        logger.warning("  Failed to generate search query; using task description fallback.")

    if len(search_query) > 512:
        search_query = search_query[:512]

    logger.info(f"  Search query: '{search_query}'")

    # ── Perform FAISS semantic search ──
    results = indexer.search(
        query=search_query,
        tool_name=current_tool,
        condensed=condense,
        top_k=top_k,
    )

    # ── Convert to TutorialInfo objects ──
    tutorials = []
    for result in results:
        try:
            file_path = result["file_path"]
            content = result["content"]
            score = result["score"]

            lines = content.split("\n")
            title = next(
                (line.lstrip("#").strip() for line in lines if line.strip().startswith("#")),
                os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title(),
            )
            summary = next(
                (line.replace("Summary:", "").strip() for line in lines if line.strip().startswith("Summary:")),
                "",
            )

            tutorials.append(TutorialInfo(
                path=file_path,
                title=title,
                summary=summary,
                score=score,
                content=content,
            ))
        except Exception as e:
            logger.warning(f"  Error converting search result: {e}")

    logger.info(f"  Retrieved {len(tutorials)} tutorial candidates")
    indexer.cleanup()

    return {"tutorial_retrieval": tutorials}
