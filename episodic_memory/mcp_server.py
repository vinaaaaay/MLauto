"""
Episodic Memory MCP Server — STANDALONE.

Fully self-contained MCP server deployable as a Lambda / Cloud Run function.
Exposes tutorial reranking (LLM-based selection) over HTTPS (SSE).

Zero imports from the `shared/` package or `nodes.py`.
All dependencies (LLM init, call logger, prompts) are inlined.
The server is completely stateless — all tutorial content arrives via the
tool arguments (no disk reads).
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("episodic_memory_mcp")
logging.basicConfig(level=logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════
#  Inlined: LLM Initialization  (from shared/llm.py)
# ═══════════════════════════════════════════════════════════════════════════


def _get_llm(config: dict = None) -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    config = config or {}
    model = config.get("model", "gpt-4o")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 16384)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )

    is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])
    if is_reasoning_model:
        return ChatOpenAI(
            model=model,
            temperature=1,
            max_completion_tokens=max_tokens,
            api_key=api_key,
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Inlined: LLM Call Logger  (from shared/logging_config.py)
# ═══════════════════════════════════════════════════════════════════════════


class _LLMCallLogger:
    """Logs every LLM call (prompt + response) to structured JSONL."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.call_count = 0

    def call(self, llm, prompt: str, node_name: str = "unknown") -> str:
        self.call_count += 1
        call_id = self.call_count

        logger.info(f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)")

        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        content = response.content

        logger.info(
            f"[Call #{call_id}] {node_name} — received response "
            f"({len(content)} chars, {elapsed:.1f}s)"
        )

        record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "prompt_length": len(prompt),
            "response_length": len(content),
            "elapsed_seconds": round(elapsed, 2),
            "prompt": prompt,
            "response": content,
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM call log: {e}")

        return content


# ═══════════════════════════════════════════════════════════════════════════
#  Inlined: Prompt Template  (from prompts.py)
# ═══════════════════════════════════════════════════════════════════════════

_RERANKER_PROMPT = """\
Given the following context and list of tutorials with their summaries, select the {max_num_tutorials} most relevant tutorials for helping with this task. Consider how well each tutorial's title and summary match the task, data, user question, and any errors.

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instruction
{user_input}

### Previous Error Analysis
{all_previous_error_analyses}

Available Tutorials:
{tutorials_info}

IMPORTANT: Respond ONLY with the numbers of the selected tutorials (up to {max_num_tutorials}) separated by commas. 
For example: "1,3,4" or "2,5" or just "1" if only one is relevant.
DO NOT include any other text, explanation, or formatting in your response.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  MCP Server Definition
# ═══════════════════════════════════════════════════════════════════════════

mcp = FastMCP("Episodic Memory Server")


@mcp.tool()
def rerank_tutorials(
    tutorial_retrieval: list[dict],
    task_description: str,
    data_prompt: str = "",
    user_input: str = "",
    all_error_analyses: list[str] = None,
    config: dict = None,
    output_folder: str = "./output",
) -> dict:
    """
    Rerank retrieved tutorials using LLM-based selection and format the best
    ones into a tutorial_prompt for the Coder agent.

    Completely stateless: all tutorial content must be passed in via
    tutorial_retrieval dicts (no disk reads).
    """
    config = config or {}
    all_error_analyses = all_error_analyses or []
    tutorials_config = config.get("tutorials", {})

    logger.info("─── [Episodic Memory Server] rerank_tutorials ───")

    # ── LLM setup ──
    llm = _get_llm(config.get("llm"))
    call_logger = _LLMCallLogger(output_folder)

    max_num = tutorials_config.get("max_num_tutorials", 3)
    max_length = tutorials_config.get("max_tutorial_length", 30000)
    use_summary = tutorials_config.get("use_tutorial_summary", True)

    # Tutorials arrive as plain dicts (JSON-serializable from semantic memory)
    tutorials = tutorial_retrieval or []

    if not tutorials:
        logger.warning("  No tutorials to rerank")
        return {"tutorial_prompt": ""}

    # ── Format tutorials info for the LLM ──
    tutorials_info_lines = []
    for i, tutorial in enumerate(tutorials):
        summary_text = tutorial.get("summary", "") if use_summary else ""
        summary_text = summary_text or "(No summary available)"
        tutorials_info_lines.append(
            f"{i + 1}. Title: {tutorial.get('title', 'Untitled')}\n   Summary: {summary_text}"
        )
    tutorials_info = "\n".join(tutorials_info_lines)

    all_errors = "\n\n".join(all_error_analyses) or "None"

    prompt = _RERANKER_PROMPT.format(
        task_description=task_description,
        data_prompt=data_prompt,
        user_input=user_input,
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
        sorted_tutorials = sorted(tutorials, key=lambda t: t.get("score") or 0.0, reverse=True)
        selected_tutorials = sorted_tutorials[:max_num]
    else:
        selected_tutorials = selected_tutorials[:max_num]

    # ── Format selected tutorials into the tutorial prompt ──
    per_tutorial_length = max_length // max(1, len(selected_tutorials))
    formatted_parts = []

    for tutorial in selected_tutorials:
        try:
            content = tutorial.get("content", "")
            if not content:
                logger.warning(f"  Tutorial '{tutorial.get('title', 'Untitled')}' has no content — skipping")
                continue
            if len(content) > per_tutorial_length:
                content = content[:per_tutorial_length] + "\n...(truncated)"
            formatted_parts.append(f"### {tutorial.get('title', 'Untitled')}\n{content}")
        except Exception as e:
            logger.warning(f"  Error formatting tutorial: {e}")

    tutorial_prompt = "\n\n".join(formatted_parts) if formatted_parts else ""

    logger.info(
        f"  Selected {len(selected_tutorials)} tutorials, "
        f"prompt length: {len(tutorial_prompt)} chars"
    )

    return {"tutorial_prompt": tutorial_prompt}


@mcp.resource("episodic-memory://reranker-config")
def get_reranker_config() -> str:
    """Get the current configuration of the Episodic Memory reranking logic."""
    return (
        "Episodic Memory Reranker Configuration:\n"
        " - Reranks retrieved tutorials using GPT-based selection\n"
        " - Falls back to retrieval score ranking if LLM parsing fails\n"
        " - Limits the total tutorial length to prevent context explosion\n"
        " - Completely stateless: all content arrives via tool arguments"
    )


@mcp.prompt("tutorial-reranker")
def get_tutorial_reranker_prompt() -> str:
    """Get the prompt template used for LLM tutorial selection and reranking."""
    return _RERANKER_PROMPT


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI App  (HTTPS/SSE mount)
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Episodic Memory MCP Server")
app.mount("/", mcp.sse_app())
