"""
LangGraph node functions for the Perception Module.

Each function takes MLAutoState and returns a partial state update dict.
These map 1:1 to the original autogluon-assistant agents:

  scan_data             → DataPerceptionAgent
  find_description_files → DescriptionFileRetrieverAgent
  generate_task_description → TaskDescriptorAgent
  select_tools          → ToolSelectorAgent
"""

import logging
import os
import random
import re

from shared.state import MLAutoState
from shared.llm import get_llm
from shared.logging_config import LLMCallLogger, log_state_snapshot
from shared.utils import (
    get_all_files,
    group_similar_files,
    pattern_to_path,
    extract_code,
    execute_code,
)
from shared.tool_registry import ToolRegistry

from .prompts import (
    PYTHON_READER_PROMPT,
    DESCRIPTION_FILE_RETRIEVER_PROMPT,
    TASK_DESCRIPTOR_PROMPT,
    TOOL_SELECTOR_PROMPT,
)

logger = logging.getLogger(__name__)

# ─── Constants (from autogluon-assistant config defaults) ────────────────
MAX_CHARS_PER_FILE = 768
MAX_FILE_GROUP_SIZE_TO_SHOW = 5
NUM_EXAMPLE_FILES_TO_SHOW = 1
DEFAULT_LIBRARY = "machine learning"


# ─── Helpers ─────────────────────────────────────────────────────────────

def _get_call_logger(state: MLAutoState) -> LLMCallLogger:
    """Create an LLMCallLogger pointing to the run's output directory."""
    output_folder = state.get("output_folder", "./output")
    return LLMCallLogger(output_folder)


def _read_file_via_llm(llm, call_logger: LLMCallLogger, file_path: str, max_chars: int) -> str:
    """
    Use the LLM to generate a Python script that reads & summarizes a file,
    then execute that script and return stdout.

    This mirrors DataPerceptionAgent.read_file().
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    prompt = PYTHON_READER_PROMPT.format(
        file_path=file_path,
        file_size_mb=f"{file_size_mb:.2f}",
        max_chars=max_chars,
    )

    response_text = call_logger.call(llm, prompt, node_name=f"scan_data/read_file({os.path.basename(file_path)})")
    generated_code = extract_code(response_text, language="python")

    logger.debug(f"Generated reader code for {file_path}:\n{generated_code}")

    success, stdout, stderr = execute_code(generated_code, language="python", timeout=60)

    if stdout:
        result = stdout
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        logger.debug(f"File read OK: {file_path} ({len(result)} chars)")
    else:
        logger.error(f"Error reading file {file_path}: {stderr}")
        result = f"Error reading file: {stderr}"

    return result


# ─── Node: scan_data ─────────────────────────────────────────────────────

def scan_data(state: MLAutoState) -> dict:
    """
    Scan the input data folder, group similar files, and use the LLM to
    read/summarize each file's content.

    Maps to: DataPerceptionAgent.__call__()

    Returns:
        {"data_prompt": str}
    """
    logger.info("─── scan_data: scanning data folder and grouping files ───")

    input_folder = state["input_data_folder"]
    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    abs_folder = os.path.abspath(input_folder)

    # 1. Collect all files
    all_files = get_all_files(abs_folder)
    logger.info(f"  Found {len(all_files)} files in {abs_folder}")
    for rel, abs_path in all_files:
        size = os.path.getsize(abs_path)
        logger.debug(f"    {rel} ({size:,} bytes)")

    # 2. Group by folder structure + extension
    file_groups = group_similar_files(all_files)
    logger.info(f"  Grouped into {len(file_groups)} patterns")
    for pattern, group_files in file_groups.items():
        logger.debug(f"    Pattern {pattern}: {len(group_files)} files")

    # 3. Read files via LLM
    file_contents = {}
    for pattern, group_files in file_groups.items():
        pattern_path = pattern_to_path(pattern, abs_folder)
        logger.info(f"  Processing pattern: {pattern_path} ({len(group_files)} files)")

        if len(group_files) > MAX_FILE_GROUP_SIZE_TO_SHOW:
            num_examples = min(NUM_EXAMPLE_FILES_TO_SHOW, len(group_files))
            example_files = random.sample(group_files, num_examples)

            group_info = (
                f"Group pattern: {pattern_path} (total {len(group_files)} files)\n"
                "Example files:"
            )
            example_contents = []
            for rel_path, abs_path in example_files:
                logger.info(f"    Reading example: {abs_path}")
                content = _read_file_via_llm(llm, call_logger, abs_path, MAX_CHARS_PER_FILE)
                example_contents.append(f"Absolute path: {abs_path}\nContent:\n{content}")

            file_contents[group_info] = "\n-----\n".join(example_contents)
        else:
            for rel_path, abs_path in group_files:
                file_info = f"Absolute path: {abs_path}"
                logger.info(f"    Reading: {abs_path}")
                file_contents[file_info] = _read_file_via_llm(llm, call_logger, abs_path, MAX_CHARS_PER_FILE)

    # 4. Assemble the data prompt
    separator = "-" * 10
    data_prompt = f"Absolute path to the folder: {abs_folder}\n\nFiles structures:\n\n{separator}\n\n"
    for info, content in file_contents.items():
        data_prompt += f"{info}\nContent:\n{content}\n{separator}\n"

    logger.info(f"  data_prompt assembled: {len(data_prompt)} chars")
    logger.debug(f"  data_prompt content:\n{data_prompt[:1000]}...")
    return {"data_prompt": data_prompt}


# ─── Node: find_description_files ────────────────────────────────────────

def find_description_files(state: MLAutoState) -> dict:
    """
    Use the LLM to identify description/README files from the data prompt.

    Maps to: DescriptionFileRetrieverAgent.__call__()

    Returns:
        {"description_files": list[str]}
    """
    logger.info("─── find_description_files: identifying description files ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    prompt = DESCRIPTION_FILE_RETRIEVER_PROMPT.format(data_prompt=state["data_prompt"])

    content = call_logger.call(llm, prompt, node_name="find_description_files")

    # Parse: look for "Description Files:" section and extract paths
    description_files = []
    in_section = False
    for line in content.split("\n"):
        stripped = line.strip()
        if "description files:" in stripped.lower():
            in_section = True
            continue
        if in_section and stripped:
            filename = stripped.strip("- []").strip()
            if filename:
                description_files.append(filename)

    logger.info(f"  Found {len(description_files)} description files:")
    for f in description_files:
        logger.info(f"    → {f}")

    return {"description_files": description_files}


# ─── Node: generate_task_description ─────────────────────────────────────

def generate_task_description(state: MLAutoState) -> dict:
    """
    Generate a concise task description from data prompt + description files.

    Maps to: TaskDescriptorAgent.__call__()

    Returns:
        {"task_description": str}
    """
    logger.info("─── generate_task_description: building task description ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    # Read description file contents
    file_contents = []
    for filepath in state.get("description_files", []):
        try:
            with open(filepath, "r") as f:
                content = f.read()
                file_contents.append(content)
                logger.info(f"  Read {filepath} ({len(content)} chars)")
        except Exception as e:
            logger.warning(f"  Could not read {filepath}: {e}")

    description_file_contents = (
        "\n\n".join(file_contents) if file_contents
        else "No description file contents could be read."
    )

    user_input = state.get("user_input", "")

    prompt = TASK_DESCRIPTOR_PROMPT.format(
        user_input=user_input,
        data_prompt=state["data_prompt"],
        description_file_contents=description_file_contents,
    )

    response_text = call_logger.call(llm, prompt, node_name="generate_task_description")
    task_description = response_text.strip() or "Failed to generate task description."

    logger.info(f"  Task description ({len(task_description)} chars):")
    logger.info(f"  {task_description[:300]}...")

    return {"task_description": task_description}


# ─── Node: select_tools ─────────────────────────────────────────────────

def select_tools(state: MLAutoState) -> dict:
    """
    Select and rank ML tools based on task + data.

    Maps to: ToolSelectorAgent.__call__()

    Returns:
        {"selected_tools": list[str], "current_tool": str, "tool_prompt": str}
    """
    logger.info("─── select_tools: ranking ML libraries ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    registry = ToolRegistry(config.get("tool_registry_path"))
    tools_info = registry.format_tools_info()
    logger.debug(f"  Available tools:\n{tools_info}")

    prompt = TOOL_SELECTOR_PROMPT.format(
        task_description=state["task_description"],
        data_prompt=state["data_prompt"],
        tools_info=tools_info,
    )

    content = call_logger.call(llm, prompt, node_name="select_tools")

    # Parse ranked libraries
    ranked_section = re.search(r"RANKED_LIBRARIES:(.*?)$", content, re.IGNORECASE | re.DOTALL)
    prioritized_tools = []
    available_names = set(registry.list_tools())

    if ranked_section:
        items = re.findall(r"^\s*\d+\.\s*(.+?)$", ranked_section.group(1), re.MULTILINE)
        for item in items:
            name = item.strip()
            if name in available_names:
                prioritized_tools.append(name)
            else:
                # Closest match fallback
                closest = min(available_names, key=lambda x: len(set(x.lower()) ^ set(name.lower())))
                logger.warning(f"  Tool '{name}' not found; using closest: '{closest}'")
                prioritized_tools.append(closest)

    if not prioritized_tools:
        logger.warning(f"  Could not parse tools from LLM response. Defaulting to '{DEFAULT_LIBRARY}'.")
        prioritized_tools = [DEFAULT_LIBRARY]

    current_tool = prioritized_tools[0]
    tool_prompt = registry.get_tool_prompt(current_tool)

    logger.info(f"  Ranked tools: {prioritized_tools}")
    logger.info(f"  Selected tool: {current_tool}")
    logger.debug(f"  Tool prompt ({len(tool_prompt)} chars): {tool_prompt[:200]}...")

    log_state_snapshot(state, "after_select_tools", state.get("output_folder", "./output"))

    return {
        "selected_tools": prioritized_tools,
        "current_tool": current_tool,
        "tool_prompt": tool_prompt,
    }
