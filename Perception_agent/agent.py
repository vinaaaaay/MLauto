"""
Single build agent compiling the Perception Agent StateGraph with inline nodes.

Graph flow:
  START → scan_data → find_description_files → generate_task_description
        → select_tools → call_semantic_memory_agent → END

The first 4 nodes perform local perception (scan data, identify descriptions,
generate task summary, rank ML tools). The final node calls the Semantic
Memory Agent via A2A to retrieve and rerank tutorials.

This agent is completely self-contained — deployable as a standalone
black-box service (e.g. in a Lambda).
"""

import logging
import os
import random
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from state import PerceptionAgentState
from prompts import (
    PYTHON_READER_PROMPT,
    DESCRIPTION_FILE_RETRIEVER_PROMPT,
    TASK_DESCRIPTOR_PROMPT,
    TOOL_SELECTOR_PROMPT,
)
from utils import (
    _get_llm,
    _LLMCallLogger,
    _log_state_snapshot,
    _get_all_files,
    _get_sandbox_client,
    _get_all_files_sandbox,
    _execute_code_sandbox,
    _group_similar_files,
    _pattern_to_path,
    _extract_code,
    _execute_code,
    _ToolRegistry,
    MAX_CHARS_PER_FILE,
    MAX_FILE_GROUP_SIZE_TO_SHOW,
    NUM_EXAMPLE_FILES_TO_SHOW,
    DEFAULT_LIBRARY,
)

logger = logging.getLogger(__name__)

from telemetry.metrics_context import MetricsContext
from telemetry.metrics_emitter import node_metrics


metric_logger = logging.getLogger("agent_metrics")
ctx = MetricsContext(agent_id="perception_agent")



def build_perception_agent_graph(ctx=None, metric_logger=None):
    """
    Build and compile the Perception Agent LangGraph.

    Contains all graph nodes inline to encapsulate execution scope,
    matching the single build agent design pattern.

    """
    active_ctx = ctx or globals().get("ctx")
    active_logger = metric_logger or globals().get("metric_logger")

    # ─── Helper ──────────────────────────────────────────────────────────

    def _get_call_logger(state: PerceptionAgentState) -> _LLMCallLogger:
        """Create an _LLMCallLogger pointing to the run's output directory."""
        output_folder = state.get("output_folder", "./output")
        return _LLMCallLogger(output_folder, ctx=active_ctx, metric_logger=active_logger)

    def _read_file_via_llm(llm, call_logger: _LLMCallLogger, file_path: str, max_chars: int, file_size: int, sandbox) -> str:
        """
        Use the LLM to generate a Python script that reads & summarizes a file,
        then execute that script inside the sandbox and return stdout.

        Mirrors DataPerceptionAgent.read_file() from MLauto.
        """
        file_size_mb = file_size / (1024 * 1024)

        prompt = PYTHON_READER_PROMPT.format(
            file_path=file_path,
            file_size_mb=f"{file_size_mb:.2f}",
            max_chars=max_chars,
        )

        response_text = call_logger.call(
            llm, prompt,
            node_name=f"scan_data/read_file({os.path.basename(file_path)})"
        )
        generated_code = _extract_code(response_text, language="python")

        logger.debug(f"Generated reader code for {file_path}:\n{generated_code}")

        success, stdout, stderr = _execute_code_sandbox(generated_code, language="python", sandbox=sandbox, timeout=60)

        if stdout:
            result = stdout
            if len(result) > max_chars:
                result = result[:max_chars - 3] + "..."
            logger.debug(f"File read OK: {file_path} ({len(result)} chars)")
        else:
            logger.error(f"Error reading file {file_path}: {stderr}")
            result = f"Error reading file: {stderr}"

        return result

    # ─── Node 1: scan_data ───────────────────────────────────────────────

    @node_metrics(active_ctx, active_logger, "scan_data")
    def scan_data(state: PerceptionAgentState) -> dict:
        """
        Scan the input data folder, group similar files, and use the LLM to
        read/summarize each file's content.

        Maps to: DataPerceptionAgent.__call__()

        Returns:
            {"data_prompt": str}
        """
        logger.info("─── [Perception Agent] scan_data ───")

        input_folder = state["input_data_folder"]
        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))
        call_logger = _get_call_logger(state)

        sandbox = _get_sandbox_client(config)

        # 1. Collect all files from the sandbox
        all_files_with_sizes = _get_all_files_sandbox(input_folder, sandbox)
        all_files = [(rel, abs_path) for rel, abs_path, _ in all_files_with_sizes]
        file_sizes = {abs_path: size for _, abs_path, size in all_files_with_sizes}

        logger.info(f"  Found {len(all_files)} files in sandbox folder {input_folder}")
        for rel, abs_path in all_files:
            size = file_sizes.get(abs_path, 0)
            logger.debug(f"    {rel} ({size:,} bytes)")

        # 2. Group by folder structure + extension
        file_groups = _group_similar_files(all_files)
        logger.info(f"  Grouped into {len(file_groups)} patterns")
        for pattern, group_files in file_groups.items():
            logger.debug(f"    Pattern {pattern}: {len(group_files)} files")

        # 3. Read files via LLM
        file_contents = {}
        for pattern, group_files in file_groups.items():
            pattern_path = _pattern_to_path(pattern, input_folder)
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
                    size = file_sizes.get(abs_path, 0)
                    content = _read_file_via_llm(llm, call_logger, abs_path, MAX_CHARS_PER_FILE, size, sandbox)
                    example_contents.append(f"Absolute path: {abs_path}\nContent:\n{content}")

                file_contents[group_info] = "\n-----\n".join(example_contents)
            else:
                for rel_path, abs_path in group_files:
                    file_info = f"Absolute path: {abs_path}"
                    logger.info(f"    Reading: {abs_path}")
                    size = file_sizes.get(abs_path, 0)
                    file_contents[file_info] = _read_file_via_llm(llm, call_logger, abs_path, MAX_CHARS_PER_FILE, size, sandbox)

        # 4. Assemble the data prompt
        separator = "-" * 10
        data_prompt = f"Absolute path to the folder: {input_folder}\n\nFiles structures:\n\n{separator}\n\n"
        for info, content in file_contents.items():
            data_prompt += f"{info}\nContent:\n{content}\n{separator}\n"

        logger.info(f"  data_prompt assembled: {len(data_prompt)} chars")
        logger.debug(f"  data_prompt content:\n{data_prompt[:1000]}...")
        return {"data_prompt": data_prompt}

    # ─── Node 2: find_description_files ──────────────────────────────────

    @node_metrics(active_ctx, active_logger, "find_description_files")
    def find_description_files(state: PerceptionAgentState) -> dict:
        """
        Use the LLM to identify description/README files from the data prompt.

        Maps to: DescriptionFileRetrieverAgent.__call__()

        Returns:
            {"description_files": list[str]}
        """
        logger.info("─── [Perception Agent] find_description_files ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))
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

    # ─── Node 3: generate_task_description ───────────────────────────────

    @node_metrics(active_ctx, active_logger, "generate_task_description")
    def generate_task_description(state: PerceptionAgentState) -> dict:
        """
        Generate a concise task description from data prompt + description files.

        Maps to: TaskDescriptorAgent.__call__()

        Returns:
            {"task_description": str}
        """
        logger.info("─── [Perception Agent] generate_task_description ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))
        call_logger = _get_call_logger(state)

        # Read description file contents from the sandbox
        sandbox = _get_sandbox_client(config)
        file_contents = []
        for filepath in state.get("description_files", []):
            try:
                content = sandbox.read_file_sync(filepath)
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

    # ─── Node 4: select_tools ────────────────────────────────────────────

    @node_metrics(active_ctx, active_logger, "select_tools")
    def select_tools(state: PerceptionAgentState) -> dict:
        """
        Select and rank ML tools based on task + data.

        Maps to: ToolSelectorAgent.__call__()

        Returns:
            {"selected_tools": list[str], "current_tool": str, "tool_prompt": str}
        """
        logger.info("─── [Perception Agent] select_tools ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))
        call_logger = _get_call_logger(state)

        registry = _ToolRegistry(config.get("tool_registry_path"))
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
                elif available_names:
                    # Closest match fallback
                    closest = min(available_names, key=lambda x: len(set(x.lower()) ^ set(name.lower())))
                    logger.warning(f"  Tool '{name}' not found; using closest: '{closest}'")
                    prioritized_tools.append(closest)
                else:
                    logger.warning(f"  Tool '{name}' not found and registry is empty; using name as is")
                    prioritized_tools.append(name)

        if not prioritized_tools:
            logger.warning(f"  Could not parse tools from LLM response. Defaulting to '{DEFAULT_LIBRARY}'.")
            prioritized_tools = [DEFAULT_LIBRARY]

        current_tool = prioritized_tools[0]
        tool_prompt = registry.get_tool_prompt(current_tool)

        logger.info(f"  Ranked tools: {prioritized_tools}")
        logger.info(f"  Selected tool: {current_tool}")
        logger.debug(f"  Tool prompt ({len(tool_prompt)} chars): {tool_prompt[:200]}...")

        _log_state_snapshot(state, "after_select_tools", state.get("output_folder", "./output"))

        return {
            "selected_tools": prioritized_tools,
            "current_tool": current_tool,
            "tool_prompt": tool_prompt,
        }

    # ─── Graph Assembly ──────────────────────────────────────────────────

    graph = StateGraph(PerceptionAgentState)

    graph.add_node("scan_data", scan_data)
    graph.add_node("find_description_files", find_description_files)
    graph.add_node("generate_task_description", generate_task_description)
    graph.add_node("select_tools", select_tools)

    graph.add_edge(START, "scan_data")
    graph.add_edge("scan_data", "find_description_files")
    graph.add_edge("find_description_files", "generate_task_description")
    graph.add_edge("generate_task_description", "select_tools")
    graph.add_edge("select_tools", END)

    return graph.compile()
