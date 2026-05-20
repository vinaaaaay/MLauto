"""
LangGraph node functions for the Perception Module.

Each function takes PerceptionState and returns a partial state update dict.
These map 1:1 to the original autogluon-assistant agents:

  scan_data               → DataPerceptionAgent
  find_description_files  → DescriptionFileRetrieverAgent
  generate_task_description → TaskDescriptorAgent
  select_tools            → ToolSelectorAgent

Note: retrieve_tutorials and rerank_tutorials are in their own modules:
  semantic_memory/nodes.py  → RetrieverAgent
  episodic_memory/nodes.py  → RerankerAgent
"""

import json
import logging
import os
import random
import re
import select
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, List, NamedTuple, Optional

from langchain_openai import ChatOpenAI

from .state import PerceptionState
from .prompts import (
    PYTHON_READER_PROMPT,
    DESCRIPTION_FILE_RETRIEVER_PROMPT,
    TASK_DESCRIPTOR_PROMPT,
    TOOL_SELECTOR_PROMPT,
)

logger = logging.getLogger(__name__)


# ─── Inlined: get_llm (from shared/llm.py) ──────────────────────────────────

def _get_llm(config: dict = None) -> ChatOpenAI:
    """
    Create and return a configured ChatOpenAI instance.

    Args:
        config: Optional dict with keys: model, temperature, max_tokens.
                Falls back to sensible defaults.

    Returns:
        A ChatOpenAI instance ready for .invoke() or .ainvoke().
    """
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

    # Reasoning models (o1, o3, gpt-5) have strict parameter rules
    is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])

    if is_reasoning_model:
        logger.info("Detected reasoning model. Forcing temp=1 and using max_completion_tokens.")
        llm = ChatOpenAI(
            model=model,
            temperature=1,  # Must be 1
            max_completion_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    logger.info(f"Initialized OpenAI LLM: model={model}, temp={temperature}")
    return llm


# ─── Inlined: LLMCallLogger + log_state_snapshot (from shared/logging_config.py) ─

class _LLMCallLogger:
    """
    Logs every LLM call (prompt + response) to both:
      - The Python logger (at DEBUG level)
      - A structured JSONL file for post-run analysis
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.logger = logging.getLogger("mlauto.llm")
        self.call_count = 0

    def call(self, llm, prompt: str, node_name: str = "unknown", mcts_node: Any = None) -> str:
        """
        Invoke the LLM, log the full prompt and response, and return the response text.
        """
        self.call_count += 1
        call_id = self.call_count

        self.logger.info(
            f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)"
        )
        self.logger.debug(
            f"[Call #{call_id}] {node_name} — PROMPT:\n"
            f"{'='*60}\n{prompt}\n{'='*60}"
        )

        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        content = response.content

        if mcts_node is not None and hasattr(mcts_node, "ai_call_time"):
            mcts_node.ai_call_time += elapsed

        self.logger.info(
            f"[Call #{call_id}] {node_name} — received response "
            f"({len(content)} chars, {elapsed:.1f}s)"
        )
        self.logger.debug(
            f"[Call #{call_id}] {node_name} — RESPONSE:\n"
            f"{'='*60}\n{content}\n{'='*60}"
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
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write LLM call log: {e}")

        return content


def _log_state_snapshot(state: dict, label: str, output_dir: str) -> None:
    """Save a snapshot of the current state dict to a JSON file."""
    _snap_logger = logging.getLogger("mlauto.state")

    keys_with_values = [k for k, v in state.items() if v]
    _snap_logger.info(f"State snapshot [{label}]: keys with values = {keys_with_values}")

    snapshots_dir = os.path.join(output_dir, "state_snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S")
    safe_label = label.replace(" ", "_").replace("/", "_")
    snapshot_path = os.path.join(snapshots_dir, f"{timestamp}_{safe_label}.json")

    serializable = {}
    for k, v in state.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            if isinstance(v, str) and len(v) > 2000:
                serializable[k] = v[:2000] + f"... [TRUNCATED, total {len(v)} chars]"
            else:
                serializable[k] = v
        else:
            serializable[k] = str(v)

    try:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        _snap_logger.debug(f"State snapshot saved to {snapshot_path}")
    except Exception as e:
        _snap_logger.warning(f"Failed to save state snapshot: {e}")


# ─── Inlined: file helpers (from shared/utils.py) ────────────────────────────

def _get_all_files(folder_path: str) -> list[tuple[str, str]]:
    """
    Recursively get all files in folder_path.

    Returns:
        List of (relative_path, absolute_path) tuples.
    """
    all_files = []
    abs_folder_path = os.path.abspath(folder_path)

    for root, _, files in os.walk(abs_folder_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, abs_folder_path)
            all_files.append((rel_path, abs_path))

    return all_files


def _group_similar_files(files: list[tuple[str, str]]) -> dict:
    """
    Group files by folder structure and extension.

    At each depth level, if there are ≤5 unique folders the actual names are
    used; otherwise a wildcard '*' is substituted.

    Returns:
        Dict mapping group-key tuples to lists of (rel_path, abs_path).
    """
    depth_folders: dict = defaultdict(set)

    for rel_path, _ in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        for depth, folder in enumerate(parts[:-1]):
            depth_folders[depth].add(folder)

    groups: dict = defaultdict(list)
    for rel_path, abs_path in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        folders = parts[:-1]
        filename = parts[-1]
        ext = os.path.splitext(filename)[1].lower()

        group_key_parts = []
        for depth, folder in enumerate(folders):
            if len(depth_folders[depth]) <= 5:
                group_key_parts.append(folder)
            else:
                group_key_parts.append("*")
        group_key_parts.append(ext if ext else "NO_EXT")

        groups[tuple(group_key_parts)].append((rel_path, abs_path))

    return groups


def _pattern_to_path(pattern: tuple, base_path: str) -> str:
    """Convert a group pattern tuple to a display path string."""
    folders = pattern[:-1]
    ext = pattern[-1]

    path_parts = list(str(f) for f in folders)
    path_parts.append("*" if ext == "NO_EXT" else f"*{ext}")

    relative_pattern = os.path.join(*path_parts) if path_parts else "*"
    return os.path.join(base_path, relative_pattern)


def _extract_code(response: str, language: str) -> str:
    """
    Extract a fenced code block from an LLM response.

    Tries ```python or ```bash first, then generic ```, then full response.
    """
    if language == "python":
        pattern = r"```python\s*\n(.*?)```"
    elif language == "bash":
        pattern = r"```bash\s*\n(.*?)```"
    else:
        raise ValueError(f"Unsupported language: {language}")

    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fallback: generic code block
    generic = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if generic:
        logger.warning(f"No {language} block found; using generic code block.")
        return generic[0].strip()

    logger.warning("No code block found; returning full response.")
    return response


def _execute_code(code: str, language: str, timeout: int = 3600) -> tuple[bool, str, str]:
    """
    Execute code with real-time output streaming and timeout.

    Args:
        code: Code string to execute.
        language: "python" or "bash".
        timeout: Maximum seconds before killing the process.

    Returns:
        (success, stdout, stderr)
    """
    if language.lower() == "python":
        cmd = ["python", "-c", code]
    elif language.lower() == "bash":
        cmd = ["bash", "-c", code]
    else:
        return False, "", f"Unsupported language: {language}"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks, stderr_chunks = [], []
        recent_stdout: deque = deque(maxlen=100)
        recent_stderr: deque = deque(maxlen=100)
        streams = [process.stdout, process.stderr]
        start_time = time.time()

        while streams:
            elapsed = time.time() - start_time
            remaining = max(0, timeout - elapsed)

            if remaining <= 0:
                process.terminate()
                time.sleep(3)
                if process.poll() is None:
                    process.kill()
                stdout_chunks.append(f"\nProcess reached time limit after {timeout} seconds.\n")
                logger.info(f"Process reached time limit after {timeout}s.")
                break

            readable, _, _ = select.select(streams, [], [], min(1, remaining))

            if not readable and process.poll() is None:
                continue
            if not readable and process.poll() is not None:
                break

            for stream in readable:
                line = stream.readline()
                if not line:
                    streams.remove(stream)
                    continue

                if stream == process.stdout:
                    if line not in recent_stdout:
                        recent_stdout.append(line)
                        stdout_chunks.append(line)
                else:
                    if line not in recent_stderr:
                        recent_stderr.append(line)
                        stderr_chunks.append(line)

        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_chunks.append("Process forcibly terminated after timeout\n")

        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)

    except Exception as e:
        return False, "", f"Error executing {language} code: {str(e)}"


# ─── Inlined: ToolRegistry (from shared/tool_registry.py) ───────────────────

_DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "tools_registry"


class _TutorialInfo(NamedTuple):
    """Stores information about a tutorial."""
    path: Path
    title: str
    summary: str
    score: Optional[float] = None
    content: Optional[str] = None


class _ToolRegistry:
    """
    Reads the tool catalog and per-tool metadata from disk.

    Usage:
        registry = _ToolRegistry()
        tools = registry.list_tools()
        info  = registry.get_tool("autogluon.tabular")
    """

    def __init__(self, registry_path: str | Path | None = None):
        self.registry_path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY_PATH
        self.catalog_path = self.registry_path / "_common" / "catalog.json"
        self._cache: Optional[dict] = None

    @property
    def tools(self) -> dict:
        if self._cache is None:
            self._load()
        return self._cache

    def _load(self) -> None:
        """Load catalog.json and merge each tool's tool.json."""
        try:
            with open(self.catalog_path, "r") as f:
                catalog = json.load(f)
        except FileNotFoundError:
            logger.warning(f"catalog.json not found at {self.catalog_path}. Using empty registry.")
            self._cache = {}
            return

        tools_info = {}
        for tool_name, tool_data in catalog.get("tools", {}).items():
            tool_dir = self.registry_path / tool_data["path"]
            tool_json_path = tool_dir / "tool.json"

            info = {
                "name": tool_name,
                "path": tool_data["path"],
                "version": tool_data.get("version", "0.0.0"),
                "description": tool_data.get("description", ""),
                "requirements": [],
                "prompt_template": [],
            }

            if tool_json_path.exists():
                try:
                    with open(tool_json_path, "r") as f:
                        tj = json.load(f)
                    info["requirements"] = tj.get("requirements", [])
                    info["prompt_template"] = tj.get("prompt_template", [])
                except Exception as e:
                    logger.warning(f"Error loading tool.json for {tool_name}: {e}")

            req_path = tool_dir / "requirements.txt"
            if req_path.exists():
                try:
                    info["requirements"] = [
                        line.strip() for line in req_path.read_text().splitlines() if line.strip()
                    ]
                except Exception as e:
                    logger.warning(f"Error loading requirements.txt for {tool_name}: {e}")

            tools_info[tool_name] = info

        self._cache = tools_info

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[dict]:
        return self.tools.get(name)

    def get_tool_prompt(self, name: str) -> str:
        """Return the prompt_template for a tool as a single string."""
        tool = self.get_tool(name)
        if not tool:
            return ""
        pt = tool.get("prompt_template", [])
        if isinstance(pt, list):
            return "\n".join(pt)
        return str(pt)

    def get_tool_path(self, name: str) -> Optional[Path]:
        """Get the absolute path for a tool's directory."""
        tool = self.get_tool(name)
        if not tool:
            return None
        return self.registry_path / tool["path"]

    def get_tool_tutorials_folder(self, name: str, condensed: bool = False) -> Path:
        """Get the tutorials folder for a specific tool."""
        tool_path = self.get_tool_path(name)
        if tool_path is None:
            raise FileNotFoundError(f"Tool {name} not found in registry")
        subfolder = "condensed_tutorials" if condensed else "tutorials"
        tutorials_dir = tool_path / subfolder
        if not tutorials_dir.exists():
            raise FileNotFoundError(f"No {subfolder} found for tool {name} at {tutorials_dir}")
        return tutorials_dir

    def get_common_requirements_file(self) -> Path:
        """Get the path to _common/requirements.txt."""
        return self.registry_path / "_common" / "requirements.txt"

    def get_tool_requirements_file(self, name: str) -> Path:
        """Get the path to a tool's requirements.txt."""
        tool_path = self.get_tool_path(name)
        if tool_path is None:
            raise FileNotFoundError(f"Tool {name} not found")
        return tool_path / "requirements.txt"

    def format_tools_info(self) -> str:
        """Format all tools for inclusion in the ToolSelector prompt."""
        lines = []
        for name, info in self.tools.items():
            lines.append(f"Library Name: {name}")
            lines.append(f"Version: v{info['version']}")
            lines.append(f"Description: {info['description']}")
            lines.append("")
        return "\n".join(lines)

# ─── Constants (from autogluon-assistant config defaults) ────────────────
MAX_CHARS_PER_FILE = 768
MAX_FILE_GROUP_SIZE_TO_SHOW = 5
NUM_EXAMPLE_FILES_TO_SHOW = 1
DEFAULT_LIBRARY = "machine learning"


# ─── Helpers ─────────────────────────────────────────────────────────────

def _get_call_logger(state: PerceptionState) -> _LLMCallLogger:
    """Create an _LLMCallLogger pointing to the run's output directory."""
    output_folder = state.get("output_folder", "./output")
    return _LLMCallLogger(output_folder)


def _read_file_via_llm(llm, call_logger: _LLMCallLogger, file_path: str, max_chars: int) -> str:
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
    generated_code = _extract_code(response_text, language="python")

    logger.debug(f"Generated reader code for {file_path}:\n{generated_code}")

    success, stdout, stderr = _execute_code(generated_code, language="python", timeout=60)

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

def scan_data(state: PerceptionState) -> dict:
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
    llm = _get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    abs_folder = os.path.abspath(input_folder)

    # 1. Collect all files
    all_files = _get_all_files(abs_folder)
    logger.info(f"  Found {len(all_files)} files in {abs_folder}")
    for rel, abs_path in all_files:
        size = os.path.getsize(abs_path)
        logger.debug(f"    {rel} ({size:,} bytes)")

    # 2. Group by folder structure + extension
    file_groups = _group_similar_files(all_files)
    logger.info(f"  Grouped into {len(file_groups)} patterns")
    for pattern, group_files in file_groups.items():
        logger.debug(f"    Pattern {pattern}: {len(group_files)} files")

    # 3. Read files via LLM
    file_contents = {}
    for pattern, group_files in file_groups.items():
        pattern_path = _pattern_to_path(pattern, abs_folder)
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

def find_description_files(state: PerceptionState) -> dict:
    """
    Use the LLM to identify description/README files from the data prompt.

    Maps to: DescriptionFileRetrieverAgent.__call__()

    Returns:
        {"description_files": list[str]}
    """
    logger.info("─── find_description_files: identifying description files ───")

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


# ─── Node: generate_task_description ─────────────────────────────────────

def generate_task_description(state: PerceptionState) -> dict:
    """
    Generate a concise task description from data prompt + description files.

    Maps to: TaskDescriptorAgent.__call__()

    Returns:
        {"task_description": str}
    """
    logger.info("─── generate_task_description: building task description ───")

    config = state.get("config", {})
    llm = _get_llm(config.get("llm"))
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

def select_tools(state: PerceptionState) -> dict:
    """
    Select and rank ML tools based on task + data.

    Maps to: ToolSelectorAgent.__call__()

    Returns:
        {"selected_tools": list[str], "current_tool": str, "tool_prompt": str}
    """
    logger.info("─── select_tools: ranking ML libraries ───")

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

    _log_state_snapshot(state, "after_select_tools", state.get("output_folder", "./output"))

    return {
        "selected_tools": prioritized_tools,
        "current_tool": current_tool,
        "tool_prompt": tool_prompt,
    }
