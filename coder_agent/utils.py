import json
import logging
import os
import re
import urllib.request
import urllib.error
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)


# ── State definition ──

class CoderAgentState(TypedDict, total=False):
    # ── Context Inputs ──
    task_description: str
    data_prompt: str
    user_input: str
    current_tool: str
    tool_prompt: str
    tutorial_prompt: str
    all_error_analyses: List[str]

    # ── Run Configuration ──
    config: Dict[str, Any]
    output_folder: str
    sandbox_client: Any

    # ── Current iteration tracking ──
    iteration: int
    node_id: int
    stage: str  # "root", "evolve", or "debug"

    # Previous attempts (if improving/debugging)
    previous_python_code: str
    previous_bash_script: str

    # ── Outputs ──
    python_code: str
    python_file_path: str
    bash_script: str
    stdout: str
    stderr: str
    decision: str  # "SUCCESS" or "FIX"
    error_summary: Optional[str]
    validation_score: Optional[float]
    error_message: str


# ── Sandbox Client ──





# ── Code Extraction ──

def extract_code(response: str, language: str) -> str:
    """Extract a fenced code block from an LLM response."""
    if language == "python":
        pattern = r"```python\s*\n(.*?)```"
    elif language == "bash":
        pattern = r"```bash\s*\n(.*?)```"
    else:
        raise ValueError(f"Unsupported language: {language}")

    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    generic = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if generic:
        logger.warning(f"No {language} block found; using generic code block.")
        return generic[0].strip()

    logger.warning(f"No code block found; returning full response.")
    return response.strip()


# ── Requirements Resolver ──

def get_requirements_contents(registry_path: str, tool_name: str) -> Tuple[str, str]:
    """Reads requirements_common.txt and tool-specific requirements.txt from host registry path."""
    common_content = ""
    tool_content = ""
    if not registry_path:
        return common_content, tool_content
    
    reg_path = Path(registry_path)
    common_file = reg_path / "_common" / "requirements.txt"
    if common_file.exists():
        try:
            with open(common_file, "r") as f:
                common_content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read common requirements: {e}")
            
    catalog_file = reg_path / "_common" / "catalog.json"
    if catalog_file.exists() and tool_name:
        try:
            with open(catalog_file, "r") as f:
                catalog = json.load(f)
            tool_data = catalog.get("tools", {}).get(tool_name)
            if tool_data and "path" in tool_data:
                tool_req_file = reg_path / tool_data["path"] / "requirements.txt"
                if tool_req_file.exists():
                    with open(tool_req_file, "r") as f:
                        tool_content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read tool requirements for {tool_name}: {e}")
            
    return common_content, tool_content
