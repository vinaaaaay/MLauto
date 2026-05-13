"""
Tool registry — reads catalog.json + per-tool tool.json files.

Aligned with autogluon-assistant's ToolsRegistry. Provides tool metadata,
prompt templates, requirements files, and tutorial folder paths.
"""

import json
import logging
from pathlib import Path
from typing import List, NamedTuple, Optional

logger = logging.getLogger(__name__)

# Default registry path: MLauto/tools_registry
_DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "tools_registry"


class TutorialInfo(NamedTuple):
    """Stores information about a tutorial."""
    path: Path
    title: str
    summary: str
    score: Optional[float] = None
    content: Optional[str] = None


class ToolRegistry:
    """
    Reads the tool catalog and per-tool metadata from disk.

    Usage:
        registry = ToolRegistry()
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

            # Merge per-tool tool.json if it exists
            if tool_json_path.exists():
                try:
                    with open(tool_json_path, "r") as f:
                        tj = json.load(f)
                    info["requirements"] = tj.get("requirements", [])
                    info["prompt_template"] = tj.get("prompt_template", [])
                except Exception as e:
                    logger.warning(f"Error loading tool.json for {tool_name}: {e}")

            # Override requirements from requirements.txt if present
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

    # ── Public API ──

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
