"""
Simplified tool registry — reads the catalog.json + per-tool tool.json
files from the autogluon-assistant tools_registry directory.

This replaces the full ToolsRegistry class but keeps the same data format
so the ToolSelector prompt can list available tools.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default registry path: points to the autogluon-assistant tools_registry
_DEFAULT_REGISTRY_PATH = (
    Path(__file__).parent.parent.parent
    / "autogluon-assistant"
    / "src"
    / "autogluon"
    / "assistant"
    / "tools_registry"
)


class ToolRegistry:
    """
    Reads the tool catalog and per-tool metadata from disk.

    Usage:
        registry = ToolRegistry()            # uses default path
        registry = ToolRegistry("/custom")   # custom path
        tools = registry.list_tools()        # ['autogluon.tabular', ...]
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

    def list_tools(self) -> list[str]:
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

    def format_tools_info(self) -> str:
        """Format all tools for inclusion in the ToolSelector prompt."""
        lines = []
        for name, info in self.tools.items():
            lines.append(f"Library Name: {name}")
            lines.append(f"Version: v{info['version']}")
            lines.append(f"Description: {info['description']}")
            lines.append("")
        return "\n".join(lines)
