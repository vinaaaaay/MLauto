"""
LangGraph node function for the Semantic Memory Module.

MCP CLIENT — calls the standalone Semantic Memory MCP server over HTTPS/SSE.

Maps to: autogluon-assistant's RetrieverAgent
  - Connects to the Semantic Memory MCP server
  - Calls the retrieve_tutorials tool
  - Deserializes response back into TutorialInfo objects
"""

import asyncio
import json
import logging
from pathlib import Path

from .state import SemanticMemoryState
from shared.tool_registry import TutorialInfo

logger = logging.getLogger(__name__)

# Default MCP server URL (overridable via config)
_DEFAULT_SERVER_URL = "http://localhost:8010"


async def _call_mcp_retrieve(server_url: str, arguments: dict) -> list[dict]:
    """Connect to the Semantic Memory MCP server and call retrieve_tutorials."""
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    sse_url = f"{server_url.rstrip('/')}/sse"

    async with sse_client(sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                name="retrieve_tutorials",
                arguments=arguments,
            )

            # MCP returns result.content as a list of content blocks
            # For tool results, the first content block contains the text
            if result.content and len(result.content) > 0:
                text = result.content[0].text
                return json.loads(text)
            return []


def retrieve_tutorials(state: SemanticMemoryState) -> dict:
    """
    Semantic Memory: MCP client node — calls the Semantic Memory MCP server.

    Extracts relevant fields from LangGraph state, sends them to the
    MCP server's retrieve_tutorials tool, and deserializes the response
    back into TutorialInfo objects for downstream nodes.

    Input from state:
        task_description, data_prompt, user_input, current_tool,
        all_error_analyses, config

    Returns:
        {"tutorial_retrieval": list[TutorialInfo]}
    """
    logger.info("─── [Semantic Memory] retrieve_tutorials: calling MCP server ───")

    config = state.get("config", {})
    mcp_servers = config.get("mcp_servers", {})
    server_url = mcp_servers.get("semantic_memory_url", _DEFAULT_SERVER_URL)

    # Build arguments for the MCP tool call
    arguments = {
        "task_description": state.get("task_description", ""),
        "data_prompt": state.get("data_prompt", ""),
        "user_input": state.get("user_input", ""),
        "current_tool": state.get("current_tool", ""),
        "all_error_analyses": state.get("all_error_analyses", []),
        "config": config,
        "output_folder": state.get("output_folder", "./output"),
    }

    try:
        raw_tutorials = asyncio.run(_call_mcp_retrieve(server_url, arguments))
    except Exception as e:
        logger.error(f"  MCP call failed: {e}")
        return {"tutorial_retrieval": []}

    # ── Convert dicts back to TutorialInfo objects ──
    tutorials = []
    for t in raw_tutorials:
        try:
            tutorials.append(TutorialInfo(
                path=Path(t["path"]),
                title=t["title"],
                summary=t.get("summary", ""),
                score=t.get("score"),
                content=t.get("content"),
            ))
        except Exception as e:
            logger.warning(f"  Failed to deserialize tutorial: {e}")

    logger.info(f"  Retrieved {len(tutorials)} tutorial candidates from MCP server")
    return {"tutorial_retrieval": tutorials}
