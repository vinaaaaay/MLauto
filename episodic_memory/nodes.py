"""
LangGraph node function for the Episodic Memory Module.

MCP CLIENT — calls the standalone Episodic Memory MCP server over HTTPS/SSE.

Maps to: autogluon-assistant's RerankerAgent
  - Connects to the Episodic Memory MCP server
  - Serializes TutorialInfo objects to dicts for JSON transport
  - Calls the rerank_tutorials tool
  - Returns the tutorial_prompt string
"""

import asyncio
import json
import logging

from .state import EpisodicMemoryState

logger = logging.getLogger(__name__)

# Default MCP server URL (overridable via config)
_DEFAULT_SERVER_URL = "http://localhost:8011"


async def _call_mcp_rerank(server_url: str, arguments: dict) -> dict:
    """Connect to the Episodic Memory MCP server and call rerank_tutorials."""
    # pyrefly: ignore [missing-import]
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    sse_url = f"{server_url.rstrip('/')}/sse"

    async with sse_client(sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                name="rerank_tutorials",
                arguments=arguments,
            )

            # MCP returns result.content as a list of content blocks
            if result.content and len(result.content) > 0:
                text = result.content[0].text
                return json.loads(text)
            return {"tutorial_prompt": ""}


def _serialize_tutorials(tutorials: list) -> list[dict]:
    """Convert TutorialInfo NamedTuple objects to JSON-compatible dicts."""
    serialized = []
    for t in tutorials:
        try:
            # Handle both NamedTuple (TutorialInfo) and dict inputs
            if isinstance(t, dict):
                serialized.append(t)
            else:
                serialized.append({
                    "path": str(getattr(t, "path", "")),
                    "title": getattr(t, "title", ""),
                    "summary": getattr(t, "summary", ""),
                    "score": getattr(t, "score", None),
                    "content": getattr(t, "content", None),
                })
        except Exception as e:
            logger.warning(f"  Failed to serialize tutorial: {e}")
    return serialized


def rerank_tutorials(state: EpisodicMemoryState) -> dict:
    """
    Episodic Memory: MCP client node — calls the Episodic Memory MCP server.

    Serializes TutorialInfo objects from state into dicts, sends them to
    the MCP server's rerank_tutorials tool, and returns the tutorial_prompt.

    Input from state:
        tutorial_retrieval (list[TutorialInfo]), task_description, data_prompt,
        user_input, all_error_analyses, config

    Returns:
        {"tutorial_prompt": str}
    """
    logger.info("─── [Episodic Memory] rerank_tutorials: calling MCP server ───")

    config = state.get("config", {})
    mcp_servers = config.get("mcp_servers", {})
    server_url = mcp_servers.get("episodic_memory_url", _DEFAULT_SERVER_URL)

    # Serialize TutorialInfo objects to dicts for JSON transport
    tutorials = state.get("tutorial_retrieval", [])
    serialized_tutorials = _serialize_tutorials(tutorials)

    # Build arguments for the MCP tool call
    arguments = {
        "tutorial_retrieval": serialized_tutorials,
        "task_description": state.get("task_description", ""),
        "data_prompt": state.get("data_prompt", ""),
        "user_input": state.get("user_input", ""),
        "all_error_analyses": state.get("all_error_analyses", []),
        "config": config,
        "output_folder": state.get("output_folder", "./output"),
    }

    try:
        result = asyncio.run(_call_mcp_rerank(server_url, arguments))
    except Exception as e:
        logger.error(f"  MCP call failed: {e}")
        return {"tutorial_prompt": ""}

    tutorial_prompt = result.get("tutorial_prompt", "")
    logger.info(f"  Received tutorial_prompt ({len(tutorial_prompt)} chars) from MCP server")

    return {"tutorial_prompt": tutorial_prompt}
