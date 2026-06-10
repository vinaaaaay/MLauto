"""
Helper utilities and abstractions for the Semantic Agent.
Keeps the agent self-contained and independent.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional, List, Dict, Any, TypedDict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════════════════

class TutorialInfo(NamedTuple):
    """Stores information about a tutorial."""
    path: Path
    title: str
    summary: str
    score: Optional[float] = None
    content: Optional[str] = None


class SemanticAgentState(TypedDict):
    """
    State representing the data flow through the Semantic Agent graph.
    Supports a flexible dictionary model to easily support additional parameters.
    """
    # Configuration
    config: Dict[str, Any]
    output_folder: str

    # Context & Inputs
    task_description: str
    data_prompt: str
    user_input: str
    all_error_analyses: List[str]
    current_tool: str

    # Outputs
    search_query: str
    tutorial_retrieval: List[TutorialInfo]
    tutorial_prompt: str


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Call Logger
# ═══════════════════════════════════════════════════════════════════════════

class _LLMCallLogger:
    """Logs every LLM call (prompt + response) to structured JSONL."""

    def __init__(self, output_dir: str, ctx=None, metric_logger: Optional[logging.Logger] = None):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.call_count = 0
        self.ctx = ctx
        self.metric_logger = metric_logger

    def call(self, llm, prompt: str, node_name: str = "unknown") -> str:
        self.call_count += 1
        call_id = self.call_count

        logger.info(f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)")

        start = time.time()
        
        invoke_config = {}
        if self.ctx and self.metric_logger:
            from .common_local import SessionMetricsCallback
            invoke_config = {"callbacks": [SessionMetricsCallback(ctx=self.ctx, metric_logger=self.metric_logger)]}

        response = llm.invoke(prompt, config=invoke_config if invoke_config else None)
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
#  MCP Client
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreMCPClient:
    """
    Client for interacting with the Vector Store MCP Server.
    """
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')

    async def retrieve_tutorials(
        self,
        query: str,
        tool_name: str,
        top_k: int = 5,
        condensed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Call the retrieve_tutorials tool on the MCP server.
        """
        # Try direct HTTP POST first
        import httpx
        try:
            logger.info(f"Attempting direct HTTP POST to {self.server_url}/retrieve_tutorials (timeout=120.0s)...")
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.server_url}/retrieve_tutorials",
                    json={
                        "query": query,
                        "tool_name": tool_name,
                        "top_k": top_k,
                        "condensed": condensed
                    }
                )
                logger.info(f"Direct HTTP POST responded with status_code={resp.status_code}")
                if resp.status_code == 200:
                    return resp.json()
                else:
                    logger.warning(f"Direct HTTP POST non-200 response: {resp.status_code} - {resp.text[:500]}")
        except Exception as e:
            import traceback
            logger.warning(f"Direct HTTP POST call failed: {e!r}. Traceback:\n{traceback.format_exc()}. Falling back to standard MCP SSE...")

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        sse_url = f"{self.server_url}/sse"
        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    name="retrieve_tutorials",
                    arguments={
                        "query": query,
                        "tool_name": tool_name,
                        "top_k": top_k,
                        "condensed": condensed,
                    }
                )
                
                raw_tutorials = []
                if result.content:
                    for block in result.content:
                        try:
                            text_val = block.text.strip()
                            if text_val.startswith("{") or text_val.startswith("["):
                                parsed = json.loads(text_val)
                                if isinstance(parsed, list):
                                    raw_tutorials.extend(parsed)
                                else:
                                    raw_tutorials.append(parsed)
                            else:
                                logger.warning(f"Non-JSON block content ignored: {text_val[:100]}")
                        except Exception as e:
                            logger.warning(f"Failed to parse block as JSON: {e}")
                return raw_tutorials
