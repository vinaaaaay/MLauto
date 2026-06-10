"""
Single build agent compiling the Semantic Agent StateGraph with inline nodes.
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from .utils import SemanticAgentState, TutorialInfo, _LLMCallLogger, VectorStoreMCPClient
from .prompts import _QUERY_GENERATOR_PROMPT, _RERANKER_PROMPT
from .common_local import MetricsContext, node_metrics, SessionMetricsCallback

logger = logging.getLogger(__name__)

metric_logger = logging.getLogger("agent_metrics")
ctx = MetricsContext(agent_id="semantic_agent")


def build_semantic_agent_graph(ctx=None, metric_logger=None):
    """
    Build and compile the Semantic Agent LangGraph.
    
    Contains all graph nodes inline to encapsulate execution scope,
    matching the single build agent design pattern.
    """
    active_ctx = ctx or globals().get("ctx")
    active_logger = metric_logger or globals().get("metric_logger")
    
    def _init_llm(llm_config: dict) -> ChatOpenAI:
        """Helper to initialize ChatOpenAI directly from config."""
        model = llm_config.get("model", "gpt-4o")
        temperature = llm_config.get("temperature", 0.1)
        max_tokens = llm_config.get("max_tokens", 16384)

        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Neither OPENROUTER_API_KEY nor OPENAI_API_KEY environment variable is set."
            )

        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        if not api_base:
            if os.environ.get("OPENROUTER_API_KEY") or api_key.startswith("sk-or-"):
                api_base = "https://openrouter.ai/api/v1"

        is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])

        if is_reasoning_model:
            logger.info("Detected reasoning model. Forcing temp=1 and using max_completion_tokens.")
            return ChatOpenAI(
                model=model,
                temperature=1,
                max_completion_tokens=max_tokens,
                api_key=api_key,
                openai_api_base=api_base,
            )
        
        logger.info(f"Initialized OpenAI LLM: model={model}, temp={temperature}, base_url={api_base}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            openai_api_base=api_base,
        )

    @node_metrics(active_ctx, active_logger, "generate_query")
    def generate_query(state: SemanticAgentState) -> dict:
        """LLM node to generate a search query from the agent state."""
        logger.info("─── [Semantic Agent] generate_query ───")

        config = state.get("config", {})
        llm_config = config.get("llm", {}).copy()
        
        # Default to 'gpt-4o-mini' if not specified
        if "model" not in llm_config:
            llm_config["model"] = "gpt-4o-mini"

        try:
            llm = _init_llm(llm_config)
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return {"search_query": state.get("task_description", "")[:256]}

        # Format the prompt
        task_desc = state.get("task_description", "")
        data_prompt = state.get("data_prompt", "")
        user_input = state.get("user_input", "")
        all_error_analyses = "\n\n".join(state.get("all_error_analyses", [])) or "None"
        selected_tool = state.get("current_tool", "")

        prompt = _QUERY_GENERATOR_PROMPT.format(
            task_description=task_desc,
            data_prompt=data_prompt,
            user_input=user_input,
            all_previous_error_analyses=all_error_analyses,
            selected_tool=selected_tool,
        )

        invoke_config = {}
        if active_ctx and active_logger:
            invoke_config = {"callbacks": [SessionMetricsCallback(ctx=active_ctx, metric_logger=active_logger)]}

        try:
            response = llm.invoke(prompt, config=invoke_config if invoke_config else None)
            search_query = response.content.strip().strip("\"'")
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            search_query = ""

        # Clean up prefixes from LLM response
        for prefix in ["search query:", "query:", "the search query is:"]:
            if search_query.lower().startswith(prefix):
                search_query = search_query[len(prefix):].strip()
                break

        if not search_query:
            search_query = (task_desc or selected_tool)[:256]
            logger.warning("Failed to generate query from LLM; using fallback.")

        logger.info(f"Generated search query: '{search_query}'")
        return {"search_query": search_query}

    @node_metrics(active_ctx, active_logger, "retrieve_tutorials")
    async def retrieve_tutorials(state: SemanticAgentState) -> dict:
        """MCP client node — calls the standalone Vector Store MCP server using the client wrapper."""
        logger.info("─── [Semantic Agent] retrieve_tutorials ───")

        config = state.get("config", {})
        mcp_servers = config.get("mcp_servers", {})
        server_url = mcp_servers.get("vector_store_url", "http://localhost:8010")

        tutorials_config = config.get("tutorials", {})
        top_k = tutorials_config.get("num_tutorial_retrievals", 5)
        condensed = tutorials_config.get("condense_tutorials", False)

        query = state.get("search_query", "")
        tool_name = state.get("current_tool", "")

        # Initialize the MCP client wrapper rather than raw connection inside the node
        client = VectorStoreMCPClient(server_url)

        import time
        import uuid
        t0 = time.time()
        status = "success"
        error_msg = None

        try:
            raw_tutorials = await client.retrieve_tutorials(
                query=query,
                tool_name=tool_name,
                top_k=top_k,
                condensed=condensed
            )
        except Exception as e:
            logger.error(f"Vector Store MCP retrieval call failed: {e}")
            status = "error"
            error_msg = str(e)
            raw_tutorials = []

        latency_ms = (time.time() - t0) * 1000

        if active_logger and active_ctx:
            tool_input = f"query: {query}, tool_name: {tool_name}, top_k: {top_k}, condensed: {condensed}"
            log_data = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
                "event_type": "tool_call",
                **active_ctx.snapshot(),
                "run_id": str(uuid.uuid4()),
                "parent_run_id": active_ctx.span_id.get(),
                "node_name": "retrieve_tutorials",
                "tool_name": "retrieve_tutorials",
                "tool_input": tool_input,
                "latency_ms": round(latency_ms, 2),
                "status": status,
            }
            if status == "success":
                log_data["tool_output"] = str(raw_tutorials)
            else:
                log_data["error"] = error_msg
            active_logger.info(json.dumps(log_data))

        # Deserialize to TutorialInfo named tuples
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
                logger.warning(f"Failed to deserialize tutorial: {e}")

        logger.info(f"Retrieved {len(tutorials)} tutorial candidates from MCP server")
        return {"tutorial_retrieval": tutorials}

    @node_metrics(active_ctx, active_logger, "rerank_tutorials")
    def rerank_tutorials(state: SemanticAgentState) -> dict:
        """Reranks the retrieved tutorials locally using LLM-based selection."""
        logger.info("─── [Semantic Agent] rerank_tutorials (running local/in-process reranker) ───")

        config = state.get("config", {})
        tutorials_config = config.get("tutorials", {})
        
        max_num = tutorials_config.get("max_num_tutorials", 3)
        max_length = tutorials_config.get("max_tutorial_length", 30000)
        use_summary = tutorials_config.get("use_tutorial_summary", True)
        output_folder = state.get("output_folder", "./output")

        tutorials = state.get("tutorial_retrieval", [])
        if not tutorials:
            logger.warning("  No tutorials to rerank")
            return {"tutorial_prompt": ""}

        # 1. Format tutorials info for the LLM selection prompt
        tutorials_info_lines = []
        for i, tutorial in enumerate(tutorials):
            summary_text = getattr(tutorial, "summary", "") if use_summary else ""
            summary_text = summary_text or "(No summary available)"
            tutorials_info_lines.append(
                f"{i + 1}. Title: {getattr(tutorial, 'title', 'Untitled')}\n   Summary: {summary_text}"
            )
        tutorials_info = "\n".join(tutorials_info_lines)

        all_error_analyses = state.get("all_error_analyses", [])
        all_errors = "\n\n".join(all_error_analyses) or "None"

        # Format the selection prompt
        prompt = _RERANKER_PROMPT.format(
            task_description=state.get("task_description", ""),
            data_prompt=state.get("data_prompt", ""),
            user_input=state.get("user_input", ""),
            all_previous_error_analyses=all_errors,
            tutorials_info=tutorials_info,
            max_num_tutorials=max_num,
        )

        try:
            # LLM setup and Call Logger
            llm_config = config.get("llm", {}).copy()
            llm = _init_llm(llm_config)
            
            call_logger = _LLMCallLogger(output_folder, ctx=active_ctx, metric_logger=active_logger)
            response = call_logger.call(llm, prompt, node_name="semantic_agent/rerank_tutorials")

            # Parse response
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
            
            # Fallback to top-k by retrieval score if parsing failed or returned empty
            if not selected_tutorials:
                logger.warning("  Reranking failed; falling back to top tutorials by score")
                sorted_tutorials = sorted(tutorials, key=lambda t: getattr(t, "score", 0.0) or 0.0, reverse=True)
                selected_tutorials = sorted_tutorials[:max_num]
            else:
                selected_tutorials = selected_tutorials[:max_num]

            # Format selected tutorials content into the prompt
            per_tutorial_length = max_length // max(1, len(selected_tutorials))
            formatted_parts = []

            for tutorial in selected_tutorials:
                content = getattr(tutorial, "content", "")
                if not content:
                    logger.warning(f"  Tutorial '{getattr(tutorial, 'title', 'Untitled')}' has no content — skipping")
                    continue
                if len(content) > per_tutorial_length:
                    content = content[:per_tutorial_length] + "\n...(truncated)"
                formatted_parts.append(f"### {getattr(tutorial, 'title', 'Untitled')}\n{content}")

            tutorial_prompt = "\n\n".join(formatted_parts) if formatted_parts else ""
            logger.info(f"  Selected {len(selected_tutorials)} tutorials. tutorial_prompt length: {len(tutorial_prompt)} chars")
            return {"tutorial_prompt": tutorial_prompt}

        except Exception as e:
            logger.error(f"In-process reranking failed: {e}")
            # Fallback in case of LLM or connection failure
            sorted_tutorials = sorted(tutorials, key=lambda t: getattr(t, "score", 0.0) or 0.0, reverse=True)
            selected_tutorials = sorted_tutorials[:max_num]
            formatted_parts = []
            for tutorial in selected_tutorials:
                content = getattr(tutorial, "content", "")
                if content:
                    formatted_parts.append(f"### {getattr(tutorial, 'title', 'Untitled')}\n{content[:max_length // max_num]}")
            return {"tutorial_prompt": "\n\n".join(formatted_parts)}

    # Graph Setup
    graph = StateGraph(SemanticAgentState)
    graph.add_node("generate_query", generate_query)
    graph.add_node("retrieve_tutorials", retrieve_tutorials)
    graph.add_node("rerank_tutorials", rerank_tutorials)

    graph.add_edge(START, "generate_query")
    graph.add_edge("generate_query", "retrieve_tutorials")
    graph.add_edge("retrieve_tutorials", "rerank_tutorials")
    graph.add_edge("rerank_tutorials", END)

    return graph.compile()
