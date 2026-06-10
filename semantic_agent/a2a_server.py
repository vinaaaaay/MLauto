import os
os.environ["USER"] = os.environ.get("USER") or "administrator"
import json
import logging
import time
import sys
import uuid
import psutil
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load environment variables if .env file exists
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from semantic_agent import build_semantic_agent_graph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_agent.server")

# Metrics logging setup
from semantic_agent.common_local import MetricsContext, emit_event, SessionMetricsCallback

metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
if not metric_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter('%(message)s'))
    metric_logger.addHandler(_handler)

from semantic_agent.agent import ctx

app = FastAPI(title="Semantic Agent")

# Compile graph once as a module-level singleton
_graph = build_semantic_agent_graph(ctx=ctx, metric_logger=metric_logger)
logger.info("Semantic Agent LangGraph pipeline compiled successfully.")

@app.post("/invoke")
async def invoke(request: Request):
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}
            
        task_description = data.get("task_description", "")
        current_tool = data.get("current_tool", "")
        all_error_analyses = data.get("all_error_analyses", [])
        run_id = data.get("run_id")
        
        if run_id:
            runs_dir = os.environ.get("RUNS_DIR", "/runs")
            log_path = os.path.join(runs_dir, run_id, "semantic_metrics.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            metric_logger.handlers = [h for h in metric_logger.handlers if not isinstance(h, logging.FileHandler)]
            _fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            _fh.setFormatter(logging.Formatter('%(message)s'))
            metric_logger.addHandler(_fh)

        extra_fields = {}
        for k, v in data.items():
            if k not in ["task_description", "current_tool", "all_error_analyses", "skill", "run_id"]:
                extra_fields[k] = v

        config = {
            "llm": {
                "model": "gpt-4o-mini",
                "temperature": 0.1
            },
            "mcp_servers": {
                "vector_store_url": os.environ.get("VECTOR_STORE_URL", "http://localhost:8010")
            },
            "tutorials": {
                "num_tutorial_retrievals": 3,
                "condense_tutorials": False,
                "max_num_tutorials": 2
            }
        }
        
        if "config" in extra_fields and isinstance(extra_fields["config"], dict):
            config.update(extra_fields.pop("config"))

        # Resolve output folder dynamically if run_id is supplied
        runs_dir = os.environ.get("RUNS_DIR", "/runs")
        if run_id:
            output_folder = os.path.join(runs_dir, run_id)
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_folder = "./a2a_output"

        initial_state = {
            "config": config,
            "output_folder": output_folder,
            "task_description": task_description,
            "current_tool": current_tool,
            "all_error_analyses": all_error_analyses,
            **extra_fields
        }

        process = psutil.Process()
        initial_mem = process.memory_info().rss
        t0 = time.time()
        
        tracing_payload = {
            "session_id": "unknown",
            "context_id": run_id or uuid.uuid4().hex,
        }
        if "tracing" in data:
            tracing_payload["tracing"] = data["tracing"]
        elif "context_id" in data:
            tracing_payload["context_id"] = data["context_id"]
        if "session_id" in data:
            tracing_payload["session_id"] = data["session_id"]
            
        ctx.init_from_payload(tracing_payload)
        
        langgraph_config = {
            "callbacks": [SessionMetricsCallback(ctx=ctx, metric_logger=metric_logger)]
        }

        result = await _graph.ainvoke(initial_state, config=langgraph_config)
        
        elapsed = time.time() - t0
        peak_mem = max(initial_mem, process.memory_info().rss)

        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "psutil_metrics_graph",
            "graph_name": "semantic_agent",
            "graph_e2e_s": round(elapsed, 4),
            "peak_RAM_GB": round(peak_mem / (1024**3), 4),
            "step_count": 3,
            "iteration_count": 0,
        })

        tutorial_prompt = result.get("tutorial_prompt", "")
        return JSONResponse({"tutorial_prompt": tutorial_prompt})
    except Exception as e:
        logger.error(f"Error during semantic graph execution: {e}")
        import traceback
        return JSONResponse({"error": str(e), "detail": traceback.format_exc()}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting Semantic Agent Server on port 8088...")
    uvicorn.run("a2a_server:app", host="0.0.0.0", port=8088)
