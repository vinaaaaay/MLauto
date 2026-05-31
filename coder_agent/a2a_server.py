import os
import json
import logging
import time
import uuid
import psutil
import sys
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

from coder_agent import build_coder_agent_graph
from coder_agent.tools import AgentInfraWSSandbox

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coder_agent.server")

# Metrics logging setup
from telemetry.metrics_context import MetricsContext
from telemetry.metrics_emitter import emit_event
from telemetry.logging_callback import SessionMetricsCallback


metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
if not metric_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter('%(message)s'))
    metric_logger.addHandler(_handler)

from coder_agent.agent import ctx

app = FastAPI(title="Coder Agent")

# Compile graph once as a module-level singleton
_graph = build_coder_agent_graph(ctx=ctx, metric_logger=metric_logger)
logger.info("Coder Agent LangGraph pipeline compiled successfully.")

@app.post("/invoke")
async def invoke(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
        
    run_id = data.get("run_id")
    if run_id:
        runs_dir = os.environ.get("RUNS_DIR", "/runs")
        log_path = os.path.join(runs_dir, run_id, "coder_metrics.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        metric_logger.handlers = [h for h in metric_logger.handlers if not isinstance(h, logging.FileHandler)]
        _fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        _fh.setFormatter(logging.Formatter('%(message)s'))
        metric_logger.addHandler(_fh)

    task_description = data.get("task_description", "")
    data_prompt = data.get("data_prompt", "")
    user_input = data.get("user_input", "")
    current_tool = data.get("current_tool", "")
    tool_prompt = data.get("tool_prompt", "")
    tutorial_prompt = data.get("tutorial_prompt", "")
    all_error_analyses = data.get("all_error_analyses", [])
    previous_python_code = data.get("previous_python_code", "")
    previous_bash_script = data.get("previous_bash_script", "")
    stage = data.get("stage", "root")
    iteration = data.get("iteration", 0)
    node_id = data.get("node_id")

    extra_fields = {}
    for k, v in data.items():
        if k not in [
            "task_description", "data_prompt", "user_input", "current_tool",
            "tool_prompt", "tutorial_prompt", "all_error_analyses",
            "previous_python_code", "previous_bash_script", "stage", "iteration", "node_id", "run_id"
        ]:
            extra_fields[k] = v

    config = {
        "llm": {
            "model": "gpt-5-nano",
            "temperature": 0.1
        },
        "mcts": {
            "continuous_improvement": True
        },
        "mcp_servers": {
            "sandbox_url": "http://localhost:8081/mcp"
        },
        "tool_registry_path": os.environ.get("REGISTRY_PATH", "/tools_registry")
    }

    if "config" in extra_fields and isinstance(extra_fields["config"], dict):
        config.update(extra_fields.pop("config"))

    sandbox_client = AgentInfraWSSandbox(base_url=os.environ.get("SANDBOX_URL", "http://sandbox:8080"))

    initial_state = {
        "config": config,
        "output_folder": "./coder_agent_output",
        "task_description": task_description,
        "data_prompt": data_prompt,
        "user_input": user_input,
        "current_tool": current_tool,
        "tool_prompt": tool_prompt,
        "tutorial_prompt": tutorial_prompt,
        "all_error_analyses": all_error_analyses,
        "previous_python_code": previous_python_code,
        "previous_bash_script": previous_bash_script,
        "stage": stage,
        "iteration": iteration,
        "node_id": node_id,
        "sandbox_client": sandbox_client,
        **extra_fields
    }

    process = psutil.Process()
    initial_mem = process.memory_info().rss
    t0 = time.time()
    
    tracing_payload = {
        "session_id": "unknown",
        "context_id": run_id or uuid.uuid4().hex,
    }
    ctx.init_from_payload(tracing_payload)
    
    langgraph_config = {
        "callbacks": [SessionMetricsCallback(ctx=ctx, metric_logger=metric_logger)]
    }
    
    try:
        result = await _graph.ainvoke(initial_state, config=langgraph_config)
        
        elapsed = time.time() - t0
        peak_mem = max(initial_mem, process.memory_info().rss)
        
        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "psutil_metrics_graph",
            "graph_name": "coder_agent",
            "graph_e2e_s": round(elapsed, 4),
            "peak_RAM_GB": round(peak_mem / (1024**3), 4),
            "step_count": result.get("iteration", 0) + 1,
            "iteration_count": result.get("iteration", 0),
        })
        
        coder_result = {
            "python_code": result.get("python_code", ""),
            "bash_script": result.get("bash_script", ""),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "decision": result.get("decision", "FIX"),
            "error_summary": result.get("error_summary", ""),
            "validation_score": result.get("validation_score"),
            "error_message": result.get("error_message", ""),
        }
        return JSONResponse(coder_result)
    except Exception as e:
        logger.error(f"Error during coder graph execution: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting Coder Agent Server on port 8089...")
    uvicorn.run("a2a_server:app", host="0.0.0.0", port=8089)
