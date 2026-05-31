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
if Path(__file__).resolve().parent.name == "Perception_agent":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load environment variables if .env file exists
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from agent import build_perception_agent_graph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception_agent.server")

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

from agent import ctx

app = FastAPI(title="Perception Agent")

# Compile graph once as a module-level singleton
_graph = build_perception_agent_graph(ctx=ctx, metric_logger=metric_logger)
logger.info("Perception Agent LangGraph pipeline compiled successfully.")

@app.post("/invoke")
async def invoke(request: Request):
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}
            
        run_id = data.get("run_id")
        if run_id:
            runs_dir = os.environ.get("RUNS_DIR", "/runs")
            log_path = os.path.join(runs_dir, run_id, "perception_metrics.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            metric_logger.handlers = [h for h in metric_logger.handlers if not isinstance(h, logging.FileHandler)]
            _fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            _fh.setFormatter(logging.Formatter('%(message)s'))
            metric_logger.addHandler(_fh)

        # Enforce robust default configuration including tool registry path
        config = {
            "llm": {
                "model": "gpt-5-nano",
                "temperature": 0.1,
                "max_tokens": 16384
            },
            "tool_registry_path": os.environ.get("REGISTRY_PATH", "/tools_registry")
        }
        
        # Safely update default config with incoming payload config
        if "config" in data and isinstance(data["config"], dict):
            if "llm" in data["config"] and isinstance(data["config"]["llm"], dict):
                config["llm"].update(data["config"]["llm"])
            for k, v in data["config"].items():
                if k != "llm":
                    config[k] = v

        input_data_folder = data.get("input_data_folder", "")
        default_output = "/app/perception_output" if os.path.exists("/app") else str(PROJECT_ROOT / "perception_output")
        output_folder = data.get("output_folder", default_output)
        user_input = data.get("user_input", "")
        
        extra_fields = {}
        for k, v in data.items():
            if k not in ["input_data_folder", "output_folder", "user_input", "config", "skill", "run_id"]:
                extra_fields[k] = v
                
        initial_state = {
            "config": config,
            "input_data_folder": input_data_folder,
            "output_folder": output_folder,
            "user_input": user_input,
            "all_error_analyses": [],
            **extra_fields,
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
        
        result = await _graph.ainvoke(initial_state, config=langgraph_config)
        
        elapsed = time.time() - t0
        peak_mem = max(initial_mem, process.memory_info().rss)

        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "psutil_metrics_graph",
            "graph_name": "perception_agent",
            "graph_e2e_s": round(elapsed, 4),
            "peak_RAM_GB": round(peak_mem / (1024**3), 4),
            "step_count": 4,
            "iteration_count": 0,
        })
        
        perception_result = {
            "data_prompt": result.get("data_prompt", ""),
            "task_description": result.get("task_description", ""),
            "selected_tools": result.get("selected_tools", []),
            "current_tool": result.get("current_tool", ""),
            "tool_prompt": result.get("tool_prompt", ""),
            "tutorial_prompt": result.get("tutorial_prompt", ""),
        }
        return JSONResponse(perception_result)
    except Exception as e:
        logger.error(f"Error during perception graph execution: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting Perception Agent Server on port 8020...")
    uvicorn.run("a2a_server:app", host="0.0.0.0", port=8020)
