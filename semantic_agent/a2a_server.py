import os
os.environ["USER"] = os.environ.get("USER") or "administrator"
import json
import logging
import time
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

from semantic_agent import build_semantic_agent_graph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_agent.server")

app = FastAPI(title="Semantic Agent")

# Compile graph once as a module-level singleton
_graph = build_semantic_agent_graph()
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

        result = await _graph.ainvoke(initial_state)
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
