from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import logging
from orchestrator import run_orchestration

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MLauto Orchestrator")

class RunRequest(BaseModel):
    run_id: str
    input_data_folder: str
    user_input: str = ""
    config: Dict[str, Any] = {}
    max_iterations: int = 3

@app.post("/run")
def run_endpoint(req: RunRequest):
    try:
        report = run_orchestration(
            run_id=req.run_id,
            input_data_folder=req.input_data_folder,
            user_input=req.user_input,
            config=req.config,
            max_iterations=req.max_iterations
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

