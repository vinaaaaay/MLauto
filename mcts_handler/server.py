from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from handler import handle_request
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MCTS Handler")

@app.post("/invoke")
def invoke_endpoint(payload: Dict[str, Any]):
    try:
        return handle_request(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
