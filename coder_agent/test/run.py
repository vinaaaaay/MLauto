import os
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.coder_agent import build_coder_agent_graph
from agents.coder_agent.utils import SyncMCPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coder_agent.run")

def main():
    logger.info("Starting Coder Agent standalone direct run...")

    # Configure initial state
    config = {
        "llm": {
            "model": "gpt-4o",
            "temperature": 0.1
        },
        "mcts": {
            "continuous_improvement": True
        },
        "mcp_servers": {
            "sandbox_url": "http://localhost:8081/mcp"
        },
        "tool_registry_path": str(PROJECT_ROOT / "agents" / "MLauto" / "tools_registry")
    }

    # Initialize MCP client
    mcp_client = SyncMCPClient(config["mcp_servers"]["sandbox_url"])
    mcp_client.connect()

    initial_state = {
        "config": config,
        "output_folder": "./coder_agent_output",
        "task_description": "Train an AutoGluon tabular model to predict passenger survival on the Spaceship Titanic dataset.",
        "data_prompt": "Train data at /home/gem/workspace/data/train.csv with label 'Transported'. Test data at /home/gem/workspace/data/test.csv. Predict on the test set and output predictions matching sample submission column formats.",
        "user_input": "Run tabular prediction.",
        "current_tool": "autogluon.tabular",
        "tool_prompt": "Use autogluon.tabular predictor with fits.",
        "tutorial_prompt": "",
        "all_error_analyses": [],
        "previous_python_code": "",
        "previous_bash_script": "",
        "stage": "root",
        "iteration": 0,
        "mcp_client": mcp_client,
    }

    graph = build_coder_agent_graph()

    logger.info("Invoking Coder Agent LangGraph graph...")
    result = graph.invoke(initial_state)

    logger.info("Execution completed. Results:")
    logger.info(f"Decision: {result.get('decision')}")
    logger.info(f"Validation Score: {result.get('validation_score')}")
    logger.info(f"Stdout:\n{result.get('stdout')}")
    logger.info(f"Stderr:\n{result.get('stderr')}")
    logger.info(f"Python file path: {result.get('python_file_path')}")
    
    mcp_client.disconnect()

if __name__ == "__main__":
    main()
