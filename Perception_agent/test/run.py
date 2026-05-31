import os
import sys
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Setup paths to import Perception_agent package and common_local
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAME_ROOT = PROJECT_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(FAME_ROOT) not in sys.path:
    sys.path.append(str(FAME_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from telemetry.metrics_context import MetricsContext
from telemetry.logging_callback import SessionMetricsCallback
from telemetry.metrics_emitter import emit_event

from agent import build_perception_agent_graph
from aggregate_logs import aggregate_logs



def setup_mock_data(data_dir: Path):
    """Creates a temporary mock data directory with files for scanning."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create a README file
    readme_content = """# Image Classification Project
This project trains a multi-modal image classification model using AutoGluon on dog and cat photos.
The dataset is split into training and validation sets inside the dataset/ directory.
We want to achieve high accuracy and evaluate on val_data.
"""
    (data_dir / "README.md").write_text(readme_content, encoding="utf-8")
    
    # 2. Create a dummy python file
    code_content = """import os
print("Starting classification pipeline...")
"""
    (data_dir / "train.py").write_text(code_content, encoding="utf-8")
    
    # 3. Create a subdirectory with a mock dataset description
    dataset_dir = data_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    (dataset_dir / "dataset_info.txt").write_text("dataset size: 1000 images, 2 classes", encoding="utf-8")


def main():
    print("═══ Running Perception Agent LangGraph Pipeline Test ═══")

    # Setup metrics logging output directory
    log_dir = Path(__file__).resolve().parent / f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = log_dir / "metrics.jsonl"

    metric_logger = logging.getLogger("agent_metrics")
    metric_logger.setLevel(logging.INFO)
    metric_logger.propagate = False
    
    # Avoid duplicate handlers if main() called multiple times
    if not metric_logger.handlers:
        _stream_handler = logging.StreamHandler(sys.stdout)
        _stream_handler.setFormatter(logging.Formatter('%(message)s'))
        metric_logger.addHandler(_stream_handler)
        _file_handler = logging.FileHandler(metrics_jsonl, mode="a", encoding="utf-8")
        _file_handler.setFormatter(logging.Formatter('%(message)s'))
        metric_logger.addHandler(_file_handler)

    # Initialize MetricsContext
    ctx = MetricsContext(agent_id="perception_agent")
    ctx.init_from_payload({"session_id": "local_test", "context_id": "local_test_run"})

    # 1. Initialize compiled graph
    graph = build_perception_agent_graph(ctx=ctx, metric_logger=metric_logger)
    print("✓ LangGraph Compiled successfully.")

    # 2. Setup self-contained mock data
    mock_data_dir = Path(__file__).resolve().parent / "mock_data"
    setup_mock_data(mock_data_dir)
    print(f"✓ Mock data directory prepared at: {mock_data_dir}")

    # 3. Configure mock inputs
    config = {
        "llm": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        },
        "a2a_agents": {
            "semantic_memory_url": "http://localhost:8088"
        }
    }

    initial_state = {
        "config": config,
        "input_data_folder": str(mock_data_dir),
        "output_folder": str(PROJECT_ROOT / "test" / "test_output"),
        "user_input": "Train an image classifier with high accuracy preset and evaluate on validation data.",
        "all_error_analyses": []
    }

    print("\nInitial State Context:")
    print(f" - Input Folder: {initial_state['input_data_folder']}")
    print(f" - User Input: {initial_state['user_input']}")
    print(f" - LLM Model: {config['llm']['model']}")

    # 4. Invoke the graph
    print("\nExecuting graph invoke...")
    try:
        langgraph_config = {
            "callbacks": [SessionMetricsCallback(ctx=ctx, metric_logger=metric_logger)]
        }
        result = graph.invoke(initial_state, config=langgraph_config)
        print("\nPipeline Result:")
        print(f" - Task Description: {result.get('task_description')}")
        print(f" - Selected Tools: {result.get('selected_tools')}")
        print(f" - Current Selected Tool: {result.get('current_tool')}")
        print(f" - Tool Prompt Length: {len(result.get('tool_prompt', ''))} chars")
        print(f" - Tutorial Prompt Length: {len(result.get('tutorial_prompt', ''))} chars")
        
        if result.get('task_description'):
            print("\n--- Task Description Snippet ---")
            print(result.get('task_description')[:300] + "\n...")
            print("---------------------------------")
            
    except Exception as e:
        print(f"✗ Graph execution failed: {e}")
        print("\nNote: Make sure the Semantic Agent A2A server (port 8088) is running if you want to test semantic memory retrieval.")
    finally:
        # Clean up mock data directory
        if mock_data_dir.exists():
            shutil.rmtree(mock_data_dir)
            print("\n✓ Temporary mock data directory cleaned up.")
        
        # Clean up test output directory
        test_output_dir = PROJECT_ROOT / "test" / "test_output"
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
            print("✓ Temporary test output directory cleaned up.")

        # Post-run log aggregation
        if metrics_jsonl.exists():
            print(f"\nMetrics written to: {metrics_jsonl}")
            aggregate_logs(metrics_jsonl, output_dir=log_dir)
            print(f"✓ Aggregated logs saved in: {log_dir}")


if __name__ == "__main__":
    main()
