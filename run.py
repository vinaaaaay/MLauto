"""
MLauto — Entry point.

Runs the Perception Module (once) and then the Iterative Coding Module (loop)
using LangGraph StateGraphs.

Usage:
    python run.py -i /path/to/data -o /path/to/output
    python run.py -i /path/to/data -o /path/to/output -n 10 -u "Train a classifier"
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Ensure the project root is on sys.path so `shared.*` imports work
sys.path.insert(0, str(Path(__file__).parent))

from shared.logging_config import configure_logging, log_state_snapshot
from perception_agent.graph import build_perception_graph
from iterativecoding_agent.graph import build_iterative_coding_graph

logger = logging.getLogger("mlauto")


def load_config(config_path: str = None) -> dict:
    """Load YAML config, falling back to defaults."""
    default_path = Path(__file__).parent / "config.yaml"
    path = Path(config_path) if config_path else default_path

    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config not found at {path}; using defaults.")
        return {}


def main():
    parser = argparse.ArgumentParser(description="MLauto — AutoML with LangGraph")
    parser.add_argument("-i", "--input_data_folder", required=True, help="Path to input data folder")
    parser.add_argument("-o", "--output_dir", default=None, help="Path to output directory")
    parser.add_argument("-c", "--config_path", default=None, help="Path to config YAML")
    parser.add_argument("-n", "--max_iterations", type=int, default=5, help="Max coding iterations")
    parser.add_argument("-u", "--user_input", default="", help="User instruction/description")
    parser.add_argument("-v", "--verbosity", type=int, default=2,
                        help="Logging verbosity: 0=ERROR, 1=WARNING, 2=INFO (default), 3=DETAIL, 4=DEBUG")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)

    # Determine output folder
    if args.output_dir:
        output_folder = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join("runs", f"mlauto-{timestamp}")

    os.makedirs(output_folder, exist_ok=True)

    # ── Set up logging (console + file handlers) ──
    configure_logging(output_dir=output_folder, verbosity=args.verbosity)

    max_iterations = args.max_iterations or config.get("execution", {}).get("max_iterations", 5)

    logger.info("=" * 60)
    logger.info("MLauto — Starting pipeline")
    logger.info(f"  Input:          {os.path.abspath(args.input_data_folder)}")
    logger.info(f"  Output:         {os.path.abspath(output_folder)}")
    logger.info(f"  Max iterations: {max_iterations}")
    logger.info(f"  Verbosity:      {args.verbosity}")
    logger.info(f"  LLM model:      {config.get('llm', {}).get('model', 'gpt-4o')}")
    logger.info(f"  Docker image:   {config.get('execution', {}).get('docker_image', 'mlauto-executor:latest')}")
    logger.info("=" * 60)

    start_time = time.time()

    # ── Step 1: Perception Module ────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("Phase 1: PERCEPTION MODULE")
    logger.info("━" * 60)

    perception_graph = build_perception_graph()

    initial_state = {
        "input_data_folder": args.input_data_folder,
        "output_folder": output_folder,
        "user_input": args.user_input,
        "config": config,
    }

    log_state_snapshot(initial_state, "initial_input", output_folder)

    perception_result = perception_graph.invoke(initial_state)

    log_state_snapshot(perception_result, "after_perception", output_folder)

    logger.info("")
    logger.info("Perception Module — Summary:")
    logger.info(f"  data_prompt length:   {len(perception_result.get('data_prompt', ''))}")
    logger.info(f"  description_files:    {perception_result.get('description_files', [])}")
    logger.info(f"  task_description:     {perception_result.get('task_description', '')[:200]}...")
    logger.info(f"  selected_tools:       {perception_result.get('selected_tools', [])}")
    logger.info(f"  current_tool:         {perception_result.get('current_tool', 'N/A')}")

    # ── Step 2: Iterative Coding Module ──────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("Phase 2: ITERATIVE CODING MODULE")
    logger.info("━" * 60)

    coding_graph = build_iterative_coding_graph()

    # Merge perception outputs into the state for the coding module
    coding_state = {
        **perception_result,
        "iteration": 0,
        "max_iterations": max_iterations,
        "all_error_analyses": [],
        "previous_python_code": "",
        "previous_bash_script": "",
        "best_score": None,
        "best_code": "",
        "is_complete": False,
    }

    log_state_snapshot(coding_state, "before_coding", output_folder)

    coding_result = coding_graph.invoke(coding_state)

    log_state_snapshot(coding_result, "after_coding", output_folder)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("MLauto — Pipeline complete")
    logger.info("=" * 60)
    logger.info(f"  Total time:     {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"  Iterations:     {coding_result.get('iteration', 0)}")
    logger.info(f"  Final decision: {coding_result.get('decision', 'N/A')}")
    logger.info(f"  Best score:     {coding_result.get('best_score', 'N/A')}")
    logger.info(f"  Tool used:      {coding_result.get('current_tool', 'N/A')}")
    logger.info(f"  Output dir:     {os.path.abspath(output_folder)}")

    # Save best code
    if coding_result.get("best_code"):
        best_path = os.path.join(output_folder, "best_code.py")
        with open(best_path, "w") as f:
            f.write(coding_result["best_code"])
        logger.info(f"  Best code:      {best_path}")

    # Log output directory contents
    logger.info("")
    logger.info("Output directory contents:")
    for root, dirs, files in os.walk(output_folder):
        level = root.replace(output_folder, "").count(os.sep)
        indent = "  " * (level + 1)
        logger.info(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 2)
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            logger.info(f"{sub_indent}{file} ({size:,} bytes)")

    logger.info("")
    logger.info("Log files saved to:")
    logger.info(f"  Console log:    {os.path.join(output_folder, 'logs.txt')}")
    logger.info(f"  Debug log:      {os.path.join(output_folder, 'debug_logs.txt')}")
    logger.info(f"  LLM calls:      {os.path.join(output_folder, 'llm_calls.jsonl')}")
    logger.info(f"  State snapshots: {os.path.join(output_folder, 'state_snapshots/')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
