"""
MLauto — Entry point for the MCTS-based AutoML pipeline.

Replicates autogluon-assistant's coding_agent.run_agent():
  1. Perception phase: scan data, identify task, select tools, retrieve tutorials
  2. Iterative coding phase: MCTS tree search (select → expand → code → execute → backpropagate)
"""

import argparse
from orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="MLauto — AutoGluon-Assistant replica in LangGraph")
    parser.add_argument("input_data_folder", help="Path to input data directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--config", "-c", default=None, help="Config YAML path")
    parser.add_argument("--user-input", "-u", default="", help="User instructions")
    parser.add_argument("--max-iterations", "-n", type=int, default=None, help="Max MCTS iterations")
    parser.add_argument("--verbosity", "-v", type=int, default=2, choices=[0, 1, 2, 3, 4], help="Set verbosity level (0-4)")

    args = parser.parse_args()

    orchestrator = Orchestrator(
        input_data_folder=args.input_data_folder,
        output_folder=args.output,
        config_path=args.config,
        user_input=args.user_input,
        max_iterations=args.max_iterations,
        verbosity=args.verbosity,
    )
    orchestrator.run()

if __name__ == "__main__":
    main()
