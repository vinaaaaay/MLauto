import os
import time
import json
import httpx
import sys
import argparse
from datetime import datetime


ORCHESTRATOR_URL = "http://localhost:8000"

def run_benchmark(dataset_name: str, mle_bench_path: str, max_iterations: int = 3, config: dict = None, user_prompt: str = "", max_runtime_seconds: int = 14400):
    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{dataset_name}"
    
    # We mount mle_bench_path to /home/gem/workspace/data in sandbox
    input_data_folder = f"/home/gem/workspace/data/{dataset_name}/public"
    
    payload = {
        "run_id": run_id,
        "input_data_folder": input_data_folder,
        "user_input": user_prompt,
        "config": config or {},
        "max_iterations": max_iterations,
        "max_runtime_seconds": max_runtime_seconds
    }
    
    print(f"Starting run for {dataset_name} (run_id: {run_id}, timeout: {max_runtime_seconds}s)")
    
    # httpx timeout = server timeout + 5 min grace period for finalization
    client_timeout = max_runtime_seconds + 300
    try:
        response = httpx.post(f"{ORCHESTRATOR_URL}/run", json=payload, timeout=client_timeout)
        response.raise_for_status()
        report = response.json()
        status = report.get('status')
        final_outcome = report.get('final_outcome') or {}
        best_score = final_outcome.get('status', {}).get('best_score') if final_outcome else None
        print(f"[{run_id}] Finished! Status: {status}. Best Score: {best_score}")
        
        # Automatically generate telemetry graphs
        run_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", run_id)
        if os.path.exists(run_folder):
            try:
                import subprocess
                plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_telemetry.py")
                subprocess.run([sys.executable, plot_script, run_folder], check=True)
            except Exception as plot_e:
                print(f"[{run_id}] Failed to generate graphs: {plot_e}")
                
    except Exception as e:
        print(f"[{run_id}] Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, help="Comma separated list of dataset names")
    parser.add_argument("--mle-bench-path", type=str, default="/home/ubuntu/mle-bench-lite", help="Path to mle-bench-lite datasets")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max MCTS iterations")
    parser.add_argument("--max-runtime-seconds", type=int, default=14400, help="Max runtime in seconds per dataset (default: 14400 = 4 hours)")
    parser.add_argument("--config-file", type=str, help="Path to a JSON configuration file (e.g., config.example.json)")
    parser.add_argument("--user-prompt", type=str, default="", help="Optional user prompt / instruction")
    args = parser.parse_args()
    
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
            config = json.load(f)
            
    datasets = args.datasets.split(",") if args.datasets else []
    if not datasets:
        # Default: auto-discover from path
        if os.path.exists(args.mle_bench_path):
            datasets = [d for d in os.listdir(args.mle_bench_path) if os.path.isdir(os.path.join(args.mle_bench_path, d))]
        else:
            print("No datasets provided and path not found.")
            exit(1)
            
    for ds in datasets:
        run_benchmark(ds.strip(), args.mle_bench_path, args.max_iterations, config, args.user_prompt, args.max_runtime_seconds)
