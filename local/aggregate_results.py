import os
import json
import csv
from glob import glob

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def safe_load_jsonl(path):
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    lines.append(json.loads(line))
                except:
                    pass
    except:
        pass
    return lines

def aggregate_results(runs_dir: str, output_csv: str):
    run_folders = glob(os.path.join(runs_dir, "*"))
    
    rows = []
    for folder in run_folders:
        if not os.path.isdir(folder):
            continue
            
        run_id = os.path.basename(folder)
        dataset = run_id.split("_", 2)[-1] if "_" in run_id else run_id
        
        report = safe_load_json(os.path.join(folder, "run_report.json"))
        if not report:
            continue
            
        orchestrator_logs = safe_load_jsonl(os.path.join(folder, "orchestrator_telemetry.jsonl"))
        perception_metrics = safe_load_jsonl(os.path.join(folder, "perception_metrics.jsonl"))
        semantic_metrics = safe_load_jsonl(os.path.join(folder, "semantic_metrics.jsonl"))
        coder_metrics = safe_load_jsonl(os.path.join(folder, "coder_metrics.jsonl"))
        
        perception_time = sum(log.get("duration_seconds", 0) for log in orchestrator_logs if log.get("action") == "perception")
        semantic_time = sum(log.get("duration_seconds", 0) for log in orchestrator_logs if log.get("action") == "retrieve_tutorials")
        coder_time = sum(log.get("duration_seconds", 0) for log in orchestrator_logs if log.get("action") == "generate_and_run")
        
        all_metrics = perception_metrics + semantic_metrics + coder_metrics
        llm_calls = [m for m in all_metrics if m.get("event_type") == "llm_call"]
        sandbox_calls = [m for m in all_metrics if m.get("event_type") == "tool_call" and m.get("tool_name") == "sandbox_exec_shell"]
        
        total_tokens = sum(m.get("input_tokens", 0) + m.get("output_tokens", 0) for m in llm_calls)
        avg_llm_latency = sum(m.get("latency_ms", 0) / 1000.0 for m in llm_calls) / len(llm_calls) if llm_calls else 0
        
        rows.append({
            "run_id": run_id,
            "dataset": dataset,
            "total_time_s": report.get("total_duration_seconds", 0),
            "perception_time_s": perception_time,
            "semantic_time_s": semantic_time,
            "coder_time_s": coder_time,
            "mcts_iterations": (report.get("mcts_tree") or {}).get("iteration", 0),
            "best_score": ((report.get("final_outcome") or {}).get("status") or {}).get("best_score"),
            "llm_calls": len(llm_calls),
            "llm_total_tokens": total_tokens,
            "sandbox_exec_calls": len(sandbox_calls),
            "avg_llm_latency_ms": round(avg_llm_latency * 1000, 2)
        })
        
    if rows:
        keys = rows[0].keys()
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Aggregated {len(rows)} runs to {output_csv}")
    else:
        print("No valid runs found to aggregate.")

if __name__ == "__main__":
    runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))
    output_csv = os.path.join(runs_dir, "results_summary.csv")
    aggregate_results(runs_dir, output_csv)
