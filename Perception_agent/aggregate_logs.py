#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

def aggregate_logs(jsonl_path: Path, output_dir: Path = None):
    if not jsonl_path.exists():
        print(f"Error: Input file {jsonl_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    if output_dir is None:
        output_dir = jsonl_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_logs = []
    debug_logs = []
    llm_call_logs = []
    tool_call_logs = []
    psutil_logs = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON and will be skipped. Error: {e}", file=sys.stderr)
                continue

            raw_logs.append(log_entry)

            event_type = log_entry.get("event_type")
            if event_type == "debug":
                debug_logs.append(log_entry)
            elif event_type == "llm_call":
                llm_call_logs.append(log_entry)
            elif event_type == "tool_call":
                tool_call_logs.append(log_entry)
            elif event_type in ("psutil_metrics_node", "psutil_metrics_graph"):
                psutil_logs.append(log_entry)

    # File 1: raw json logs
    raw_json_path = output_dir / "metrics.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_logs, f, indent=2)
    print(f"Created: {raw_json_path}")

    # File 2: debug events
    debug_json_path = output_dir / "debug.json"
    with open(debug_json_path, "w", encoding="utf-8") as f:
        json.dump(debug_logs, f, indent=2)
    print(f"Created: {debug_json_path}")

    # File 3: llm_call events
    llm_call_json_path = output_dir / "llm_call.json"
    with open(llm_call_json_path, "w", encoding="utf-8") as f:
        json.dump(llm_call_logs, f, indent=2)
    print(f"Created: {llm_call_json_path}")

    # File 3b: tool_call events
    tool_call_json_path = output_dir / "tool_call.json"
    with open(tool_call_json_path, "w", encoding="utf-8") as f:
        json.dump(tool_call_logs, f, indent=2)
    print(f"Created: {tool_call_json_path}")

    # File 4: psutil events (includes both node and graph metrics)
    psutil_json_path = output_dir / "psutil.json"
    with open(psutil_json_path, "w", encoding="utf-8") as f:
        json.dump(psutil_logs, f, indent=2)
    print(f"Created: {psutil_json_path}")

    print("\nSummary of events aggregated:")
    print(f"  Total raw events: {len(raw_logs)}")
    print(f"  Debug events:     {len(debug_logs)}")
    print(f"  LLM call events:  {len(llm_call_logs)}")
    print(f"  Tool call events: {len(tool_call_logs)}")
    print(f"  psutil events:    {len(psutil_logs)}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate jsonl logs into JSON format and group by event types.")
    parser.add_argument("jsonl_path", type=str, help="Path to the metrics.jsonl file")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Directory to save the generated JSON files (default: same as input file)")
    
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    aggregate_logs(jsonl_path, output_dir)

if __name__ == "__main__":
    main()
