#!/usr/bin/env python3
"""
MCTS Run Timeline and Duration Analyzer.
Analyzes run_report.json to check if the run exceeded the 4-hour threshold
and outputs a detailed breakdown of where the time was spent.
"""

import os
import sys
import json
from datetime import datetime

def analyze_run(report_path):
    if not os.path.exists(report_path):
        print(f"Error: Report file '{report_path}' not found.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        try:
            report = json.load(f)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return

    run_id = report.get("run_id", "unknown")
    start_time_str = report.get("orchestrator_start_time")
    end_time_str = report.get("orchestrator_end_time")
    total_duration = report.get("total_duration_seconds", 0.0)

    print("====================================================")
    print(f"Run ID: {run_id}")
    print(f"Start Time: {start_time_str}")
    print(f"End Time:   {end_time_str}")
    print(f"Total Duration: {total_duration / 3600:.3f} hours ({total_duration:.2f} seconds)")
    print("====================================================")

    # 4 hour threshold in seconds
    threshold = 4 * 3600
    exceeded = total_duration > threshold

    if exceeded:
        over_by = total_duration - threshold
        print(f"[-] YES: The run exceeded the 4-hour threshold by {over_by / 3600:.3f} hours ({over_by:.2f} seconds).")
    else:
        under_by = threshold - total_duration
        print(f"[+] NO: The run completed within the 4-hour threshold ({under_by / 3600:.3f} hours remaining).")

    telemetry = report.get("telemetry_logs", [])
    if not telemetry:
        print("Warning: No telemetry logs found in the report.")
        return

    # Track time spent per agent/action
    perception_time = 0.0
    semantic_time = 0.0
    coder_time = 0.0
    mcts_time = 0.0
    finalize_reached = False

    crossed_threshold_call = None
    first_start = None

    print("\nTimeline Analysis:")
    print("----------------------------------------------------")
    for call in telemetry:
        idx = call.get("call_index")
        action = call.get("action")
        target = call.get("target", "")
        duration = call.get("duration_seconds", 0.0)
        status = call.get("status")
        start_str = call.get("start_time")

        if start_str:
            try:
                # Handle ISO timestamps
                dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if first_start is None:
                    first_start = dt
                elapsed = (dt - first_start).total_seconds()
            except Exception:
                elapsed = 0.0
        else:
            elapsed = 0.0

        # Check if this call crossed the 4-hour threshold
        if exceeded and crossed_threshold_call is None and elapsed + duration > threshold:
            crossed_threshold_call = (idx, action, elapsed)

        # Categorize durations
        if action == "perception":
            perception_time += duration
        elif action == "retrieve_tutorials":
            semantic_time += duration
        elif action == "generate_and_run":
            coder_time += duration
        elif action in ["init", "select", "expand", "update", "backpropagate"]:
            mcts_time += duration
        elif action == "finalize":
            finalize_reached = True
            mcts_time += duration

    if crossed_threshold_call:
        idx, action, elapsed = crossed_threshold_call
        print(f"[*] Crossed 4-hour mark at Call #{idx} (Action: '{action}')")
        print(f"    Elapsed time when call started: {elapsed / 3600:.3f} hours ({elapsed:.2f} seconds)")

    print(f"\nFinalize API Call Reached: {finalize_reached}")

    # Compute overheads
    unexplained_overhead = total_duration - (perception_time + semantic_time + coder_time + mcts_time)

    print("\nDuration Breakdown by Component:")
    print("----------------------------------------------------")
    print(f"Perception Agent:        {perception_time / 3600:.3f} hrs ({perception_time:10.2f}s) | {perception_time / total_duration * 100:5.2f}%")
    print(f"Semantic Agent (Retrieval): {semantic_time / 3600:.3f} hrs ({semantic_time:10.2f}s) | {semantic_time / total_duration * 100:5.2f}%")
    print(f"Coder Agent (Gen & Run):  {coder_time / 3600:.3f} hrs ({coder_time:10.2f}s) | {coder_time / total_duration * 100:5.2f}%")
    print(f"MCTS Handler Overhead:    {mcts_time / 3600:.3f} hrs ({mcts_time:10.2f}s) | {mcts_time / total_duration * 100:5.2f}%")
    print(f"System Overhead/Gaps:     {unexplained_overhead / 3600:.3f} hrs ({unexplained_overhead:10.2f}s) | {unexplained_overhead / total_duration * 100:5.2f}%")
    print("----------------------------------------------------")
    print(f"Total:                   {total_duration / 3600:.3f} hrs ({total_duration:10.2f}s)")
    print("====================================================")

if __name__ == "__main__":
    report_file = "/home/ubuntu/MLauto/runs/20260606_172644_dog-breed-identification/run_report.json"
    if len(sys.argv) > 1:
        report_file = sys.argv[1]
    analyze_run(report_file)
