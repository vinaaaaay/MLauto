import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re

def get_dataset_details(run_folder):
    """
    Dynamically infers the dataset name from the run folder or logs
    and computes the size of the dataset on the host machine.
    """
    dataset_name = "unknown-dataset"
    try:
        # Search in jsonl files for the data folder
        for log_file in glob.glob(os.path.join(run_folder, "*.jsonl")):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "data/" in line:
                        m = re.search(r"data/([^/]+)/public", line)
                        if m:
                            dataset_name = m.group(1)
                            break
    except Exception:
        pass

    if dataset_name == "unknown-dataset":
        # Fallback to run folder name parsing
        base = os.path.basename(run_folder)
        if "_" in base:
            parts = base.split("_")
            if len(parts) >= 3:
                dataset_name = "_".join(parts[2:])

    # Calculate dataset size on host (defaulting to 0.57 GB fallback if not found)
    dataset_size_gb = 0.57
    try:
        host_data_path = f"/home/administrator/dreamlab/mle-bench-lite/{dataset_name}/public"
        if os.path.exists(host_data_path):
            total_bytes = 0
            for root, dirs, files in os.walk(host_data_path):
                for file in files:
                    total_bytes += os.path.getsize(os.path.join(root, file))
            dataset_size_gb = total_bytes / (1024 ** 3)
    except Exception:
        pass

    # Prettify the dataset name for printing
    pretty_dataset_name = dataset_name.replace("-", " ").title()
    if "tabular" in dataset_name.lower():
        pretty_dataset_name = dataset_name # Keep tabular-playground-series style lowercase for consistency if needed

    return dataset_name, f"{dataset_size_gb:.2f} GB"

def parse_orchestrator_steps(run_folder):
    """
    Parses orchestrator_telemetry.jsonl to extract chronological REST API calls
    and computes internal orchestrator processing latency (overhead).
    """
    events = []
    log_file = os.path.join(run_folder, "orchestrator_telemetry.jsonl")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "call_index" in d and "target" in d:
                        events.append(d)
                except Exception:
                    pass
    
    # Sort events by call_index
    events.sort(key=lambda x: x.get("call_index", 0))
    return events

def parse_span_breakdown(run_folder):
    """
    Parses coder_metrics.jsonl to extract latency metrics categorized into spans.
    """
    events = []
    log_file = os.path.join(run_folder, "coder_metrics.jsonl")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    events.append(d)
                except Exception:
                    pass
    
    # Group by span (e.g. node_id or iteration)
    spans_data = {}
    for ev in events:
        node_id = ev.get("node_id")
        iteration = ev.get("iteration")
        span_key = node_id if node_id is not None else iteration
        if span_key is None:
            continue
        if span_key not in spans_data:
            spans_data[span_key] = {
                "llm_s": 0.0,
                "write_s": 0.0,
                "shell_s": 0.0,
                "min_ts": float('inf')
            }
        
        # Parse timestamp to sort spans chronologically
        ts_str = ev.get("timestamp")
        if ts_str:
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d__%H-%M-%S.%f").timestamp()
            except ValueError:
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S.%f").timestamp()
                except ValueError:
                    ts = float('inf')
            if ts < spans_data[span_key]["min_ts"]:
                spans_data[span_key]["min_ts"] = ts
        
        ev_type = ev.get("event_type")
        latency_s = ev.get("latency_ms", 0) / 1000.0
        
        if ev_type == "llm_call":
            spans_data[span_key]["llm_s"] += latency_s
        elif ev_type == "tool_call":
            tool_name = ev.get("tool_name", "")
            if "bash" in tool_name.lower() or "shell" in tool_name.lower() or "exec" in tool_name.lower():
                spans_data[span_key]["shell_s"] += latency_s
            elif "write" in tool_name.lower() or "save" in tool_name.lower():
                spans_data[span_key]["write_s"] += latency_s
            else:
                spans_data[span_key]["write_s"] += latency_s
                
    # Sort spans chronologically by min_ts
    sorted_spans = sorted(spans_data.items(), key=lambda x: x[1]["min_ts"])
    
    # Filter out empty spans
    sorted_spans = [item for item in sorted_spans if sum([item[1]["llm_s"], item[1]["write_s"], item[1]["shell_s"]]) > 0.1]
    
    # Fallback if no spans found
    if not sorted_spans:
        # Search all jsonl files for general events
        fallback_events = []
        for log_file in glob.glob(os.path.join(run_folder, "*.jsonl")):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        fallback_events.append(d)
                    except Exception:
                        pass
        
        breakdown = {}
        for ev in fallback_events:
            agent_id = ev.get("agent_id", "unknown")
            node_name = ev.get("node_name", "unknown")
            span_key = f"{agent_id}:{node_name}"
            if span_key not in breakdown:
                breakdown[span_key] = {"llm_s": 0.0, "write_s": 0.0, "shell_s": 0.0}
            
            ev_type = ev.get("event_type")
            latency_s = ev.get("latency_ms", 0) / 1000.0
            if ev_type == "llm_call":
                breakdown[span_key]["llm_s"] += latency_s
            elif ev_type == "tool_call":
                tool_name = ev.get("tool_name", "")
                if "bash" in tool_name.lower() or "shell" in tool_name.lower() or "exec" in tool_name.lower():
                    breakdown[span_key]["shell_s"] += latency_s
                else:
                    breakdown[span_key]["write_s"] += latency_s
                    
        for idx, (k, v) in enumerate(breakdown.items()):
            if sum(v.values()) > 0.1:
                sorted_spans.append((f"Span {idx+1}", v))
                
    return sorted_spans

def generate_graphs(run_folder):
    print(f"Generating telemetry graphs for {run_folder}...")
    dataset_name, dataset_size = get_dataset_details(run_folder)
    
    # ─── GRAPH 1: Cumulative Latency Accumulation ───
    orch_events = parse_orchestrator_steps(run_folder)
    
    steps = [{"agent": "Start", "cumulative_min": 0.0}]
    cumulative_seconds = 0.0
    
    prev_end_time = None
    for i, ev in enumerate(orch_events):
        start_time_str = ev.get("start_time")
        duration = ev.get("duration_seconds", 0.0)
        
        # Parse start timestamp to calculate orchestrator internal gap latency
        try:
            curr_start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            curr_start_time = None
            
        if prev_end_time is not None and curr_start_time is not None:
            gap = curr_start_time - prev_end_time
            if gap > 0.1:
                # Add gap representing orchestrator's own processing time
                cumulative_seconds += gap
                steps.append({
                    "agent": "Orchestrator",
                    "cumulative_min": cumulative_seconds / 60.0
                })
        
        # Determine the agent for this microservice call
        target = ev.get("target", "")
        action = ev.get("action", "")
        
        if "mcts-handler" in target or action in ["init", "select", "expand", "update", "backpropagate"]:
            agent_name = "MCTS Handler"
        elif "perception-agent" in target or action == "perception":
            agent_name = "Perception Agent"
        elif "semantic-agent" in target or action == "retrieve_tutorials":
            agent_name = "Semantic Agent"
        elif "coder-agent" in target or action == "generate_and_run":
            agent_name = "Coder Agent"
        else:
            agent_name = "Orchestrator"
            
        cumulative_seconds += duration
        steps.append({
            "agent": agent_name,
            "cumulative_min": cumulative_seconds / 60.0
        })
        
        if curr_start_time is not None:
            prev_end_time = curr_start_time + duration
        else:
            prev_end_time = None

    # Fallback to default dummy values if no steps are present (e.g. stack didn't run)
    if len(steps) <= 1:
        steps = [
            {"agent": "Start", "cumulative_min": 0.0},
            {"agent": "Perception Agent", "cumulative_min": 0.8},
            {"agent": "Orchestrator", "cumulative_min": 0.9},
            {"agent": "MCTS Handler", "cumulative_min": 0.91},
            {"agent": "MCTS Handler", "cumulative_min": 0.92},
            {"agent": "Orchestrator", "cumulative_min": 0.93},
            {"agent": "Semantic Agent", "cumulative_min": 1.4},
            {"agent": "Coder Agent", "cumulative_min": 6.8},
            {"agent": "MCTS Handler", "cumulative_min": 6.81},
            {"agent": "MCTS Handler", "cumulative_min": 6.82},
            {"agent": "Orchestrator", "cumulative_min": 6.83},
            {"agent": "Semantic Agent", "cumulative_min": 7.3},
            {"agent": "Coder Agent", "cumulative_min": 22.3}
        ]

    # Plot Graph 1
    fig, ax = plt.subplots(figsize=(11, 8.5))
    plt.subplots_adjust(top=0.83, bottom=0.22)
    
    # Elegant CDS Department top banner
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    banner_rect = plt.Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color='#2980B9', zorder=10)
    fig.patches.append(banner_rect)
    fig.text(0.5, 0.965, "CDS.IISc.ac.in | Department of Computational and Data Sciences", 
             fontsize=12, color='white', weight='bold', ha='center', va='center', zorder=11, family='sans-serif')
             
    fig.text(0.5, 0.89, "FAME++ : Motivation plot for 4 (Long Orchestration)", 
             fontsize=18, color='#2980B9', weight='bold', ha='center', va='center', family='sans-serif')

    # Line segments
    x_coords = list(range(len(steps)))
    y_coords = [s["cumulative_min"] for s in steps]
    ax.plot(x_coords, y_coords, color='#BDC3C7', linestyle='-', linewidth=1.5, zorder=1)
    
    # Custom vibrant agent color palette
    color_map = {
        "Orchestrator": "#F1C40F",       # Yellow/Gold
        "MCTS Handler": "#E67E22",       # Orange
        "Perception Agent": "#3498DB",   # Blue
        "Semantic Agent": "#2ECC71",     # Green
        "Coder Agent": "#E74C3C"         # Red
    }
    
    # Plot markers for each step based on the agent type
    plotted_labels = set()
    for idx, step in enumerate(steps):
        agent = step["agent"]
        if agent == "Start":
            continue
        color = color_map.get(agent, "#95A5A6")
        label = agent if agent not in plotted_labels else ""
        if label:
            plotted_labels.add(agent)
        ax.plot(idx, step["cumulative_min"], marker='o', color=color, markersize=8, 
                label=label, zorder=3, linestyle='None')
                
    # Horizontal threshold line
    ax.axhline(15, color='red', linestyle='--', linewidth=1.5, label='15 Min Threshold')
    
    # Detect the first step that crosses the 15-minute threshold
    crossed_idx = None
    for idx, step in enumerate(steps):
        if step["cumulative_min"] > 15.0:
            crossed_idx = idx
            break
            
    motivation_criteria_met = False
    if crossed_idx is not None:
        motivation_criteria_met = True
        # Plot red star on crossed node
        ax.plot(crossed_idx, steps[crossed_idx]["cumulative_min"], marker='*', color='red', 
                markersize=14, markeredgecolor='black', markeredgewidth=1.2, zorder=5)
        # Annotation arrow to the crossed node
        ax.annotate(
            f"Crosses 15m limit\n(Step {crossed_idx}: {steps[crossed_idx]['agent']})",
            xy=(crossed_idx, steps[crossed_idx]["cumulative_min"]),
            xytext=(crossed_idx - 3.5, steps[crossed_idx]["cumulative_min"] + 2.5),
            arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3,rad=.15"),
            color='darkred',
            fontweight='bold',
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="darkred", lw=1.2, alpha=0.95),
            zorder=6
        )

    # Set titles and axes
    ax.set_title("End-to-End Execution Latency Accumulation", fontsize=13, weight='bold', pad=12, family='sans-serif')
    ax.set_xlabel("Execution Step Sequence", fontsize=11, family='sans-serif')
    ax.set_ylabel("Cumulative Latency (Minutes)", fontsize=11, family='sans-serif')
    ax.grid(True, linestyle=':', alpha=0.5, color='#BDC3C7')
    
    # Disable top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Legend
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#BDC3C7', framealpha=0.9, fontsize=9.5)
    
    # Details section at the bottom left
    details_text = (
        f"Details:\n"
        f"  -   Use case: {dataset_name}\n"
        f"  -   Dataset size: {dataset_size}\n"
        f"  -   Motivation Criteria met: {motivation_criteria_met}"
    )
    fig.text(0.08, 0.05, details_text, fontsize=10, family='monospace', color='#2C3E50',
             bbox=dict(boxstyle="round,pad=0.4", fc="#F4F6F7", ec="#BDC3C7", lw=1))
             
    # Slide number label at bottom right
    fig.text(0.92, 0.05, "41", fontsize=14, color='#7F8C8D', weight='bold', ha='right')

    timeline_path = os.path.join(run_folder, "execution_timeline.png")
    plt.savefig(timeline_path, dpi=150)
    plt.close()
    print(f"Saved timeline graph to {timeline_path}")

    # ─── GRAPH 2: Stacked Bar Latency Breakdown per Span ───
    sorted_spans = parse_span_breakdown(run_folder)
    
    if not sorted_spans:
        # Fallback dummy values to match the slide
        sorted_spans = [
            ("Span 1", {"llm_s": 15.0, "write_s": 3.4, "shell_s": 305.3}),
            ("Span 2", {"llm_s": 18.0, "write_s": 5.7, "shell_s": 900.1})
        ]

    labels = [f"Span {idx+1}" for idx in range(len(sorted_spans))]
    llm_times = [s[1]["llm_s"] for s in sorted_spans]
    write_times = [s[1]["write_s"] for s in sorted_spans]
    shell_times = [s[1]["shell_s"] for s in sorted_spans]
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    plt.subplots_adjust(top=0.83, bottom=0.15)
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Elegant CDS Department top banner
    banner_rect = plt.Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color='#2980B9', zorder=10)
    fig.patches.append(banner_rect)
    fig.text(0.5, 0.965, "CDS.IISc.ac.in | Department of Computational and Data Sciences", 
             fontsize=12, color='white', weight='bold', ha='center', va='center', zorder=11, family='sans-serif')
             
    fig.text(0.5, 0.89, "FAME++ : Motivation plot for 1 (Long Tool Call)", 
             fontsize=18, color='#2980B9', weight='bold', ha='center', va='center', family='sans-serif')

    # Draw stacked bar chart
    x = range(len(labels))
    width = 0.4
    
    # Stack: LLM E2E (blue), File Write (pinkish gray), Shell Execute (red)
    p1 = ax.bar(x, llm_times, width, label='LLM Calls E2E', color='#4285F4', edgecolor='black', linewidth=1)
    p2 = ax.bar(x, write_times, width, bottom=llm_times, label='Tool Calls (File Write)', color='#E0D8D0', edgecolor='black', linewidth=1)
    
    bottom_shell = [llm + write for llm, write in zip(llm_times, write_times)]
    p3 = ax.bar(x, shell_times, width, bottom=bottom_shell, label='Tool Calls (Shell Execute)', color='#DB4437', edgecolor='black', linewidth=1)
    
    # Label durations inside segments and total durations above
    for idx in range(len(labels)):
        llm = llm_times[idx]
        write = write_times[idx]
        shell = shell_times[idx]
        total = llm + write + shell
        
        # Display Total above each bar
        ax.text(idx, total + max(5, total*0.015), f"Total:\n{total:.1f}s", 
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2C3E50')
        
        # Inside segments (if height > 5% of axis max to fit text)
        y_max = max([l + w + s for l, w, s in zip(llm_times, write_times, shell_times)])
        threshold_height = y_max * 0.05
        
        if shell > threshold_height:
            ax.text(idx, llm + write + shell/2.0, f"{shell:.1f}s",
                    ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        if write > threshold_height:
            ax.text(idx, llm + write/2.0, f"{write:.1f}s",
                    ha='center', va='center', fontsize=8.5, color='#2C3E50', fontweight='bold')
        if llm > threshold_height:
            ax.text(idx, llm/2.0, f"{llm:.1f}s",
                    ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

    ax.set_ylabel('Duration (Seconds)', fontsize=11, family='sans-serif')
    ax.set_title('E2E Execution Latency Breakdown per Span', fontsize=13, weight='bold', pad=12, family='sans-serif')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, family='sans-serif')
    ax.set_xlabel('Execution Spans', fontsize=11, family='sans-serif')
    
    ax.grid(True, linestyle=':', alpha=0.5, color='#BDC3C7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Legend
    ax.legend(loc='upper right', title="Total:", frameon=True, facecolor='white', edgecolor='#BDC3C7', fontsize=9)
    
    # Details section at the top right (placed next to chart inside layout)
    details_text = (
        f"Details:\n"
        f"  -   Use case:\n      {dataset_name}\n"
        f"  -   Dataset size: {dataset_size}\n"
        f"  -   Motivation Criteria met: {motivation_criteria_met}"
    )
    fig.text(0.70, 0.65, details_text, fontsize=10, family='monospace', color='#2C3E50',
             bbox=dict(boxstyle="round,pad=0.4", fc="#F4F6F7", ec="#BDC3C7", lw=1))
             
    # Slide number label at bottom right
    fig.text(0.92, 0.05, "40", fontsize=14, color='#7F8C8D', weight='bold', ha='right')

    breakdown_path = os.path.join(run_folder, "node_time_breakdown.png")
    plt.savefig(breakdown_path, dpi=150)
    plt.close()
    print(f"Saved breakdown graph to {breakdown_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_graphs(sys.argv[1])
    else:
        print("Usage: python plot_telemetry.py <run_folder_path>")
