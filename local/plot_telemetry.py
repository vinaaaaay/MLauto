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
    
    # First pass: map span_id to iteration or node_id
    span_to_key = {}
    for ev in events:
        span_id = ev.get("span_id")
        if not span_id:
            continue
        node_id = ev.get("node_id")
        iteration = ev.get("iteration")
        key = node_id if node_id is not None else iteration
        if key is not None:
            span_to_key[span_id] = key

    # Second pass: group by span key
    spans_data = {}
    for ev in events:
        span_id = ev.get("span_id")
        span_key = span_to_key.get(span_id) if span_id else None
        
        # Fallback to direct event attributes
        if span_key is None:
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
    
    # Do not filter out empty spans, so we get a bar for every node.
    # sorted_spans = [item for item in sorted_spans if sum([item[1]["llm_s"], item[1]["write_s"], item[1]["shell_s"]]) > 0.1]
    
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
            # Include all nodes without filtering by threshold
            sorted_spans.append((f"Node {idx}", v))
                
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

    # Print error and exit if no execution steps are present (e.g. stack didn't run)
    if len(steps) <= 1:
        print(f"Error: No execution steps found in orchestrator_telemetry.jsonl for run folder: {run_folder}")
        return

    # Plot Graph 1
    fig, ax = plt.subplots(figsize=(11, 8.5))
    plt.subplots_adjust(top=0.92, bottom=0.22, right=0.80)
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

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
    
    # Legend placed outside the axis on the right
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, facecolor='white', edgecolor='#BDC3C7', framealpha=0.9, fontsize=9.5)
    
    # Details section placed in the bottom-left of the figure to avoid any overlap with the plot
    details_text = (
        f"Details:\n"
        f"  - Use case: {dataset_name}\n"
        f"  - Dataset size: {dataset_size}"
    )
    fig.text(0.06, 0.04, details_text, transform=fig.transFigure, fontsize=10, 
             family='monospace', color='#2C3E50', horizontalalignment='left',
             verticalalignment='bottom', zorder=5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#BDC3C7'))

    timeline_path = os.path.join(run_folder, "execution_timeline.png")
    fig.savefig(timeline_path, dpi=150)
    plt.close(fig)
    print(f"Saved timeline graph to {timeline_path}")

    # ─── GRAPH 2: Stacked Bar Latency Breakdown per Span ───
    sorted_spans = parse_span_breakdown(run_folder)
    
    if not sorted_spans:
        print(f"Warning: No span details found in coder_metrics.jsonl. Falling back to internal node state timing.")

    # Load MCTS node metadata if available
    nodes = {}
    tree_file = os.path.join(run_folder, "mcts_tree.json")
    if os.path.exists(tree_file):
        try:
            with open(tree_file, "r", encoding="utf-8") as f:
                tree_data = json.load(f)
                nodes_raw = tree_data.get("nodes", {})
                for k, v in nodes_raw.items():
                    nodes[str(k)] = {
                        "node_id": int(v.get("node_id", k)),
                        "parent_id": v.get("parent_id"),
                        "child_ids": list(v.get("child_ids", [])),
                        "stage": v.get("stage", "evolve"),
                        "validation_score": v.get("validation_score"),
                        "execution_time": v.get("execution_time", 0.0),
                        "ai_call_time": v.get("ai_call_time", 0.0)
                    }
        except Exception as e:
            print(f"Error parsing mcts_tree.json: {e}")

    # Ensure virtual root exists
    if nodes and "-1" not in nodes:
        children_of_root = []
        for nid, ninfo in nodes.items():
            if ninfo.get("parent_id") is None or ninfo.get("parent_id") == -1:
                children_of_root.append(int(nid))
                ninfo["parent_id"] = -1
        
        nodes["-1"] = {
            "node_id": -1,
            "parent_id": None,
            "child_ids": children_of_root,
            "stage": "root",
            "validation_score": None,
            "execution_time": 0.0,
            "ai_call_time": 0.0
        }

    # Fallback to mcts_tree.txt if JSON parsing was not successful or returned empty
    if not nodes:
        tree_txt_path = os.path.join(run_folder, "mcts_tree.txt")
        if os.path.exists(tree_txt_path):
            try:
                with open(tree_txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                level_parents = {0: "-1"}
                nodes["-1"] = {
                    "node_id": -1,
                    "parent_id": None,
                    "child_ids": [],
                    "stage": "root",
                    "validation_score": None,
                    "execution_time": 0.0,
                    "ai_call_time": 0.0
                }
                
                for line in lines:
                    if not line.strip():
                        continue
                    m_node = re.search(r"(Node\s+(\d+))", line)
                    if m_node:
                        full_node_name, node_id = m_node.groups()
                        pos = line.index(full_node_name)
                        level = max(1, (pos // 4) - 1)
                        
                        stage = "evolve"
                        validation_score = None
                        m_brackets = re.search(r"\[([^\]]+)\]", line)
                        if m_brackets:
                            parts = m_brackets.group(1).split("|")
                            if len(parts) >= 1:
                                stage = parts[0].strip().lower()
                            for part in parts:
                                part = part.strip()
                                if "score=" in part:
                                    try:
                                        validation_score = float(part.split("=")[1])
                                    except Exception:
                                        pass
                                elif part.replace(".", "").replace("e-", "").replace("e+", "").replace("-", "").replace("+", "").replace("0", "").isdigit() or "score=" not in part and "." in part:
                                    try:
                                        validation_score = float(part)
                                    except Exception:
                                        pass
                        
                        parent_level = level - 1
                        parent_id = level_parents.get(parent_level, "-1")
                        
                        nodes[str(node_id)] = {
                            "node_id": int(node_id),
                            "parent_id": int(parent_id) if parent_id != "-1" else -1,
                            "child_ids": [],
                            "stage": stage,
                            "validation_score": validation_score,
                            "execution_time": 0.0,
                            "ai_call_time": 0.0
                        }
                        
                        if str(parent_id) in nodes:
                            if int(node_id) not in nodes[str(parent_id)]["child_ids"]:
                                nodes[str(parent_id)]["child_ids"].append(int(node_id))
                        
                        level_parents[level] = str(node_id)
            except Exception as e:
                print(f"Error parsing mcts_tree.txt: {e}")

    # Print error and exit if no tree structure could be parsed
    if not nodes:
        print(f"Error: No tree structure could be parsed from mcts_tree.json or mcts_tree.txt in: {run_folder}")
        return

    # Map times from coder_metrics spans
    node_times = {}
    for s in sorted_spans:
        key_str = str(s[0]).replace("Node ", "")
        node_times[key_str] = s[1]

    # Symmetric horizontal division tree coordinate layout algorithm
    x_coords = {}
    y_coords = {}
    
    def get_max_depth(node_key, depth=0):
        node_info = nodes.get(str(node_key)) or {}
        children = node_info.get("child_ids", [])
        valid_children = [str(c) for c in children if str(c) in nodes]
        if not valid_children:
            return depth
        return max(get_max_depth(c, depth + 1) for c in valid_children)
        
    def layout_tree(node_key, x_min=0.0, x_max=1.0, depth=0, depth_step=0.2):
        x_coords[str(node_key)] = (x_min + x_max) / 2.0
        y_coords[str(node_key)] = 1.0 - (depth * depth_step)
        
        node_info = nodes.get(str(node_key)) or {}
        children = node_info.get("child_ids", [])
        valid_children = [str(c) for c in children if str(c) in nodes]
        if not valid_children:
            return
            
        num_children = len(valid_children)
        width = (x_max - x_min) / num_children
        for idx, child in enumerate(valid_children):
            c_min = x_min + idx * width
            c_max = c_min + width
            layout_tree(child, c_min, c_max, depth + 1, depth_step)
            
    # Calculate coords starting from Root (-1) with dynamic depth step scaling
    max_d = get_max_depth("-1", 0)
    depth_step = 0.8 / max(1, max_d)
    layout_tree("-1", 0.05, 0.95, 0, depth_step)
    
    fig, ax = plt.subplots(figsize=(15.5, 11.0))
    plt.subplots_adjust(top=0.91, bottom=0.18, left=0.04, right=0.96)
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    pretty_dataset = dataset_name.replace("-", " ").title()
    ax.set_title("E2E Execution Latency Breakdown per Node", fontsize=14, weight='bold', pad=15, family='sans-serif', color='#2C3E50')
    
    # 1. Draw connection edges and stage labels (E/D) at midpoints
    for node_key, node_info in nodes.items():
        if str(node_key) not in x_coords:
            continue
        children = node_info.get("child_ids", [])
        p_x = x_coords[str(node_key)]
        p_y = y_coords[str(node_key)]
        
        for child in children:
            c_key = str(child)
            if c_key not in x_coords:
                continue
            c_x = x_coords[c_key]
            c_y = y_coords[c_key]
            
            # Draw line connecting parent to child
            ax.plot([p_x, c_x], [p_y, c_y], color='#CFD8DC', linestyle='-', linewidth=2.0, zorder=1)
            
            # Determine E/D stage for the child edge
            child_info = nodes.get(c_key) or {}
            stage_str = child_info.get("stage", "evolve")
            stage_abbr = "E" if stage_str.lower() == "evolve" else ("D" if stage_str.lower() == "debug" else stage_str.upper())
            
            # Midpoint edge label inside a crisp white circle
            mid_x = (p_x + c_x) / 2.0
            mid_y = (p_y + c_y) / 2.0
            ax.text(mid_x, mid_y, stage_abbr, ha='center', va='center', fontsize=9.5, fontweight='bold',
                    color='#37474F', zorder=3,
                    bbox=dict(boxstyle='circle,pad=0.25', facecolor='white', edgecolor='#B0BEC5', alpha=0.98, linewidth=1.2))
                    
    # 2. Draw node cards with styled color codes (Green for Success, Gray-Blue for Failure, Light-Gray for Root)
    for node_key, node_info in nodes.items():
        if str(node_key) not in x_coords:
            continue
        x = x_coords[str(node_key)]
        y = y_coords[str(node_key)]
        
        nid = node_info.get("node_id", -1)
        score_val = node_info.get("validation_score")
        
        if nid == -1:
            node_title = "MCTS Root"
            score_str = "Status: Root"
            time_str = "Orchestrator Init"
            card_face = '#F5F5F5'
            card_edge = '#9E9E9E'
        else:
            node_title = f"Node {nid}"
            score_str = f"Score: {score_val:.4f}" if isinstance(score_val, (int, float)) else "No Score"
            
            # Retrieve time stats for this node
            times = node_times.get(str(nid), {"llm_s": 0.0, "write_s": 0.0, "shell_s": 0.0})
            llm_s = times.get("llm_s", 0.0) or node_info.get("ai_call_time", 0.0) or 0.0
            shell_s = times.get("shell_s", 0.0) or node_info.get("execution_time", 0.0) or 0.0
            write_s = times.get("write_s", 0.0) or 0.0
            
            # Show original detailed latency breakdown matching user request
            time_str = (
                f"LLM Calls Time: {llm_s:.1f}s\n"
                f"Tool Call (File Write): {write_s:.1f}s\n"
                f"Tool Call (Shell Execute): {shell_s:.1f}s"
            )
            
            if isinstance(score_val, (int, float)):
                card_face = '#E8F5E9'  # Premium Light-Green for successful models
                card_edge = '#2ECC71'  # Green Border
            else:
                card_face = '#ECEFF1'  # Premium Gray-Blue for unsuccessful iterations
                card_edge = '#90A4AE'  # Dark Gray-Blue Border
                
        node_text = f"{node_title}\n{score_str}\n{time_str}"
        
        # Display the formatted card
        ax.text(x, y, node_text, ha='center', va='center', fontsize=9.0, fontweight='semibold',
                family='sans-serif', color='#2C3E50', zorder=2,
                bbox=dict(boxstyle='round,pad=0.65', facecolor=card_face, edgecolor=card_edge, alpha=0.98, linewidth=1.5))
                
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    
    # Legend for Tree Nodes (Green for Successful Model, Gray for Unsuccessful Model)
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2ECC71', label='Model Trained (Score)')
    gray_patch = mpatches.Patch(facecolor='#ECEFF1', edgecolor='#90A4AE', label='Iteration Failed (No Score)')
    root_patch = mpatches.Patch(facecolor='#F5F5F5', edgecolor='#9E9E9E', label='MCTS Root')
    ax.legend(handles=[green_patch, gray_patch, root_patch], loc='upper right', frameon=True, facecolor='white', edgecolor='#BDC3C7', fontsize=9.5)
    
    # Details section placed cleanly in the bottom-left of the canvas
    details_text = (
        f"Details:\n"
        f"  - Use case: {dataset_name}\n"
        f"  - Dataset size: {dataset_size}"
    )
    fig.text(0.06, 0.04, details_text, transform=fig.transFigure, fontsize=10, 
             family='monospace', color='#2C3E50', horizontalalignment='left',
             verticalalignment='bottom', zorder=5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#BDC3C7'))

    breakdown_path = os.path.join(run_folder, "node_time_breakdown.png")
    fig.savefig(breakdown_path, dpi=150)
    plt.close(fig)
    print(f"Saved breakdown graph to {breakdown_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_graphs(sys.argv[1])
    else:
        print("Usage: python plot_telemetry.py <run_folder_path>")
