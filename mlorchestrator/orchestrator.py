import os
import json
import time
import logging
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

def invoke_agent(url: str, payload: dict) -> dict:
    """
    Synchronously invokes an agent HTTP endpoint and parses the response.
    """
    try:
        # Most endpoints are A2A standard, except MCTS which is custom POST /invoke
        endpoint = f"{url}/invoke" if not url.endswith("/invoke") else url
        
        # Use HTTP POST for agent invocation
        response = httpx.post(endpoint, json=payload, timeout=18000.0)
        if response.status_code != 200:
            try:
                err_json = response.json()
                detail = err_json.get("error") or err_json.get("detail") or response.text
            except Exception:
                detail = response.text
            raise RuntimeError(f"Server error '{response.status_code} Internal Server Error' for url '{endpoint}'. Detail: {detail}")
        
        return response.json()
    except Exception as e:
        logger.error(f"Failed to invoke agent at {url}: {e}")
        raise e

def run_orchestration(run_id: str, input_data_folder: str, user_input: str, config: dict, max_iterations: int) -> dict:
    """
    Executes the MCTS Orchestration loop sequentially using synchronous HTTP invokes.
    Captures telemetry and writes a run report to local disk.
    """
    start_time_iso = datetime.utcnow().isoformat()
    start_time = time.time()
    
    # Resolve target URLs from environment
    perception_url = os.environ.get("PERCEPTION_URL", "http://perception-agent:8020")
    mcts_url = os.environ.get("MCTS_URL", "http://mcts-handler:8001")
    semantic_url = os.environ.get("SEMANTIC_URL", "http://semantic-agent:8088")
    coder_url = os.environ.get("CODER_URL", "http://coder-agent:8089")
    
    runs_dir = os.environ.get("RUNS_DIR", "/runs")
    run_folder = os.path.join(runs_dir, run_id)
    os.makedirs(run_folder, exist_ok=True)
    
    telemetry_file = os.path.join(run_folder, "orchestrator_telemetry.jsonl")
    
    # Telemetry logging setup
    telemetry_logs = []
    call_index = 0
    
    def log_call(target_url: str, action: str, input_payload: dict):
        nonlocal call_index
        call_index += 1
        call_start = time.time()
        call_start_iso = datetime.utcnow().isoformat()
        
        # Inject run_id into the payload for tracing
        if "tracing" not in input_payload:
            input_payload["tracing"] = {}
        input_payload["tracing"]["context_id"] = run_id
        input_payload["run_id"] = run_id
        
        telemetry_entry = {
            "call_index": call_index,
            "target": target_url,
            "action": action,
            "start_time": call_start_iso,
            "duration_seconds": 0.0,
            "status": "PENDING",
            "payload": input_payload,
            "response": None,
            "error": None
        }
        
        try:
            res = invoke_agent(target_url, input_payload)
            duration = time.time() - call_start
            telemetry_entry.update({
                "duration_seconds": round(duration, 3),
                "status": "SUCCESS",
                "response": res
            })
            
            telemetry_logs.append(telemetry_entry)
            with open(telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(telemetry_entry) + "\n")
                
            return res
        except Exception as e:
            duration = time.time() - call_start
            telemetry_entry.update({
                "duration_seconds": round(duration, 3),
                "status": "FAILED",
                "error": str(e)
            })
            
            telemetry_logs.append(telemetry_entry)
            with open(telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(telemetry_entry) + "\n")
                
            raise e

    try:
        # Step 1: PerceptionAgent
        logger.info("Executing Perception Agent...")
        perception_payload = {
            "skill": "analyze_task",
            "input_data_folder": input_data_folder,
            "user_input": user_input,
            "config": config
        }
        perception_results = log_call(perception_url, "perception", perception_payload)

        # Step 2: InitMCTS
        logger.info("Initializing MCTS tree...")
        init_payload = {
            "action": "init",
            "selected_tools": perception_results.get("selected_tools", ["machine learning"]),
            "config": config,
            "max_iterations": max_iterations
        }
        mcts_tree = log_call(mcts_url, "init", init_payload)

        # Main MCTS loop
        iteration = 0
        is_complete = False
        final_outcome = None
        
        while iteration < max_iterations:
            logger.info(f"--- MCTS Iteration {iteration} ---")
            
            # Step 3: SelectNode
            select_payload = {
                "action": "select",
                "mcts_tree": mcts_tree
            }
            current_selection = log_call(mcts_url, "select", select_payload)
            
            # Check loop termination
            if current_selection.get("is_complete") or mcts_tree.get("iteration", 0) >= max_iterations:
                logger.info("MCTS selection marked search complete or max iterations reached.")
                is_complete = True
                break
                
            # Step 4: ExpandNode
            expand_payload = {
                "action": "expand",
                "mcts_tree": mcts_tree,
                "current_selection": current_selection
            }
            expand_res = log_call(mcts_url, "expand", expand_payload)
            mcts_tree = expand_res["mcts_tree"]
            current_selection = expand_res["current_selection"]
            
            # Step 5: SemanticAgent
            semantic_payload = {
                "skill": "retrieve_tutorials",
                "config": config,
                "task_description": perception_results.get("task_description", ""),
                "current_tool": current_selection.get("current_tool", ""),
                "all_error_analyses": mcts_tree.get("all_error_analyses", []),
                "stage": current_selection.get("stage", "evolve"),
                "user_input": user_input,
                "data_prompt": perception_results.get("data_prompt", "")
            }
            semantic_results = log_call(semantic_url, "retrieve_tutorials", semantic_payload)
            
            # Step 6: CoderAgent
            parent_ctx = current_selection.get("parent_context", {})
            coder_payload = {
                "skill": "generate_and_run",
                "config": config,
                "task_description": perception_results.get("task_description", ""),
                "data_prompt": perception_results.get("data_prompt", ""),
                "user_input": user_input,
                "current_tool": current_selection.get("current_tool", ""),
                "tool_prompt": perception_results.get("tool_prompt", ""),
                "tutorial_prompt": semantic_results.get("tutorial_prompt", ""),
                "all_error_analyses": mcts_tree.get("all_error_analyses", []),
                "previous_python_code": parent_ctx.get("parent_code", ""),
                "previous_bash_script": parent_ctx.get("parent_bash", ""),
                "stage": current_selection.get("stage", "evolve"),
                "iteration": mcts_tree.get("iteration", 0),
                "node_id": current_selection.get("node_id")
            }
            coding_results = log_call(coder_url, "generate_and_run", coder_payload)
            
            # Step 7: UpdateNode
            update_payload = {
                "action": "update",
                "mcts_tree": mcts_tree,
                "current_selection": current_selection,
                "coding_results": coding_results
            }
            update_res = log_call(mcts_url, "update", update_payload)
            mcts_tree = update_res["mcts_tree"]
            current_selection = update_res["current_selection"]
            
            # Step 8: Backpropagate
            backprop_payload = {
                "action": "backpropagate",
                "mcts_tree": mcts_tree,
                "current_selection": current_selection
            }
            mcts_tree = log_call(mcts_url, "backpropagate", backprop_payload)
            
            iteration += 1

        # Step 9: FinalizeResults
        logger.info("Finalizing MCTS search results...")
        finalize_payload = {
            "action": "finalize",
            "mcts_tree": mcts_tree
        }
        final_outcome = log_call(mcts_url, "finalize", finalize_payload)
        
        status = "SUCCESS"
        error_msg = None

    except Exception as e:
        logger.exception("Orchestration failed with unexpected exception")
        status = "FAILED"
        error_msg = str(e)

    end_time_iso = datetime.utcnow().isoformat()
    total_duration = time.time() - start_time
    
    # Reload full MCTS tree from local storage for the report
    full_mcts_tree = None
    tree_path = os.path.join(run_folder, "mcts_tree.json")
    if os.path.exists(tree_path):
        with open(tree_path, "r", encoding="utf-8") as f:
            full_mcts_tree = json.load(f)
            
    report = {
        "run_id": run_id,
        "status": status,
        "error": error_msg,
        "orchestrator_start_time": start_time_iso,
        "orchestrator_end_time": end_time_iso,
        "total_duration_seconds": round(total_duration, 3),
        "input_parameters": {
            "input_data_folder": input_data_folder,
            "user_input": user_input,
            "max_iterations": max_iterations,
            "config": config
        },
        "final_outcome": final_outcome if 'final_outcome' in locals() else None,
        "telemetry_logs": telemetry_logs,
        "mcts_tree": full_mcts_tree
    }
    
    # Save report
    try:
        report_path = os.path.join(run_folder, "run_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved run report to {report_path}")
        
        if final_outcome and final_outcome.get("tree_visualization"):
            viz_path = os.path.join(run_folder, "mcts_tree.txt")
            with open(viz_path, "w", encoding="utf-8") as f:
                f.write(final_outcome["tree_visualization"])
    except Exception as e:
        logger.error(f"Failed to save report locally: {e}")
            
    return report
