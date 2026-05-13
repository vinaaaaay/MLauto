"""
LangGraph node functions for the Iterative Coding Module (MCTS).

Each function takes MLAutoState and returns a partial state update dict.
The MCTS tree is managed externally by NodeManager (stored on state["_node_manager"]).

Node functions:
  select_node           → MCTS selection (UCT)
  expand_node           → Create evolve/debug child
  retrieve_node_tutorials → Per-node tutorial retrieval
  rerank_node_tutorials → Per-node tutorial reranking
  generate_python_code  → CoderAgent(language="python")
  generate_bash_script  → CoderAgent(language="bash")
  execute_and_evaluate  → ExecuterAgent
  analyze_error         → ErrorAnalyzerAgent
  backpropagate         → MCTS backpropagation
"""

import logging
import os
import re

from shared.state import MLAutoState
from shared.llm import get_llm
from shared.logging_config import LLMCallLogger, log_state_snapshot
from shared.utils import extract_code, execute_in_docker
from shared.tool_registry import ToolRegistry, TutorialInfo

from .prompts import (
    PYTHON_CODER_PROMPT,
    BASH_CODER_PROMPT,
    EXECUTER_PROMPT,
    ERROR_ANALYZER_PROMPT,
    build_environment_prompt,
    build_validation_prompt,
)

# Import nodes from the dedicated memory modules
from semantic_memory.nodes import retrieve_tutorials as _semantic_retrieve
from episodic_memory.nodes import rerank_tutorials as _episodic_rerank

logger = logging.getLogger(__name__)


def _get_call_logger(state: MLAutoState) -> LLMCallLogger:
    """Create an LLMCallLogger pointing to the run's output directory."""
    output_folder = state.get("output_folder", "./output")
    return LLMCallLogger(output_folder)


def _get_node_manager(state: MLAutoState):
    """Get the NodeManager from state."""
    return state.get("_node_manager")


# ─── Node: select_node (MCTS Selection) ─────────────────────────────────

def select_node(state: MLAutoState) -> dict:
    """
    MCTS selection phase: traverse tree using UCT to find a node to expand.

    Returns:
        {"_selected_node": Node} — the node chosen for expansion
    """
    mgr = _get_node_manager(state)
    if mgr is None:
        logger.error("NodeManager not found in state")
        return {"is_complete": True}

    node = mgr.select_node()

    if node is None:
        logger.info("All nodes are terminal. Search complete.")
        return {"is_complete": True}

    mgr.current_node = node
    logger.info(f"─── select_node: selected Node {node.id} (stage={node.stage}, depth={node.depth}) ───")

    return {
        "node_id": node.id if node.id is not None else -1,
        "stage": node.stage,
        "depth": node.depth,
    }


# ─── Node: expand_node (MCTS Expansion) ─────────────────────────────────

def expand_node(state: MLAutoState) -> dict:
    """
    MCTS expansion: create a new child node (evolve or debug).

    Returns:
        {"node_id": int, "stage": str, "current_tool": str, "time_step": int}
    """
    mgr = _get_node_manager(state)
    if mgr is None:
        return {"is_complete": True}

    if state.get("is_complete"):
        return {}

    # Expand creates a new child node and generates code
    # We split this: expand creates the node, code gen is separate
    parent = mgr.current_node

    if parent.stage == "root":
        child = mgr._create_evolve_node_only()
    elif parent.is_successful:
        child = mgr._create_evolve_node_only()
    else:
        child = mgr._create_debug_node_only()

    logger.info(f"─── expand_node: created Node {child.id} (stage={child.stage}, "
                f"tool={child.tool_used}, depth={child.depth}) ───")

    # Get tool-specific prompt
    config = state.get("config", {})
    registry = ToolRegistry(config.get("tool_registry_path"))
    tool_prompt = registry.get_tool_prompt(child.tool_used)

    return {
        "node_id": child.id,
        "stage": child.stage,
        "current_tool": child.tool_used,
        "tool_prompt": tool_prompt,
        "time_step": mgr.time_step,
        "depth": child.depth,
    }


# ─── Node: retrieve_node_tutorials ───────────────────────────────────────

def retrieve_node_tutorials(state: MLAutoState) -> dict:
    """
    Per-node tutorial retrieval — delegates to semantic_memory module.

    Returns:
        {"tutorial_retrieval": list[TutorialInfo]}
    """
    if state.get("is_complete"):
        return {}

    mgr = _get_node_manager(state)
    logger.info(f"─── retrieve_node_tutorials: delegating to semantic_memory ───")

    # Delegate to the semantic memory module
    result = _semantic_retrieve(state)

    # Store on MCTS node
    if mgr and mgr.current_node:
        mgr.current_node.tutorial_retrieval = str(result.get("tutorial_retrieval", []))

    return result


# ─── Node: rerank_node_tutorials ─────────────────────────────────────────

def rerank_node_tutorials(state: MLAutoState) -> dict:
    """
    Per-node tutorial reranking — delegates to episodic_memory module.

    Returns:
        {"tutorial_prompt": str}
    """
    if state.get("is_complete"):
        return {}

    mgr = _get_node_manager(state)
    logger.info(f"─── rerank_node_tutorials: delegating to episodic_memory ───")

    # Delegate to the episodic memory module
    result = _episodic_rerank(state)

    # Store on MCTS node
    if mgr and mgr.current_node:
        mgr.current_node.tutorial_prompt = result.get("tutorial_prompt", "")

    return result


# ─── Node: generate_python_code ──────────────────────────────────────────

def generate_python_code(state: MLAutoState) -> dict:
    """
    Generate the Python ML training script.

    Returns:
        {"python_code": str, "python_file_path": str}
    """
    if state.get("is_complete"):
        return {}

    mgr = _get_node_manager(state)
    node_id = state.get("node_id", 0)
    iteration = state.get("iteration", 0)
    logger.info(f"─── generate_python_code [node {node_id}, iter {iteration}] ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)
    mcts_config = config.get("mcts", {})
    continuous_improvement = mcts_config.get("continuous_improvement", True)

    # Build code improvement / debug context
    code_improvement_prompt = ""
    if mgr and mgr.current_node:
        node = mgr.current_node
        if node.stage == "debug" and node.parent and node.parent.python_code:
            code_improvement_prompt = f"""\
### Previous Code to Debug
```python
{node.parent.python_code}
```
Please fix the errors in the code above. Make minimal changes necessary to fix the issues.
"""
            logger.info("  Mode: DEBUGGING previous code")
        elif node.stage == "evolve" and node.parent and node.parent.python_code:
            code_improvement_prompt = f"""\
### Previous Code to Improve
```python
{node.parent.python_code}
```
Please prioritize model architecture improvements and training optimization to enhance performance.
"""
            logger.info("  Mode: IMPROVING previous code")

    # Validation prompt
    validation_prompt = build_validation_prompt(continuous_improvement)

    # Format all previous error analyses
    all_error_analyses = "\n\n".join(state.get("all_error_analyses", []))

    # Get output folder for this node
    output_folder = state.get("output_folder", "./output")
    if mgr and mgr.current_node:
        iter_folder = os.path.join(output_folder, f"node_{mgr.current_node.id}")
        per_iter_output = os.path.join(iter_folder, "output")
        os.makedirs(per_iter_output, exist_ok=True)
    else:
        iter_folder = os.path.join(output_folder, f"iteration_{iteration}")
        per_iter_output = iter_folder
        os.makedirs(iter_folder, exist_ok=True)

    prompt = PYTHON_CODER_PROMPT.format(
        current_tool=state.get("current_tool", ""),
        output_folder=per_iter_output,
        tool_prompt=state.get("tool_prompt", ""),
        code_improvement_prompt=code_improvement_prompt,
        validation_prompt=validation_prompt,
        task_description=state.get("task_description", ""),
        data_prompt=state.get("data_prompt", ""),
        user_input=state.get("user_input", ""),
        all_error_analyses=all_error_analyses or "None",
        tutorial_prompt=state.get("tutorial_prompt", "") or "None",
    )

    response_text = call_logger.call(llm, prompt, node_name=f"generate_python_code[node={node_id}]")
    python_code = extract_code(response_text, language="python")

    # Save to file
    python_file_path = os.path.join(iter_folder, "generated_code.py")
    os.makedirs(os.path.dirname(python_file_path), exist_ok=True)
    with open(python_file_path, "w") as f:
        f.write(python_code)

    logger.info(f"  Python code saved: {python_file_path} ({len(python_code)} chars)")

    # Store on MCTS node
    if mgr and mgr.current_node:
        mgr.current_node.python_code = python_code

    return {"python_code": python_code, "python_file_path": python_file_path}


# ─── Node: generate_bash_script ──────────────────────────────────────────

def generate_bash_script(state: MLAutoState) -> dict:
    """
    Generate the bash execution script.

    Returns:
        {"bash_script": str}
    """
    if state.get("is_complete"):
        return {}

    mgr = _get_node_manager(state)
    node_id = state.get("node_id", 0)
    iteration = state.get("iteration", 0)
    logger.info(f"─── generate_bash_script [node {node_id}, iter {iteration}] ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    output_folder = state.get("output_folder", "./output")
    current_tool = state.get("current_tool", "")

    if mgr and mgr.current_node:
        iter_folder = os.path.join(output_folder, f"node_{mgr.current_node.id}")
    else:
        iter_folder = os.path.join(output_folder, f"iteration_{iteration}")
    os.makedirs(iter_folder, exist_ok=True)

    # Get requirements files from tool registry
    registry = ToolRegistry(config.get("tool_registry_path"))
    try:
        common_env_file = str(registry.get_common_requirements_file())
        tool_env_file = str(registry.get_tool_requirements_file(current_tool))
    except FileNotFoundError:
        common_env_file = ""
        tool_env_file = ""

    # Determine if env configuration is needed
    configure_env = current_tool.lower() in ["machine learning", "huggingface", "fairseq"]

    environment_prompt = build_environment_prompt(
        iteration_folder=iter_folder,
        current_tool=current_tool,
        common_env_file=common_env_file,
        tool_env_file=tool_env_file,
        configure_env=configure_env,
    )

    all_error_analyses = "\n\n".join(state.get("all_error_analyses", []))

    # Get previous bash script from parent node
    previous_bash = ""
    if mgr and mgr.current_node and mgr.current_node.parent:
        previous_bash = mgr.current_node.parent.bash_script

    prompt = BASH_CODER_PROMPT.format(
        environment_prompt=environment_prompt,
        python_file_path=state.get("python_file_path", ""),
        python_code=state.get("python_code", ""),
        all_error_analyses=all_error_analyses or "None",
        previous_bash_script=previous_bash or "None",
    )

    response_text = call_logger.call(llm, prompt, node_name=f"generate_bash_script[node={node_id}]")
    bash_script = extract_code(response_text, language="bash")

    # Save to file
    bash_file_path = os.path.join(iter_folder, "execution_script.sh")
    with open(bash_file_path, "w") as f:
        f.write(bash_script)

    logger.info(f"  Bash script saved: {bash_file_path} ({len(bash_script)} chars)")

    # Store on MCTS node
    if mgr and mgr.current_node:
        mgr.current_node.bash_script = bash_script

    return {"bash_script": bash_script}


# ─── Node: execute_and_evaluate ──────────────────────────────────────────

def execute_and_evaluate(state: MLAutoState) -> dict:
    """
    Execute the generated bash script inside a Docker container,
    then use the LLM to evaluate results.

    Returns:
        dict with stdout, stderr, decision, error_summary, validation_score, etc.
    """
    if state.get("is_complete"):
        return {}

    mgr = _get_node_manager(state)
    node_id = state.get("node_id", 0)
    iteration = state.get("iteration", 0)
    logger.info(f"─── execute_and_evaluate [node {node_id}, iter {iteration}] ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)
    timeout = config.get("execution", {}).get("timeout", 3600)
    docker_image = config.get("execution", {}).get("docker_image", "mlauto-executor:latest")

    output_folder = os.path.abspath(state.get("output_folder", "./output"))
    input_folder = os.path.abspath(state.get("input_data_folder", "."))

    if mgr and mgr.current_node:
        iter_folder = os.path.join(output_folder, f"node_{mgr.current_node.id}")
    else:
        iter_folder = os.path.join(output_folder, f"iteration_{iteration}")

    # Save bash script
    bash_file_path = os.path.join(iter_folder, "execution_script.sh")
    os.makedirs(iter_folder, exist_ok=True)
    with open(bash_file_path, "w") as f:
        f.write(state.get("bash_script", ""))

    logger.info(f"  Docker image: {docker_image}")
    logger.info(f"  Timeout: {timeout}s")

    # Execute inside Docker
    success, stdout, stderr = execute_in_docker(
        bash_script_path=bash_file_path,
        input_data_folder=input_folder,
        output_folder=output_folder,
        docker_image=docker_image,
        timeout=timeout,
    )

    logger.info(f"  Execution {'SUCCEEDED' if success else 'FAILED'}")

    # Save raw output
    with open(os.path.join(iter_folder, "stdout.txt"), "w") as f:
        f.write(stdout)
    with open(os.path.join(iter_folder, "stderr.txt"), "w") as f:
        f.write(stderr)

    # Truncate for LLM
    def truncate_start(text, max_len=8192):
        if len(text) > max_len:
            return f"[...TRUNCATED ({len(text) - max_len} chars)...]\n" + text[-max_len:]
        return text

    # Ask LLM to evaluate
    prompt = EXECUTER_PROMPT.format(
        task_description=state.get("task_description", ""),
        data_prompt=state.get("data_prompt", ""),
        python_code=state.get("python_code", ""),
        stdout=truncate_start(stdout) or "No standard output",
        stderr=truncate_start(stderr) or "No standard error",
    )

    content = call_logger.call(llm, prompt, node_name=f"execute_and_evaluate[node={node_id}]")

    # Parse decision
    decision = "FIX"
    if "DECISION:" in content:
        for line in content.split("\n"):
            if "DECISION:" in line:
                if "SUCCESS" in line.upper():
                    decision = "SUCCESS"
                break

    # Parse error summary
    error_summary = None
    if "ERROR_SUMMARY:" in content:
        es = content.split("ERROR_SUMMARY:")[1].strip().split("\n")[0].strip()
        if es.lower() != "none" and es:
            error_summary = es

    # Parse validation score
    validation_score = None
    if "VALIDATION_SCORE:" in content:
        vs_text = content.split("VALIDATION_SCORE:")[1].strip().split("\n")[0].strip()
        if vs_text.lower() != "none" and vs_text:
            try:
                validation_score = float(vs_text)
            except ValueError:
                pass
    if decision != "SUCCESS":
        validation_score = None

    # Build error message
    error_message = ""
    if stderr:
        error_message = f"stderr: {stderr}\n\n"
    if error_summary:
        error_message += f"Error summary: {error_summary}"

    # Update MCTS node
    if mgr and mgr.current_node:
        mgr.current_node.stdout = stdout
        mgr.current_node.stderr = stderr
        mgr.current_node.validation_score = validation_score
        mgr.current_node.error_message = error_message

        if decision == "SUCCESS":
            mgr.current_node.is_successful = True

    # Track best score
    best_score = state.get("best_score")
    best_code = state.get("best_code", "")
    best_node_id = state.get("best_node_id")
    if validation_score is not None:
        if best_score is None or validation_score > best_score:
            best_score = validation_score
            best_code = state.get("python_code", "")
            best_node_id = node_id
            logger.info(f"  ★ New best score: {best_score}")

    logger.info(f"  Decision: {decision}")
    logger.info(f"  Validation score: {validation_score}")
    logger.info(f"  Best score so far: {best_score}")

    return {
        "stdout": stdout,
        "stderr": stderr,
        "decision": decision,
        "error_summary": error_summary,
        "validation_score": validation_score,
        "error_message": error_message,
        "best_score": best_score,
        "best_code": best_code,
        "best_node_id": best_node_id,
        "iteration": iteration + 1,
    }


# ─── Node: analyze_error ────────────────────────────────────────────────

def analyze_error(state: MLAutoState) -> dict:
    """
    Analyze the execution error and produce debugging suggestions.

    Returns:
        {"error_analysis": str, "all_error_analyses": list[str]}
    """
    if state.get("is_complete"):
        return {}

    mgr = _get_node_manager(state)
    node_id = state.get("node_id", 0)
    iteration = state.get("iteration", 0)
    logger.info(f"─── analyze_error [node {node_id}, iter {iteration}] ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    prompt = ERROR_ANALYZER_PROMPT.format(
        error_message=state.get("error_message", "No error message available."),
        task_description=state.get("task_description", ""),
        data_prompt=state.get("data_prompt", ""),
        user_input=state.get("user_input", ""),
        python_code=state.get("python_code", ""),
        bash_script=state.get("bash_script", ""),
    )

    content = call_logger.call(llm, prompt, node_name=f"analyze_error[node={node_id}]")

    # Parse error analysis
    analysis_match = re.search(r"ERROR_SUMMARY:\s*(.*)", content, re.DOTALL)
    if analysis_match:
        error_analysis = f"ERROR_SUMMARY: {analysis_match.group(1).strip()}"
    else:
        error_analysis = "Failed to extract error analysis from LLM response."

    # Accumulate
    all_analyses = list(state.get("all_error_analyses", []))
    all_analyses.append(error_analysis)

    # Store on MCTS node
    if mgr and mgr.current_node:
        mgr.current_node.error_analysis = error_analysis

    logger.info(f"  Error analysis: {error_analysis[:300]}")

    return {
        "error_analysis": error_analysis,
        "all_error_analyses": all_analyses,
    }


# ─── Node: backpropagate (MCTS Backpropagation) ─────────────────────────

def backpropagate(state: MLAutoState) -> dict:
    """
    MCTS backpropagation: update statistics up the tree.

    Returns:
        {} (side-effects on NodeManager tree)
    """
    if state.get("is_complete"):
        return {}

    mgr = _get_node_manager(state)
    if mgr is None:
        return {}

    node = mgr.current_node
    decision = state.get("decision", "FIX")
    validation_score = state.get("validation_score")

    is_failure = decision != "SUCCESS"
    is_validated = validation_score is not None

    logger.info(f"─── backpropagate [node {node.id}]: decision={decision}, "
                f"score={validation_score}, failure={is_failure} ───")

    # Update validation score tracking on manager
    if validation_score is not None:
        if mgr._best_validation_score is None or validation_score > mgr._best_validation_score:
            mgr._best_node = node
            mgr._best_validation_score = validation_score
        if mgr._worst_validation_score is None:
            mgr._worst_validation_score = validation_score
        else:
            mgr._worst_validation_score = min(mgr._worst_validation_score, validation_score)

    # Handle successful debug nodes: promote to parent level
    if decision == "SUCCESS" and node.stage == "debug":
        debug_origin = mgr._find_debug_origin(node)
        if debug_origin:
            node.parent.remove_child(node)
            node.parent = debug_origin.parent
            debug_origin.parent.add_child(node)
            mgr.mark_node_terminal(debug_origin)
            node.is_debug_successful = True
            logger.info(f"  Promoted debug node {node.id}, marked origin {debug_origin.id} terminal")

    # Handle failed debug: increment parent debug attempts
    if is_failure and node.stage == "debug" and node.parent:
        node.parent.debug_attempts += 1
        if node.parent.debug_attempts >= mgr.max_debug_depth:
            mgr.mark_node_terminal(node.parent)
            logger.warning(f"  Parent {node.parent.id} reached max debug depth, marked terminal")

    # Backpropagate up the tree
    current = node
    while current is not None:
        current.update(validation_score, is_validated, is_failure)
        current = current.parent

    return {}
