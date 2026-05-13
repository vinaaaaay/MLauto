"""
LangGraph node functions for the Iterative Coding Module.

Each function takes MLAutoState and returns a partial state update dict.
These map 1:1 to the original autogluon-assistant agents:

  generate_python_code  → CoderAgent(language="python")
  generate_bash_script  → CoderAgent(language="bash")
  execute_and_evaluate  → ExecuterAgent  (runs inside Docker container)
  analyze_error         → ErrorAnalyzerAgent
"""

import logging
import os
import re

from shared.state import MLAutoState
from shared.llm import get_llm
from shared.logging_config import LLMCallLogger, log_state_snapshot
from shared.utils import extract_code, execute_in_docker

from .prompts import (
    PYTHON_CODER_PROMPT,
    BASH_CODER_PROMPT,
    EXECUTER_PROMPT,
    ERROR_ANALYZER_PROMPT,
    build_environment_prompt,
)

logger = logging.getLogger(__name__)


def _get_call_logger(state: MLAutoState) -> LLMCallLogger:
    """Create an LLMCallLogger pointing to the run's output directory."""
    output_folder = state.get("output_folder", "./output")
    return LLMCallLogger(output_folder)


# ─── Node: generate_python_code ──────────────────────────────────────────

def generate_python_code(state: MLAutoState) -> dict:
    """
    Generate the Python ML training script.

    Maps to: CoderAgent(language="python").__call__()

    Returns:
        {"python_code": str, "python_file_path": str}
    """
    iteration = state.get("iteration", 0)
    logger.info(f"─── generate_python_code [iteration {iteration}] ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    # Build code improvement / debug context
    code_improvement_prompt = ""
    if iteration > 0:
        prev_code = state.get("previous_python_code", "")
        decision = state.get("decision", "")
        if decision == "FIX" and prev_code:
            code_improvement_prompt = f"""\
### Previous Code to Debug
```python
{prev_code}
```
Please fix the errors in the code above. Make minimal changes necessary to fix the issues.
"""
            logger.info("  Mode: DEBUGGING previous code")
        elif prev_code:
            code_improvement_prompt = f"""\
### Previous Code to Improve
```python
{prev_code}
```
Please prioritize model architecture improvements and training optimization to enhance performance.
"""
            logger.info("  Mode: IMPROVING previous code")

    # Format all previous error analyses
    all_error_analyses = "\n\n".join(state.get("all_error_analyses", []))

    prompt = PYTHON_CODER_PROMPT.format(
        current_tool=state.get("current_tool", ""),
        output_folder=state.get("output_folder", "./output"),
        tool_prompt=state.get("tool_prompt", ""),
        code_improvement_prompt=code_improvement_prompt,
        task_description=state.get("task_description", ""),
        data_prompt=state.get("data_prompt", ""),
        user_input=state.get("user_input", ""),
        all_error_analyses=all_error_analyses or "None",
    )

    response_text = call_logger.call(llm, prompt, node_name=f"generate_python_code[iter={iteration}]")
    python_code = extract_code(response_text, language="python")

    # Save to file
    output_folder = state.get("output_folder", "./output")
    iter_folder = os.path.join(output_folder, f"iteration_{iteration}")
    os.makedirs(iter_folder, exist_ok=True)
    python_file_path = os.path.join(iter_folder, "generated_code.py")
    with open(python_file_path, "w") as f:
        f.write(python_code)

    logger.info(f"  Python code saved: {python_file_path} ({len(python_code)} chars, {python_code.count(chr(10))} lines)")

    return {"python_code": python_code, "python_file_path": python_file_path}


# ─── Node: generate_bash_script ──────────────────────────────────────────

def generate_bash_script(state: MLAutoState) -> dict:
    """
    Generate the bash execution script (runs inside Docker container).

    Maps to: CoderAgent(language="bash").__call__()

    Returns:
        {"bash_script": str}
    """
    iteration = state.get("iteration", 0)
    logger.info(f"─── generate_bash_script [iteration {iteration}] ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)

    output_folder = state.get("output_folder", "./output")
    iter_folder = os.path.join(output_folder, f"iteration_{iteration}")

    environment_prompt = build_environment_prompt(
        iteration_folder=iter_folder,
        current_tool=state.get("current_tool", ""),
    )

    all_error_analyses = "\n\n".join(state.get("all_error_analyses", []))

    prompt = BASH_CODER_PROMPT.format(
        environment_prompt=environment_prompt,
        python_file_path=state.get("python_file_path", ""),
        python_code=state.get("python_code", ""),
        all_error_analyses=all_error_analyses or "None",
        previous_bash_script=state.get("previous_bash_script", "") or "None",
    )

    response_text = call_logger.call(llm, prompt, node_name=f"generate_bash_script[iter={iteration}]")
    bash_script = extract_code(response_text, language="bash")

    # Save to file
    os.makedirs(iter_folder, exist_ok=True)
    bash_file_path = os.path.join(iter_folder, "execution_script.sh")
    with open(bash_file_path, "w") as f:
        f.write(bash_script)

    logger.info(f"  Bash script saved: {bash_file_path} ({len(bash_script)} chars)")
    logger.debug(f"  Bash script content:\n{bash_script}")

    return {"bash_script": bash_script}


# ─── Node: execute_and_evaluate ──────────────────────────────────────────

def execute_and_evaluate(state: MLAutoState) -> dict:
    """
    Execute the generated bash script inside a Docker container,
    then use the LLM to evaluate results.

    Maps to: ExecuterAgent.__call__()

    Returns:
        {
            "stdout": str, "stderr": str,
            "decision": str, "error_summary": str | None,
            "validation_score": float | None,
            "error_message": str,
            "iteration": int,
        }
    """
    iteration = state.get("iteration", 0)
    logger.info(f"─── execute_and_evaluate [iteration {iteration}] ───")

    config = state.get("config", {})
    llm = get_llm(config.get("llm"))
    call_logger = _get_call_logger(state)
    timeout = config.get("execution", {}).get("timeout", 3600)
    docker_image = config.get("execution", {}).get("docker_image", "mlauto-executor:latest")

    # Build paths
    output_folder = os.path.abspath(state.get("output_folder", "./output"))
    input_folder = os.path.abspath(state.get("input_data_folder", "."))
    iter_folder = os.path.join(output_folder, f"iteration_{iteration}")

    # Save bash script to the iteration folder so Docker can access it
    bash_file_path = os.path.join(iter_folder, "execution_script.sh")
    os.makedirs(iter_folder, exist_ok=True)
    with open(bash_file_path, "w") as f:
        f.write(state.get("bash_script", ""))

    logger.info(f"  Docker image: {docker_image}")
    logger.info(f"  Timeout: {timeout}s")
    logger.info(f"  Input mount: {input_folder} → /workspace/data")
    logger.info(f"  Output mount: {output_folder} → /workspace/output")
    logger.info(f"  Script: {bash_file_path}")

    # Execute inside Docker
    success, stdout, stderr = execute_in_docker(
        bash_script_path=bash_file_path,
        input_data_folder=input_folder,
        output_folder=output_folder,
        docker_image=docker_image,
        timeout=timeout,
    )

    logger.info(f"  Execution {'SUCCEEDED' if success else 'FAILED'}")
    logger.info(f"  stdout: {len(stdout)} chars, stderr: {len(stderr)} chars")

    # Save raw execution output
    with open(os.path.join(iter_folder, "stdout.txt"), "w") as f:
        f.write(stdout)
    with open(os.path.join(iter_folder, "stderr.txt"), "w") as f:
        f.write(stderr)
    logger.debug(f"  Saved stdout/stderr to {iter_folder}")

    if stderr:
        logger.debug(f"  stderr (last 500 chars):\n{stderr[-500:]}")

    # Truncate for the LLM prompt (keep last 8192 chars)
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

    content = call_logger.call(llm, prompt, node_name=f"execute_and_evaluate[iter={iteration}]")

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

    # Build error message for error analyzer
    error_message = ""
    if stderr:
        error_message = f"stderr: {stderr}\n\n"
    if error_summary:
        error_message += f"Error summary: {error_summary}"

    # Track best score
    best_score = state.get("best_score")
    best_code = state.get("best_code", "")
    if validation_score is not None:
        if best_score is None or validation_score > best_score:
            best_score = validation_score
            best_code = state.get("python_code", "")
            logger.info(f"  ★ New best score: {best_score}")

    logger.info(f"  Decision: {decision}")
    logger.info(f"  Error summary: {error_summary or 'None'}")
    logger.info(f"  Validation score: {validation_score}")
    logger.info(f"  Best score so far: {best_score}")

    log_state_snapshot(state, f"after_execution_iter_{iteration}", state.get("output_folder", "./output"))

    return {
        "stdout": stdout,
        "stderr": stderr,
        "decision": decision,
        "error_summary": error_summary,
        "validation_score": validation_score,
        "error_message": error_message,
        "best_score": best_score,
        "best_code": best_code,
        "iteration": iteration + 1,  # increment after evaluation
    }


# ─── Node: analyze_error ────────────────────────────────────────────────

def analyze_error(state: MLAutoState) -> dict:
    """
    Analyze the execution error and produce debugging suggestions.

    Maps to: ErrorAnalyzerAgent.__call__()

    Returns:
        {
            "error_analysis": str,
            "all_error_analyses": list[str],
            "previous_python_code": str,
            "previous_bash_script": str,
        }
    """
    iteration = state.get("iteration", 0)
    logger.info(f"─── analyze_error [iteration {iteration}] ───")

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

    content = call_logger.call(llm, prompt, node_name=f"analyze_error[iter={iteration}]")

    # Parse error analysis
    analysis_match = re.search(r"ERROR_SUMMARY:\s*(.*)", content, re.DOTALL)
    if analysis_match:
        error_analysis = f"ERROR_SUMMARY: {analysis_match.group(1).strip()}"
    else:
        error_analysis = "Failed to extract error analysis from LLM response."

    # Accumulate error analyses
    all_analyses = list(state.get("all_error_analyses", []))
    all_analyses.append(error_analysis)

    logger.info(f"  Error analysis: {error_analysis[:300]}")

    return {
        "error_analysis": error_analysis,
        "all_error_analyses": all_analyses,
        "previous_python_code": state.get("python_code", ""),
        "previous_bash_script": state.get("bash_script", ""),
    }
