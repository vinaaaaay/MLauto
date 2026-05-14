"""
Utility functions ported from autogluon-assistant.

Sources:
  - get_all_files, group_similar_files: agents/data_perception_agent.py
  - extract_code: prompts/utils.py
  - execute_code: agents/executer_agent.py
"""

import logging
import os
import re
import select
import subprocess
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ─── File scanning (from DataPerceptionAgent) ──────────────────────────────

def get_all_files(folder_path: str) -> list[tuple[str, str]]:
    """
    Recursively get all files in folder_path.

    Returns:
        List of (relative_path, absolute_path) tuples.
    """
    all_files = []
    abs_folder_path = os.path.abspath(folder_path)

    for root, _, files in os.walk(abs_folder_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, abs_folder_path)
            all_files.append((rel_path, abs_path))

    return all_files


def group_similar_files(files: list[tuple[str, str]]) -> dict:
    """
    Group files by folder structure and extension.

    At each depth level, if there are ≤5 unique folders the actual names are
    used; otherwise a wildcard '*' is substituted.

    Returns:
        Dict mapping group-key tuples to lists of (rel_path, abs_path).
    """
    depth_folders = defaultdict(set)

    for rel_path, _ in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        for depth, folder in enumerate(parts[:-1]):
            depth_folders[depth].add(folder)

    groups = defaultdict(list)
    for rel_path, abs_path in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        folders = parts[:-1]
        filename = parts[-1]
        ext = os.path.splitext(filename)[1].lower()

        group_key_parts = []
        for depth, folder in enumerate(folders):
            if len(depth_folders[depth]) <= 5:
                group_key_parts.append(folder)
            else:
                group_key_parts.append("*")
        group_key_parts.append(ext if ext else "NO_EXT")

        groups[tuple(group_key_parts)].append((rel_path, abs_path))

    return groups


def pattern_to_path(pattern: tuple, base_path: str) -> str:
    """Convert a group pattern tuple to a display path string."""
    folders = pattern[:-1]
    ext = pattern[-1]

    path_parts = list(str(f) for f in folders)
    path_parts.append("*" if ext == "NO_EXT" else f"*{ext}")

    relative_pattern = os.path.join(*path_parts) if path_parts else "*"
    return os.path.join(base_path, relative_pattern)


# ─── Code extraction (from prompts/utils.py) ──────────────────────────────

def extract_code(response: str, language: str) -> str:
    """
    Extract a fenced code block from an LLM response.

    Tries ```python or ```bash first, then generic ```, then full response.
    """
    if language == "python":
        pattern = r"```python\s*\n(.*?)```"
    elif language == "bash":
        pattern = r"```bash\s*\n(.*?)```"
    else:
        raise ValueError(f"Unsupported language: {language}")

    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fallback: generic code block
    generic = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if generic:
        logger.warning(f"No {language} block found; using generic code block.")
        return generic[0].strip()

    logger.warning(f"No code block found; returning full response.")
    return response


# ─── Code execution (from agents/executer_agent.py) ───────────────────────

def execute_code(code: str, language: str, timeout: int = 3600) -> tuple[bool, str, str]:
    """
    Execute code with real-time output streaming and timeout.

    Args:
        code: Code string to execute.
        language: "python" or "bash".
        timeout: Maximum seconds before killing the process.

    Returns:
        (success, stdout, stderr)
    """
    if language.lower() == "python":
        cmd = ["python", "-c", code]
    elif language.lower() == "bash":
        cmd = ["bash", "-c", code]
    else:
        return False, "", f"Unsupported language: {language}"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks, stderr_chunks = [], []
        recent_stdout = deque(maxlen=100)
        recent_stderr = deque(maxlen=100)
        streams = [process.stdout, process.stderr]
        start_time = time.time()

        while streams:
            elapsed = time.time() - start_time
            remaining = max(0, timeout - elapsed)

            if remaining <= 0:
                process.terminate()
                time.sleep(3)
                if process.poll() is None:
                    process.kill()
                stdout_chunks.append(f"\nProcess reached time limit after {timeout} seconds.\n")
                logger.info(f"Process reached time limit after {timeout}s.")
                break

            readable, _, _ = select.select(streams, [], [], min(1, remaining))

            if not readable and process.poll() is None:
                continue
            if not readable and process.poll() is not None:
                break

            for stream in readable:
                line = stream.readline()
                if not line:
                    streams.remove(stream)
                    continue

                if stream == process.stdout:
                    if line not in recent_stdout:
                        recent_stdout.append(line)
                        stdout_chunks.append(line)
                else:
                    if line not in recent_stderr:
                        recent_stderr.append(line)
                        stderr_chunks.append(line)

        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_chunks.append("Process forcibly terminated after timeout\n")

        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)

    except Exception as e:
        return False, "", f"Error executing {language} code: {str(e)}"


# ─── Docker execution (containerized environment) ─────────────────────────

def execute_in_docker(
    bash_script_path: str,
    input_data_folder: str,
    output_folder: str,
    docker_image: str = "mlauto-executor:latest",
    timeout: int = 86400,
) -> tuple[bool, str, str]:
    """
    Execute a bash script inside a Docker container.

    The container gets:
      - /workspace/data  ← input_data_folder (read-only bind mount)
      - /workspace/output ← output_folder (read-write bind mount)
      - /workspace/script.sh ← the bash script to execute

    This mirrors autogluon-assistant's containerized execution model.

    Args:
        bash_script_path: Absolute path to the bash script on the host.
        input_data_folder: Absolute path to the input data directory.
        output_folder: Absolute path to the output directory.
        docker_image: Docker image to use for execution.
        timeout: Maximum seconds before killing the container.

    Returns:
        (success, stdout, stderr)
    """
    container_name = f"mlauto-exec-{os.getpid()}-{int(time.time())}"

    # Build the docker run command using Named Volumes for high-performance, permission-safe caching
    docker_cmd = [
        "docker", "run",
        "--name", container_name,
        "--rm",                                             # auto-remove on exit
        "--gpus", "all",                                    # enable GPU access
        "--ipc=host",                                       # avoid shared memory limits
        "-v", f"{input_data_folder}:/workspace/data:ro",    # data is read-only
        "-v", f"{output_folder}:/workspace/output",         # output is read-write
        "-v", f"{bash_script_path}:/workspace/script.sh:ro",
        "-v", "mlauto_pip_cache:/root/.cache/pip",          # high-perf named volume for pip
        "-v", "mlauto_uv_cache:/root/.cache/uv",            # high-perf named volume for uv
        "-w", "/workspace",
        docker_image,
        "bash", "/workspace/script.sh",
    ]

    logger.info(f"Docker exec: {' '.join(docker_cmd[:8])}...")

    try:
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks, stderr_chunks = [], []
        streams = [process.stdout, process.stderr]
        start_time = time.time()

        while streams:
            elapsed = time.time() - start_time
            remaining = max(0, timeout - elapsed)

            if remaining <= 0:
                # Kill the container on timeout
                logger.warning(f"Docker container {container_name} hit timeout ({timeout}s). Killing.")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                stdout_chunks.append(f"\nDocker container killed after {timeout}s timeout.\n")
                break

            readable, _, _ = select.select(streams, [], [], min(1, remaining))

            if not readable and process.poll() is None:
                continue
            if not readable and process.poll() is not None:
                break

            for stream in readable:
                line = stream.readline()
                if not line:
                    streams.remove(stream)
                    continue
                
                # Real-time console logging
                if stream == process.stdout:
                    print(f"  [DOCKER STDOUT] {line}", end="", flush=True)
                    stdout_chunks.append(line)
                else:
                    print(f"  [DOCKER STDERR] {line}", end="", flush=True)
                    stderr_chunks.append(line)

        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                stderr_chunks.append("Docker container forcibly killed after timeout\n")

        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)

    except FileNotFoundError:
        return False, "", (
            "Docker is not installed or not in PATH. "
            "Install Docker to enable containerized execution."
        )
    except Exception as e:
        return False, "", f"Error running Docker container: {str(e)}"
