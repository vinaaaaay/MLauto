# FAME Coder Agent

The **Coder Agent** is an agentic framework designed to generate, execute, and evaluate Machine Learning code inside a secure Docker sandbox. It is built on **LangGraph** (to handle execution flow), exposes services over the **A2A (Agent-to-Agent) JSON-RPC protocol**, and uses a **hybrid WebSocket/HTTP client** to manage sandbox file state and stream execution logs.

---

## 🏗️ Architecture Overview

The agent consists of three main components:
1. **A2A Server (`a2a_server.py`)**: A Starlette-based ASGI server that registers the agent's capabilities and skills, processes incoming tasks via JSON-RPC, and communicates progress.
2. **LangGraph Pipeline (`agent.py`)**: A state machine that sequentially executes:
   - **`generate_python_code`**: Utilizes an LLM to generate the Python script for the requested ML task.
   - **`generate_bash_script`**: Generates a shell script setting up dependencies (resolved from a tool registry) and triggering the Python script.
   - **`execute_and_evaluate`**: Runs the shell script inside the sandbox and parses results, determining success/failure.
3. **Sandbox Client (`tools/ws_sandbox.py`)**: Connects to the sandbox container. It reads and writes files via HTTP endpoints, and streams terminal execution output in real time over WebSockets.

---

## 📋 Prerequisites & Requirements

Before starting, ensure you have the following installed on your host system:
* **Docker** (to run the unconfined sandbox environment)
* **Python 3.11+**
* [**`uv`**](https://github.com/astral-sh/uv) (recommended package and project manager)
* An **OpenAI API Key** with access to reasoning models (e.g., `gpt-4o`, `gpt-5-nano`)

---

## 🚀 Step-by-Step Setup

### 1. Launch the Sandbox Container
The sandbox provides an isolated execution environment. You can spin it up using the `ghcr.io/agent-infra/sandbox` image.

> [!IMPORTANT]
> The default configuration in `a2a_server.py` and the test files expects the sandbox to be accessible on host port **`8081`** (mapping to container port `8080`).

You have two options to launch the container depending on your preferred port configuration:

#### Option A: Port `8081` (Recommended - No Code Changes)
To align with the default port settings in the codebase, run:
```bash
docker run --security-opt seccomp=unconfined --rm -it -p 8081:8080 ghcr.io/agent-infra/sandbox:latest
```

#### Option B: Port `8080` (Default Sandbox Port)
If you want to use the default container-to-host port mapping:
```bash
docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
```
*Note: If you choose Option B, ensure you update the sandbox URLs in `a2a_server.py` (change `8081` to `8080`) and your test configurations.*

---

### 2. Configure Environment Variables
Create a `.env` file inside the `agents/coder_agent` directory:

```bash
# Path: agents/coder_agent/.env
OPENAI_API_KEY="your-openai-api-key"
```

---

### 3. Install Host Dependencies
From the root of the `FAME` repository, sync the project dependencies (which will create/update the `.venv` virtual environment):

```bash
uv sync
```

### 4. Build the Lambda Docker Image
To containerize the Coder Agent for AWS Lambda deployment, build the Docker image from the **root of the `FAME` repository** so the build context correctly accesses both `agents/` and `common_local/`:

```bash
docker build -t coder-agent-lambda -f agents/coder_agent/Dockerfile .
```

*Note: The Dockerfile relies on copying the local utility modules from `agents/coder_agent/common_local/` into the image.*

---

## 🏃 Running the Agent

### 1. Start the A2A Server
Run the A2A server from the `FAME` project root. This starts the Starlette web server on `http://127.0.0.1:8089`.

```bash
uv run python -m agents.coder_agent.a2a_server
```

### 2. Run the Tests & Sanity Checks
Ensure the sandbox and WebSocket communication is operating correctly.

* **Sandbox API Sanity Test**:
  Verifies basic file reads/writes and command execution:
  ```bash
  uv run python -m unittest agents.coder_agent.test.test_sandbox_sanity
  ```

* **End-to-End (E2E) Integration Test**:
  Runs a full pipeline execution. It copies sample CSV data to the sandbox, registers a task with the local A2A server, streams the compilation/training steps, fetches the final model predictions (`results.csv`), and cleans up the sandbox:
  ```bash
  uv run python agents/coder_agent/test/test_coder_a2a_e2e.py
  ```

---

## 📊 Metrics & Logger Integration

The agent features unified metrics and logging instrumentation using the **`common_local`** library. It tracks execution characteristics through several key events:
- **`llm_call`**: Captures LLM invocation latency, input/output tokens (with reasoning and cached details), byte estimates, and provider-side processing duration (e.g., `openai-processing-ms`).
- **`tool_call`**: Logs individual sandbox commands and writes (`sandbox_exec_shell` and `sandbox_write_file`), capturing request parameters, status (success/error), and execution latency.
- **`psutil_metrics_node`**: Approximates peak memory (RAM) usage and execution time per individual LangGraph node.
- **`psutil_metrics_graph`**: Records the end-to-end memory usage, steps, and execution duration of the overall graph.

### Aggregating Metrics Logs
Logs are appended to `metrics.jsonl` in the `FAME` project root during execution. You can group and split these logs by event type using the `aggregate_logs.py` utility:

```bash
uv run python agents/coder_agent/aggregate_logs.py metrics.jsonl -o agents/coder_agent/test/logs/your_run_timestamp/
```

This aggregates and exports the logs into:
- `metrics.json`: Raw JSON events.
- `debug.json`: LLM invocation debug details.
- `llm_call.json`: Structured LLM metrics.
- `tool_call.json`: Sandbox tool call latencies.
- `psutil.json`: Node-level and graph-level performance statistics.

---

## 📁 File Structure

```text
agents/coder_agent/
├── .env                       # Local environment configurations (API keys)
├── README.md                  # This file
├── Dockerfile                 # Multi-stage image build for AWS Lambda deployment
├── requirements-lambda.txt    # Production dependencies for AWS Lambda environment
├── a2a_server.py              # A2A JSON-RPC interface exposing the agent
├── agent.py                   # StateGraph/LangGraph implementation
├── prompts.py                 # System and LLM execution prompts
├── utils.py                   # State variables, LLM logger, code parser
├── aggregate_logs.py          # Log parser and aggregator utility
├── common_local/              # Local copy of the unified logging utilities
├── tools/
│   ├── __init__.py
│   ├── sandbox_client.py      # Abstract base client interface
│   ├── ws_sandbox.py          # WebSocket/HTTP client for container communication
│   ├── bastion_sandbox.py     # AWS Bastion proxy client
│   └── mcp_client.py          # Legacy MCP synchronous runner
└── test/
    ├── test_coder_a2a_e2e.py  # Main E2E integration test
    ├── test_ws_sandbox.py     # WebSocket execution sanity tests
    └── test_sandbox_sanity.py # Standard file/command execution tests
```
