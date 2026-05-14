# MLauto

A faithful replication of the [AutoGluon-Assistant](https://github.com/autogluon/autogluon-assistant) architecture, implemented in [LangGraph](https://langchain-ai.github.io/langgraph/).

MLauto automates solving ML tasks end-to-end: it understands the data, selects the right library, retrieves relevant tutorials, generates and executes code inside Docker, and uses **MCTS** to intelligently search the solution space — backtracking out of dead ends.

---

## Architecture

```
Phase 1 — Perception
  scan_data → find_description_files → generate_task_description → select_tools
      → [Semantic Memory] retrieve_tutorials → [Episodic Memory] rerank_tutorials

Phase 2 — Iterative Coding (MCTS loop)
  select_node → expand_node → retrieve_node_tutorials → rerank_node_tutorials
      → generate_python_code → generate_bash_script → execute_and_evaluate
      → backpropagate → (repeat or done)
```

## Modules

| Module | Maps to | Role |
|---|---|---|
| `perception_agent/` | `DataPerceptionAgent` etc. | Understand data & select tools |
| `semantic_memory/` | `RetrieverAgent` | FAISS + BGE tutorial search |
| `episodic_memory/` | `RerankerAgent` | LLM-based tutorial selection |
| `iterativecoding_agent/` | `CodingAgent` + `NodeManager` | MCTS code generation loop |
| `shared/` | Core infrastructure | State, LLM, Node, NodeManager, TutorialIndexer |

## Quick Start

### 1. Build the Docker Image
MLauto executes all generated code inside an isolated Docker container. You must build the base executor image first:
```bash
# Build the docker image (make sure the Docker daemon is running)
docker build -t mlauto-executor:latest .
```

### 2. Install Dependencies & Setup
```bash
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=sk-...
```

### 3. Run the Pipeline
Here is a full-fledged command to run the pipeline, with a breakdown of what each argument does:

```bash
python run.py /home/administrator/dreamlab/data \
    -u "Solve the denoising dirty documents task according to the description file." \
    -o ./my_results \
    -v 4 \
    -n 3
```

**Arguments Explained:**
*   `/path/to/your/dataset`: **(Required)** The absolute or relative path to your input data folder. This is a positional argument, so it requires no flag.
*   `-u` / `--user-input`: **(Required)** The specific instructions or task description for the ML agent.
*   `-v` / `--verbosity`: *(Optional)* Sets the terminal logging level from `0` to `4`. The default is `2` (INFO). We recommend `3` (DETAIL) for tracking the MCTS tree progress, and `4` (DEBUG) for viewing raw LLM prompts.
*   `-o` / `--output`: *(Optional)* The directory where generated code, logs, and state snapshots will be saved. If omitted, it auto-generates a unique folder in `./runs`.
*   `-n` / `--max-iterations`: *(Optional)* Overrides the maximum MCTS tree search iterations specified in `config.yaml`.
*   `-c` / `--config`: *(Optional)* Path to a custom YAML configuration file.

## Config

Edit `config.yaml` to control:
- LLM model and temperature
- MCTS parameters (iterations, exploration constant, failure penalty)
- Tutorial retrieval (top-k, condensed vs full, max length)
- Docker execution settings