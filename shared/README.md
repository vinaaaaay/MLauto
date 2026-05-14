# shared

Core infrastructure used by all modules.

## Components

| File | Purpose |
|---|---|
| `node.py` | `Node` dataclass — MCTS tree node with UCT stats (visits, reward, children) |
| `node_manager.py` | `NodeManager` — MCTS orchestrator: UCT selection, expansion, backpropagation |
| `tutorial_indexer.py` | `TutorialIndexer` — builds and queries FAISS indices over `tools_registry/` |
| `tool_registry.py` | `ToolRegistry` — loads `catalog.json`, resolves tool paths and tutorial folders |
| `llm.py` | `get_llm()` — returns a configured `ChatOpenAI` instance from config |
| `utils.py` | `get_all_files`, `group_similar_files`, `extract_code`, `execute_in_docker`, etc. |
| `logging_config.py` | `configure_logging`, `LLMCallLogger`, `log_state_snapshot` |
## Key Design: Agent-specific States

Each module has its own `state.py` containing a `TypedDict` for the specific state it requires. LangGraph automatically maps keys during execution.

```
input_data_folder, user_input, config          ← set at startup
data_prompt, task_description, current_tool   ← set by perception_agent
tutorial_retrieval, tutorial_prompt            ← set by memory modules
python_code, bash_script, validation_score     ← set by iterativecoding_agent
best_score, best_code, is_complete             ← final outputs
_node_manager                                  ← MCTS tree (NodeManager instance)
```
