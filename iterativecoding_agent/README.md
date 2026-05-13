# iterativecoding_agent

**Maps to:** `CodingAgent` + `NodeManager` in autogluon-assistant.

MCTS-based LangGraph loop that generates, executes, and improves ML code until a satisfactory solution is found or the iteration budget is exhausted.

## MCTS Loop

```
select_node → expand_node → retrieve_node_tutorials → rerank_node_tutorials
    → generate_python_code → generate_bash_script → execute_and_evaluate
    → (analyze_error) → backpropagate → [repeat or done]
```

| Node | Role |
|---|---|
| `select_node` | UCT-based selection: picks the most promising node to work on |
| `expand_node` | Creates a child node — `evolve` (improve) or `debug` (fix error) |
| `retrieve_node_tutorials` | Delegates to `semantic_memory` for per-node tutorial retrieval |
| `rerank_node_tutorials` | Delegates to `episodic_memory` for per-node tutorial reranking |
| `generate_python_code` | LLM writes the Python ML training script |
| `generate_bash_script` | LLM writes the bash environment + execution script |
| `execute_and_evaluate` | Runs code in Docker, extracts validation score |
| `analyze_error` | LLM diagnoses the error if execution failed |
| `backpropagate` | Updates UCT statistics (visits, reward) up the tree |

## Files

```
iterativecoding_agent/
├── graph.py    # LangGraph StateGraph with conditional MCTS edges
├── nodes.py    # All node function implementations
└── prompts.py  # PYTHON_CODER_PROMPT, BASH_CODER_PROMPT, ERROR_ANALYZER_PROMPT, etc.
```

## State I/O

- **Input:** Everything from perception + memory phases
- **Output:** `best_code`, `best_score`, `submission_file`, `is_complete`
