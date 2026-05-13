# perception_agent

**Maps to:** `DataPerceptionAgent`, `DescriptionFileRetrieverAgent`, `TaskDescriptorAgent`, `ToolSelectorAgent` in autogluon-assistant.

Linear LangGraph chain that understands the input data and selects the right ML tool.

## Nodes (in order)

| Node | What it does |
|---|---|
| `scan_data` | Reads all files in the data folder; uses LLM to summarize each file's content |
| `find_description_files` | Identifies README/description files from the scan |
| `generate_task_description` | Synthesizes a concise task description from file contents + user input |
| `select_tools` | Ranks available ML libraries for the task (e.g. `autogluon.tabular`) |

After `select_tools`, the chain hands off to:
- `semantic_memory/` → retrieves relevant tutorials
- `episodic_memory/` → reranks and formats them into a prompt

## Files

```
perception_agent/
├── graph.py    # LangGraph StateGraph (linear chain)
├── nodes.py    # Node function implementations
└── prompts.py  # Prompt templates for each node
```
