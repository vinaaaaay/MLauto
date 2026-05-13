# semantic_memory

**Maps to:** `RetrieverAgent` in autogluon-assistant.

Retrieves the most semantically relevant tutorials for the current task using **FAISS vector search** and **BGE embeddings**.

## How it works

1. An LLM generates a focused search query from the task description + prior errors.
2. The query is embedded using `BAAI/bge-large-en-v1.5` (via `FlagEmbedding`).
3. FAISS searches a pre-built index of all tutorial documents in `tools_registry/`.
4. The top-`k` results are returned as `TutorialInfo` objects for reranking.

Indices are built once and cached in `tools_registry/indices/`.

## Files

```
semantic_memory/
├── graph.py    # Standalone LangGraph subgraph (single node)
├── nodes.py    # retrieve_tutorials() node function
└── prompts.py  # RETRIEVER_PROMPT for search query generation
```

## State I/O

- **Input:** `task_description`, `user_input`, `current_tool`, `all_error_analyses`
- **Output:** `tutorial_retrieval` (list of `TutorialInfo`)
