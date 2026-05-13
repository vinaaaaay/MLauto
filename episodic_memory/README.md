# episodic_memory

**Maps to:** `RerankerAgent` in autogluon-assistant.

Takes the tutorial candidates from `semantic_memory` and uses an **LLM to select the most relevant ones**, then formats them into a `tutorial_prompt` string ready for the coder.

## How it works

1. Receives `tutorial_retrieval` (list of `TutorialInfo`) from semantic memory.
2. Sends their titles + summaries to an LLM with the task context.
3. LLM returns comma-separated indices of the best tutorials.
4. Selected tutorials are read from disk, truncated to `max_tutorial_length`, and concatenated.
5. The final `tutorial_prompt` string is injected into the coder's prompt.

Falls back to score-ranked selection if the LLM response cannot be parsed.

## Files

```
episodic_memory/
├── graph.py    # Standalone LangGraph subgraph (single node)
├── nodes.py    # rerank_tutorials() node function
└── prompts.py  # RERANKER_PROMPT for tutorial selection
```

## State I/O

- **Input:** `tutorial_retrieval` (from semantic memory), `task_description`, `user_input`
- **Output:** `tutorial_prompt` (formatted string injected into the coder prompt)
