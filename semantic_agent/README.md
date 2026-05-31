# Semantic Agent (A2A Protocol Compliance)

The **Semantic Agent** is a specialized Agent-to-Agent (A2A) microservice that automates semantic search and contextual machine learning tutorial retrieval for your multi-agent network. 

---

## What this Agent Does
When your other agents, workflows, or orchestrators hit coding challenges, library import issues, or execution tracebacks, they call the Semantic Agent. Given a task description, selected tool, and error trace, it automatically:
1. **Generates Optimized Search Queries**: Employs an LLM to analyze the task context and execution error tracebacks, formulating an optimized semantic search query.
2. **Retrieves Relevant Tutorials**: Connects to the **Vector Store MCP Server** to perform standard vector similarity search (using FAISS + BGE embeddings) against a centralized tools registry of markdown tutorials.
3. **Reranks and Filters Candidates**: Dynamically evaluates the relevance of the retrieved tutorial titles and summaries using an LLM to select the most matching candidates.
4. **Assembles Context-Rich Prompts**: Extracts and structures the complete markdown content of the selected tutorials into a context prompt that calling agents can immediately use to write self-correcting code or solve integration errors.

---

## File Layout & Purposes

*   **[`mcp_server.py`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/mcp_server.py)**: Operates as an independent FastMCP/FastAPI Vector Store server. Ingests local or S3 tools registry tutorials, builds an in-memory FAISS similarity database using `BAAI/bge-base-en-v1.5` embeddings, and exposes the `retrieve_tutorials` tool over standard POST/SSE routes.
*   **[`agent.py`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/agent.py)**: Compiles the core **LangGraph StateGraph** pipeline. Integrates the orchestrating nodes (`generate_query` -> `retrieve_tutorials` -> `rerank_tutorials`) into a single state machine.
*   **[`utils.py`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/utils.py)**: Houses shared data structures (`SemanticAgentState`, `TutorialInfo`), the direct `VectorStoreMCPClient` wrapper (with 120s timeout and smart SSE Lambda bypass to prevent hangs), and the LLM call logger.
*   **[`prompts.py`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/prompts.py)**: Stores structured instructions for query generation and LLM-based reranking nodes.
*   **[`a2a_server.py`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/a2a_server.py)**: Wraps the LangGraph pipeline inside the standard A2A JSON-RPC 2.0 interface. Declares the agent's cards and skills, and serves the application using Starlette.
*   **[`lambda_handler_a2a.py`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/lambda_handler_a2a.py)**: ASGI adapter using `Mangum` to invoke the Starlette A2A server on AWS Lambda.
*   **[`lambda_handler_mcp.py`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/lambda_handler_mcp.py)**: ASGI adapter using `Mangum` to invoke the FastAPI MCP server on AWS Lambda.
*   **[`Dockerfile`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/Dockerfile)**: Docker container configuration that pre-caches embedding weights and packages both handlers into a single shared image.
*   **[`requirements.txt`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/requirements.txt)** & **[`requirements-lambda.txt`](file:///home/administrator/dreamlab/FAME/agents/semantic_agent/requirements-lambda.txt)**: Active and reference package dependencies.

---

## AWS Lambda Deployment

Deploy both the MCP Server and the A2A Semantic Agent to AWS Lambda using ECR and the shared Docker image:

### 1. Build the local Docker image
```bash
docker build -t semantic-agent-a2a -f FAME/agents/semantic_agent/Dockerfile FAME
```

### 2. Deploy MCP Vector Store Lambda
```bash
python src/deploy/deploy_lambda.py --config src/deploy/config.mcp.yaml
```

### 3. Deploy A2A Semantic Agent Lambda
```bash
python src/deploy/deploy_lambda.py --config src/deploy/config.a2a.yaml
```
