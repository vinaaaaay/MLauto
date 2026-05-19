"""
Semantic Memory MCP Server — STANDALONE.

Fully self-contained MCP server deployable as a Lambda / Cloud Run function.
Exposes tutorial retrieval via FAISS + BGE semantic search over HTTPS (SSE).

Zero imports from the `shared/` package or `nodes.py`.
All dependencies (LLM init, TutorialIndexer, prompts) are inlined.
"""

import contextlib
import io
import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from FlagEmbedding import FlagModel
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("semantic_memory_mcp")
logging.basicConfig(level=logging.INFO)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# ═══════════════════════════════════════════════════════════════════════════
#  Paths & Sync
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_REGISTRY = Path(__file__).parent.parent / "tools_registry"
TMP_REGISTRY = Path("/tmp/tools_registry")


def sync_registry():
    """Copies tools_registry from source to ephemeral /tmp directory.

    Simulates downloading from S3/GCS during a Lambda cold-start.
    """
    logger.info(f"Syncing tools_registry from {SOURCE_REGISTRY} to {TMP_REGISTRY}")
    try:
        if TMP_REGISTRY.exists():
            shutil.rmtree(TMP_REGISTRY)
        shutil.copytree(SOURCE_REGISTRY, TMP_REGISTRY)
        logger.info("tools_registry sync completed successfully.")
    except Exception as e:
        logger.error(f"Failed to sync tools_registry to /tmp: {e}")


# Run sync immediately on startup / cold-start
sync_registry()

# ═══════════════════════════════════════════════════════════════════════════
#  Inlined: LLM Initialization  (from shared/llm.py)
# ═══════════════════════════════════════════════════════════════════════════


def _get_llm(config: dict = None) -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    config = config or {}
    model = config.get("model", "gpt-4o")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 16384)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )

    is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])
    if is_reasoning_model:
        return ChatOpenAI(
            model=model,
            temperature=1,
            max_completion_tokens=max_tokens,
            api_key=api_key,
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Inlined: LLM Call Logger  (from shared/logging_config.py)
# ═══════════════════════════════════════════════════════════════════════════


class _LLMCallLogger:
    """Logs every LLM call (prompt + response) to structured JSONL."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.call_count = 0

    def call(self, llm, prompt: str, node_name: str = "unknown") -> str:
        self.call_count += 1
        call_id = self.call_count

        logger.info(f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)")

        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        content = response.content

        logger.info(
            f"[Call #{call_id}] {node_name} — received response "
            f"({len(content)} chars, {elapsed:.1f}s)"
        )

        record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "prompt_length": len(prompt),
            "response_length": len(content),
            "elapsed_seconds": round(elapsed, 2),
            "prompt": prompt,
            "response": content,
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM call log: {e}")

        return content


# ═══════════════════════════════════════════════════════════════════════════
#  Inlined: Tutorial Indexer  (from shared/tutorial_indexer.py)
# ═══════════════════════════════════════════════════════════════════════════


class _TutorialIndexer:
    """FAISS + BGE embedding indexer for tutorial semantic search."""

    def __init__(self, registry_path: Path = None, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.registry_path = registry_path or TMP_REGISTRY
        self.embedding_model_name = embedding_model_name
        self.sanitized_model_name = embedding_model_name.replace("/", "_")
        self.model = None
        self.indices: Dict[str, Dict[str, faiss.Index]] = {}
        self.metadata: Dict[str, Dict[str, List[Dict]]] = {}
        self.index_dir = self.registry_path / "indices" / self.sanitized_model_name
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        self.cleanup()

    def _silent_encode(self, input_texts):
        with contextlib.redirect_stderr(io.StringIO()):
            return self.model.encode(input_texts)

    def cleanup(self):
        if self.model is not None:
            try:
                if hasattr(self.model, "close"):
                    self.model.close()
                elif hasattr(self.model, "stop_multi_process_pool"):
                    self.model.stop_multi_process_pool()
                else:
                    del self.model
            except Exception as e:
                logger.debug(f"Error during model cleanup: {e}")
            finally:
                self.model = None

    def _load_embedding_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = FlagModel(
                self.embedding_model_name,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )
            logger.info("Embedding model loaded successfully")

    def _get_tutorials_folder(self, tool_name: str, condensed: bool = False) -> Path:
        subfolder = "condensed_tutorials" if condensed else "tutorials"
        tool_dir = self.registry_path / tool_name / subfolder
        if tool_dir.exists():
            return tool_dir
        raise FileNotFoundError(f"No {subfolder} found for tool {tool_name} at {tool_dir}")

    def _extract_summary_from_md(self, md_path: Path) -> Optional[str]:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.split("\n")
            for line in lines:
                if line.strip().startswith("Summary: "):
                    return line.strip()[9:]
            return None
        except Exception as e:
            logger.warning(f"Error extracting summary from {md_path}: {e}")
            return None

    def _build_tool_index(self, tool_name: str, tutorial_type: str) -> Tuple[Optional[faiss.Index], List[Dict]]:
        self._load_embedding_model()

        try:
            condensed = tutorial_type == "condensed_tutorials"
            tutorials_folder = self._get_tutorials_folder(tool_name, condensed=condensed)
        except FileNotFoundError as e:
            logger.warning(f"No {tutorial_type} found for tool {tool_name}: {e}")
            return None, []

        summaries = []
        metadata = []

        for md_file in tutorials_folder.rglob("*.md"):
            summary = self._extract_summary_from_md(md_file)
            if summary and summary.strip():
                summaries.append(summary)
                metadata.append({
                    "tool_name": tool_name,
                    "tutorial_type": tutorial_type,
                    "file_path": str(md_file),
                    "relative_path": str(md_file.relative_to(tutorials_folder)),
                    "summary": summary,
                })

        if not summaries:
            logger.warning(f"No summaries found for {tool_name} {tutorial_type}")
            return None, []

        logger.info(f"Found {len(summaries)} summaries for {tool_name} {tutorial_type}")

        try:
            batch_size = 16
            all_embeddings = []

            for i in range(0, len(summaries), batch_size):
                batch = summaries[i:i + batch_size]
                batch_emb = self._silent_encode(batch)
                if not isinstance(batch_emb, np.ndarray):
                    batch_emb = np.array(batch_emb)
                all_embeddings.append(batch_emb)

            embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
            embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

            if len(embeddings.shape) != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
            if not np.isfinite(embeddings).all():
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)

            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            index.add(embeddings_normalized)

            logger.info(f"Built index for {tool_name} {tutorial_type} with {index.ntotal} vectors")
            return index, metadata

        except Exception as e:
            logger.error(f"Error building index for {tool_name} {tutorial_type}: {e}")
            return None, []

    def build_indices(self, tools: Optional[List[str]] = None) -> None:
        if tools is None:
            tools = []
            for item in self.registry_path.iterdir():
                if item.is_dir() and not item.name.startswith("_") and item.name not in ("indices", "FlagEmbedding", "__pycache__"):
                    tools.append(item.name)

        logger.info(f"Building indices for tools: {tools}")

        for tool_name in tools:
            if tool_name not in self.indices:
                self.indices[tool_name] = {}
                self.metadata[tool_name] = {}

            for tutorial_type in ["tutorials", "condensed_tutorials"]:
                try:
                    index, metadata = self._build_tool_index(tool_name, tutorial_type)
                    if index is not None:
                        self.indices[tool_name][tutorial_type] = index
                        self.metadata[tool_name][tutorial_type] = metadata
                except Exception as e:
                    logger.error(f"Error building {tutorial_type} index for {tool_name}: {e}")

    def save_indices(self) -> None:
        for tool_name, tool_indices in self.indices.items():
            tool_dir = self.index_dir / tool_name
            tool_dir.mkdir(exist_ok=True)

            for tutorial_type, index in tool_indices.items():
                faiss.write_index(index, str(tool_dir / f"{tutorial_type}.index"))
                with open(tool_dir / f"{tutorial_type}.metadata", "wb") as f:
                    pickle.dump(self.metadata[tool_name][tutorial_type], f)

    def load_indices(self) -> bool:
        self.indices = {}
        self.metadata = {}

        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            logger.warning(f"No index exists at {self.index_dir}")
            return False

        for tool_dir in self.index_dir.iterdir():
            if tool_dir.is_dir():
                tool_name = tool_dir.name
                self.indices[tool_name] = {}
                self.metadata[tool_name] = {}

                for tutorial_type in ["tutorials", "condensed_tutorials"]:
                    index_file = tool_dir / f"{tutorial_type}.index"
                    metadata_file = tool_dir / f"{tutorial_type}.metadata"

                    if index_file.exists() and metadata_file.exists():
                        try:
                            index = faiss.read_index(str(index_file))
                            self.indices[tool_name][tutorial_type] = index
                            with open(metadata_file, "rb") as f:
                                self.metadata[tool_name][tutorial_type] = pickle.load(f)
                            logger.info(f"Loaded {tool_name} {tutorial_type} index ({index.ntotal} vectors)")
                        except Exception as e:
                            logger.error(f"Error loading {tool_name} {tutorial_type} index: {e}")
                            return False
                    else:
                        logger.warning(f"Missing index or metadata for {tool_name} {tutorial_type}")
                        return False
        return True

    def search(self, query: str, tool_name: str, condensed: bool = False, top_k: int = 5) -> List[Dict]:
        self._load_embedding_model()

        tutorial_type = "condensed_tutorials" if condensed else "tutorials"

        if tool_name not in self.indices or tutorial_type not in self.indices[tool_name]:
            logger.warning(f"No index found for {tool_name} {tutorial_type}")
            return []

        index = self.indices[tool_name][tutorial_type]
        metadata = self.metadata[tool_name][tutorial_type]

        if index.ntotal == 0:
            return []

        query_embedding = self._silent_encode([query])
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        faiss.normalize_L2(query_embedding)

        scores, indices_arr = index.search(query_embedding, min(top_k, index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices_arr[0]):
            if idx == -1:
                break
            meta = metadata[idx]
            try:
                with open(meta["file_path"], "r", encoding="utf-8") as f:
                    content = f.read()
                results.append({
                    "tool_name": meta["tool_name"],
                    "tutorial_type": meta["tutorial_type"],
                    "file_path": meta["file_path"],
                    "relative_path": meta["relative_path"],
                    "summary": meta["summary"],
                    "content": content,
                    "score": float(score),
                })
            except Exception as e:
                logger.error(f"Error loading content from {meta['file_path']}: {e}")

        logger.info(f"Found {len(results)} results for query: {query[:80]}...")
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ═══════════════════════════════════════════════════════════════════════════
#  Inlined: Prompt Template  (from prompts.py)
# ═══════════════════════════════════════════════════════════════════════════

_RETRIEVER_PROMPT = """\
You are an expert at generating search queries to find relevant machine learning tutorials. Given the context below, generate a concise and effective search query that will help find the most relevant tutorials for this task.

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instruction
{user_input}

### Previous Error Analysis
{all_previous_error_analyses}

### Selected Tool/Library
{selected_tool}


Based on the above context, generate a search query that will help find tutorials most relevant to this task. The query should:
1. Include key technical terms and concepts
2. Focus on the main task/problem to solve
3. Be concise but specific

IMPORTANT: Respond ONLY with the search query text. Do not include explanations, quotes, or any other formatting.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  MCP Server Definition
# ═══════════════════════════════════════════════════════════════════════════

mcp = FastMCP("Semantic Memory Server")


@mcp.tool()
def retrieve_tutorials(
    task_description: str,
    data_prompt: str = "",
    user_input: str = "",
    current_tool: str = "",
    all_error_analyses: list[str] = None,
    config: dict = None,
    output_folder: str = "./output",
) -> list[dict]:
    """
    Retrieve semantically relevant tutorials using FAISS vector search and BGE embeddings.

    Generates a search query via LLM, performs FAISS semantic search over the
    tool tutorials synced to /tmp, and returns matching tutorials as dicts.
    """
    config = config or {}
    all_error_analyses = all_error_analyses or []
    tutorials_config = config.get("tutorials", {})

    logger.info("─── [Semantic Memory Server] retrieve_tutorials ───")

    # ── LLM setup ──
    llm = _get_llm(config.get("llm"))
    call_logger = _LLMCallLogger(output_folder)

    current_tool = current_tool or ""
    top_k = tutorials_config.get("num_tutorial_retrievals", 10)
    condense = tutorials_config.get("condense_tutorials", True)

    # ── Initialize tutorial indexer (pointing to /tmp) ──
    indexer = _TutorialIndexer(registry_path=TMP_REGISTRY)

    try:
        loaded = indexer.load_indices()
        if not loaded:
            logger.info("  Building tutorial indices for the first time...")
            indexer.build_indices()
            indexer.save_indices()
            logger.info("  Tutorial indices built and saved.")
    except Exception as e:
        logger.error(f"  Error initializing tutorial indexer: {e}")
        return []

    # ── Generate search query via LLM ──
    all_errors = "\n\n".join(all_error_analyses) or "None"

    prompt = _RETRIEVER_PROMPT.format(
        task_description=task_description,
        data_prompt=data_prompt,
        user_input=user_input,
        all_previous_error_analyses=all_errors,
        selected_tool=current_tool,
    )

    response = call_logger.call(llm, prompt, node_name="semantic_memory/retrieve_tutorials")

    # ── Parse search query from LLM response ──
    search_query = response.strip().split("\n")[0].strip().strip("\"'")

    for prefix in ["search query:", "query:", "the search query is:"]:
        if search_query.lower().startswith(prefix):
            search_query = search_query[len(prefix):].strip()
            break

    if not search_query:
        search_query = (task_description or current_tool)[:256]
        logger.warning("  Failed to generate search query; using task description fallback.")

    if len(search_query) > 512:
        search_query = search_query[:512]

    logger.info(f"  Search query: '{search_query}'")

    # ── Perform FAISS semantic search ──
    results = indexer.search(
        query=search_query,
        tool_name=current_tool,
        condensed=condense,
        top_k=top_k,
    )

    # ── Convert to JSON-serializable dicts ──
    serialized = []
    for result in results:
        file_path = result["file_path"]
        content = result["content"]
        score = result["score"]

        lines = content.split("\n")
        title = next(
            (line.lstrip("#").strip() for line in lines if line.strip().startswith("#")),
            os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title(),
        )
        summary = next(
            (line.replace("Summary:", "").strip() for line in lines if line.strip().startswith("Summary:")),
            "",
        )

        serialized.append({
            "path": file_path,
            "title": title,
            "summary": summary,
            "score": score,
            "content": content,
        })

    logger.info(f"  Retrieved {len(serialized)} tutorial candidates")
    indexer.cleanup()

    return serialized


@mcp.resource("semantic-memory://index-status")
def get_index_status() -> str:
    """Get the status of FAISS indices in the ephemeral storage."""
    if not TMP_REGISTRY.exists():
        return "Error: Ephemeral tools_registry (/tmp/tools_registry) is not populated."

    indices_dir = TMP_REGISTRY / "indices"
    if not indices_dir.exists():
        return f"tools_registry exists at {TMP_REGISTRY}, but no indices directory found."

    lines = [f"Ephemeral Registry Path: {TMP_REGISTRY}", "FAISS Indices status:"]
    for path in indices_dir.rglob("*.index"):
        rel_path = path.relative_to(indices_dir)
        size_bytes = path.stat().st_size
        lines.append(f" - {rel_path} ({size_bytes} bytes)")

    return "\n".join(lines)


@mcp.prompt("tutorial-search-query-generator")
def get_search_query_generator_prompt() -> str:
    """Get the prompt template used to generate the semantic search query."""
    return _RETRIEVER_PROMPT


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI App  (HTTPS/SSE mount)
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Semantic Memory MCP Server")
app.mount("/", mcp.sse_app())
