import os
# Set default USER env var to avoid pwd.getpwuid lookup error in scratch-built Docker containers
os.environ["USER"] = os.environ.get("USER") or "administrator"

import logging

# Monkeypatch to resolve PyTorch/Transformers "precompile already registered" bug
try:
    import torch.compiler._cache as tc_cache
    def safe_register(cls, factory):
        if not hasattr(cls, "_factories"):
            cls._factories = {}
        if factory.type not in cls._factories:
            cls._factories[factory.type] = factory
        return factory
    tc_cache.CacheArtifactFactory.register = classmethod(safe_register)
    logging.getLogger("vector_store_mcp").info("Applied PyTorch CacheArtifactFactory double-registration monkeypatch.")
except Exception:
    pass

import os
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

# Setup logging
logger = logging.getLogger("vector_store_mcp")
logging.basicConfig(level=logging.INFO)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

def get_registry_path() -> Path:
    """Ensure tools registry is resolved dynamically and return path."""
    registry_env = os.environ.get("REGISTRY_PATH")
    if registry_env:
        return Path(registry_env)
        
    return Path(__file__).resolve().parent.parent / "tools_registry"

# ═══════════════════════════════════════════════════════════════════════════
#  Data Source & Document Abstractions (Modular Data Loading)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """Standardized document representation passed downstream to the vector indexer."""
    content: str
    metadata: Dict[str, Any]


class LocalTutorialParser:
    """Parses tutorial markdown files from local directory structure."""

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)

    def _extract_summary_from_md(self, md_path: Path) -> str:
        """Extract summary prefix from tutorial markdown file."""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            for line in content.split("\n"):
                if line.strip().startswith("Summary: "):
                    return line.strip()[9:]
            return ""
        except Exception as e:
            logger.warning(f"Error extracting summary from {md_path}: {e}")
            return ""

    def _extract_title(self, content: str, default: str) -> str:
        """Extract first header as title, or fallback to file stem."""
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                return line.strip().lstrip("#").strip()
        return default.replace("_", " ").title()

    def parse_tutorials(self, tool_name: str, condensed: bool = False) -> List[Document]:
        """Loads and parses tutorials for a specific tool."""
        subfolder = "condensed_tutorials" if condensed else "tutorials"
        tool_dir = self.registry_path / tool_name / subfolder
        if not tool_dir.exists():
            logger.warning(f"Tutorial folder {tool_dir} does not exist.")
            return []

        documents = []
        for md_file in tool_dir.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                summary = self._extract_summary_from_md(md_file)
                title = self._extract_title(content, md_file.stem)

                documents.append(Document(
                    content=content,
                    metadata={
                        "tool_name": tool_name,
                        "tutorial_type": subfolder,
                        "file_path": str(md_file),
                        "relative_path": str(md_file.relative_to(tool_dir)),
                        "summary": summary,
                        "title": title
                    }
                ))
            except Exception as e:
                logger.error(f"Failed parsing markdown file {md_file}: {e}")

        logger.info(f"Parsed {len(documents)} documents for {tool_name} ({subfolder})")
        return documents


# ═══════════════════════════════════════════════════════════════════════════
#  Vector Indexing & Search (FAISS + BGE)
# ═══════════════════════════════════════════════════════════════════════════

class _TutorialIndexer:
    """FAISS index manager decoupled from physical source layout."""

    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.embedding_model_name = embedding_model_name
        self.model: Optional["FlagModel"] = None
        self.index: Optional["faiss.Index"] = None
        self.documents: List[Document] = []

    def _silent_encode(self, input_texts: List[str]) -> np.ndarray:
        with contextlib.redirect_stderr(io.StringIO()):
            return self.model.encode(input_texts)

    def _load_model(self) -> None:
        """Load BGE model lazily to avoid cold start issues."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            
            # Apply double registration monkeypatch directly before importing FlagEmbedding
            try:
                import torch
                import torch.compiler._cache as tc_cache
                def safe_register(cls, factory):
                    if not hasattr(cls, "_factories"):
                        cls._factories = {}
                    if factory.type not in cls._factories:
                        cls._factories[factory.type] = factory
                    return factory
                tc_cache.CacheArtifactFactory.register = classmethod(safe_register)
                logger.info("Directly applied CacheArtifactFactory monkeypatch before FlagEmbedding import.")
            except Exception as e:
                logger.warning(f"Could not apply CacheArtifactFactory patch in _load_model: {e}")
                
            try:
                import transformers
                logger.info("Successfully pre-loaded transformers.")
            except Exception as e:
                logger.warning(f"Could not pre-load transformers in _load_model: {e}")
                
            from FlagEmbedding import FlagModel
            self.model = FlagModel(
                self.embedding_model_name,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )
            logger.info("Embedding model loaded successfully.")

    def build_index(self, documents: List[Document]) -> None:
        """Create FAISS Index Flat IP from a list of standardized Documents."""
        import faiss
        self._load_model()
        self.documents = documents
        
        if not documents:
            self.index = None
            return

        # Embed document summaries (fall back to first 1000 characters of content if summary is empty)
        texts_to_embed = [
            doc.metadata.get("summary") or doc.content[:1000]
            for doc in documents
        ]

        logger.info(f"Embedding {len(texts_to_embed)} text targets for FAISS indexing...")
        embeddings = self._silent_encode(texts_to_embed)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Standard L2 Normalization for Cosine Similarity (Inner Product)
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logger.info(f"Successfully initialized FAISS store with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the active index for the query vector."""
        if self.index is None or not self.documents:
            logger.warning("Search called on an uninitialized or empty FAISS index.")
            return []

        import faiss
        self._load_model()

        query_embedding = self._silent_encode([query])
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        faiss.normalize_L2(query_embedding)

        scores, indices_arr = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices_arr[0]):
            if idx == -1:
                break
            doc = self.documents[idx]
            results.append({
                "path": doc.metadata.get("file_path", ""),
                "title": doc.metadata.get("title", ""),
                "summary": doc.metadata.get("summary", ""),
                "score": float(score),
                "content": doc.content,
            })
        return results

    def cleanup(self) -> None:
        if self.model is not None:
            try:
                if hasattr(self.model, "close"):
                    self.model.close()
                elif hasattr(self.model, "stop_multi_process_pool"):
                    self.model.stop_multi_process_pool()
                else:
                    del self.model
            except Exception as e:
                logger.debug(f"Error cleaning up BGE model: {e}")
            finally:
                self.model = None
                self.index = None
                self.documents = []


# In-memory index cache to avoid re-embedding files on every search request
# Schema: { (tool_name, condensed): (faiss.Index, List[Document]) }
_INDEX_CACHE: Dict[tuple, tuple] = {}


# Singleton / cached instance of the indexer to keep model weights loaded
_INDEXER: Optional[_TutorialIndexer] = None

def get_indexer() -> _TutorialIndexer:
    global _INDEXER
    if _INDEXER is None:
        _INDEXER = _TutorialIndexer()
    return _INDEXER


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Server Definition
# ═══════════════════════════════════════════════════════════════════════════

mcp = FastMCP("Semantic Vector Store Server", host="0.0.0.0")


@mcp.tool()
def retrieve_tutorials(
    query: str,
    tool_name: str,
    top_k: int = 5,
    condensed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Retrieve matching tutorials from the vector store index.
    Creates and loads the FAISS store index dynamically when invoked.
    """
    logger.info(f"─── retrieve_tutorials (tool_name={tool_name}, top_k={top_k}) ───")
    
    cache_key = (tool_name, condensed)
    indexer = get_indexer()

    # Check if index for this tool is already built and cached
    if cache_key in _INDEX_CACHE:
        logger.info(f"Using cached FAISS index for {tool_name} (condensed={condensed})")
        indexer.index, indexer.documents = _INDEX_CACHE[cache_key]
    else:
        # 1. Parse documents using the LocalTutorialParser
        registry_path = get_registry_path()
        parser = LocalTutorialParser(registry_path)
        documents = parser.parse_tutorials(tool_name, condensed=condensed)
        
        if not documents:
            logger.warning(f"No documents loaded for tool '{tool_name}' (condensed={condensed})")
            return []

        # 2. Build index dynamically for this call
        indexer.build_index(documents)

        # 3. Cache the index
        if indexer.index is not None:
            _INDEX_CACHE[cache_key] = (indexer.index, indexer.documents)

    # 4. Perform retrieval search
    results = indexer.search(query, top_k=top_k)
    logger.info(f"Search found {len(results)} matches for query '{query[:60]}'")
    return results


# FastAPI SSE Mounting
app = FastAPI(title="Semantic Vector Store MCP Server")

@app.post("/retrieve_tutorials")
def retrieve_tutorials_direct(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Direct HTTP POST route optimized for AWS Lambda / Mangum (buffered execution)."""
    return retrieve_tutorials(
        query=payload["query"],
        tool_name=payload["tool_name"],
        top_k=payload.get("top_k", 5),
        condensed=payload.get("condensed", False)
    )

app.mount("/", mcp.sse_app())


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Ensure the parent folder of this server is in sys.path so uvicorn can import the module
    curr_dir = Path(__file__).resolve().parent
    if str(curr_dir) not in sys.path:
        sys.path.insert(0, str(curr_dir))
        
    logger.info("Starting Semantic Vector Store MCP Server on port 8010...")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8010, log_level="info")

