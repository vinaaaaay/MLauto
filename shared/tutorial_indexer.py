"""
Tutorial indexer using FAISS + BGE embeddings for semantic retrieval.

Ported from autogluon-assistant/src/autogluon/assistant/tools_registry/indexing.py.
Indexes tutorial summaries and supports top-k semantic search.
"""

import contextlib
import io
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from FlagEmbedding import FlagModel

logger = logging.getLogger(__name__)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class TutorialIndexer:
    """
    Indexes tutorial summaries using FAISS and BGE embeddings for efficient retrieval.
    Maintains separate indices for regular and condensed tutorials for each tool.
    """

    def __init__(self, registry_path: Path = None, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.registry_path = registry_path or (Path(__file__).parent.parent / "tools_registry")
        self.embedding_model_name = embedding_model_name
        self.sanitized_model_name = self.embedding_model_name.replace("/", "_")
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
        """Get the tutorials folder for a specific tool."""
        subfolder = "condensed_tutorials" if condensed else "tutorials"
        # Try to find tool directory (handle spaces in names)
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
            # Auto-discover tools by scanning registry directories
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
