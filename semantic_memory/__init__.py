"""
Semantic Memory Module — Tutorial Retrieval via FAISS + BGE embeddings.

Maps to autogluon-assistant's RetrieverAgent.
Uses an LLM to generate a search query, then performs semantic search
over tool tutorials using FAISS indices.
"""

from .graph import build_semantic_memory_graph
from .nodes import retrieve_tutorials
