"""
LLM initialization helper — OpenAI only.

Replaces the entire llm/ directory + ChatLLMFactory from autogluon-assistant.
"""

import os
import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def get_llm(config: dict = None) -> ChatOpenAI:
    """
    Create and return a configured ChatOpenAI instance.

    Args:
        config: Optional dict with keys: model, temperature, max_tokens.
                Falls back to sensible defaults.

    Returns:
        A ChatOpenAI instance ready for .invoke() or .ainvoke().
    """
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

    # Reasoning models (o1, o3, gpt-5) have strict parameter rules
    is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])

    if is_reasoning_model:
        logger.info(f"Detected reasoning model. Forcing temp=1 and using max_completion_tokens.")
        llm = ChatOpenAI(
            model=model,
            temperature=1,  # Must be 1
            max_completion_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    logger.info(f"Initialized OpenAI LLM: model={model}, temp={temperature}")
    return llm
