"""
Factory function for creating the configured LLM provider.
"""

import logging
from typing import Optional

from llm.base import BaseLLM

logger = logging.getLogger(__name__)


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Create and return the appropriate LLM based on configuration.

    Args:
        provider: Override provider ("local" or "ibm"). Defaults to config.

    Returns:
        An instantiated LLM provider ready for use.
    """
    import config  # lazy import to avoid circular deps at module load

    effective_provider = provider or config.get_active_provider()

    if effective_provider == "ibm":
        from llm.granite import GraniteLLM
        logger.info("Using IBM Granite LLM via watsonx.ai")
        return GraniteLLM(
            model=config.GRANITE_MODEL,
            api_key=config.IBM_CLOUD_API_KEY,
            project_id=config.WATSONX_PROJECT_ID,
            url=config.WATSONX_URL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
    else:
        from llm.local import LocalLLM
        logger.info(f"Using local Ollama LLM: {config.OLLAMA_MODEL}")
        return LocalLLM(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
