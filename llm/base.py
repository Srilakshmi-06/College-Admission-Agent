"""
Base LLM interface.
All LLM providers must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Structured response from an LLM."""
    text: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text)


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The user prompt / question.
            system_prompt: Optional system instruction for the model.

        Returns:
            LLMResponse with the generated text and metadata.
        """
        ...

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """
        Check if this LLM provider is available and working.

        Returns:
            Tuple of (is_available: bool, message: str)
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
