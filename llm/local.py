"""
Local LLM provider using Ollama.
Supports any model available via the Ollama API (llama3.2, mistral, etc.)
"""

import logging
import requests
from typing import Optional

from llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)

OLLAMA_GENERATE_URL = "{base_url}/api/generate"
OLLAMA_TAGS_URL = "{base_url}/api/tags"
# Generous timeout: first call loads the model into RAM (can take 60-120 s on CPU).
# Subsequent calls are much faster once the model is warm.
TIMEOUT = 300  # seconds


class LocalLLM(BaseLLM):
    """
    LLM provider that communicates with a locally running Ollama server.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url.rstrip("/")
        self._generate_url = f"{self.base_url}/api/generate"
        self._tags_url = f"{self.base_url}/api/tags"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Send a prompt to Ollama and return the generated text."""
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                self._generate_url,
                json=payload,
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            tokens = data.get("eval_count")
            return LLMResponse(
                text=text,
                model=self.model,
                provider="local",
                tokens_used=tokens,
            )

        except requests.exceptions.ConnectionError:
            msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please make sure Ollama is running: https://ollama.com"
            )
            logger.error(msg)
            return LLMResponse(text="", model=self.model, provider="local", error=msg)

        except requests.exceptions.Timeout:
            msg = f"Ollama request timed out after {TIMEOUT}s. The model may be loading."
            logger.error(msg)
            return LLMResponse(text="", model=self.model, provider="local", error=msg)

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                msg = (
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
            else:
                msg = f"Ollama HTTP error: {e}"
            logger.error(msg)
            return LLMResponse(text="", model=self.model, provider="local", error=msg)

        except Exception as e:
            msg = f"Unexpected error calling Ollama: {e}"
            logger.exception(msg)
            return LLMResponse(text="", model=self.model, provider="local", error=msg)

    def health_check(self) -> tuple[bool, str]:
        """Verify Ollama is reachable and the configured model exists."""
        try:
            response = requests.get(self._tags_url, timeout=5)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
            # Ollama tags include ":latest" suffix — check with and without
            model_names = [m.split(":")[0] for m in models]
            if self.model in models or self.model in model_names:
                return True, f"Ollama running. Model '{self.model}' available."
            available = ", ".join(models) if models else "none"
            return (
                False,
                f"Ollama running, but model '{self.model}' not found. "
                f"Available: {available}. Run: ollama pull {self.model}",
            )
        except requests.exceptions.ConnectionError:
            return (
                False,
                f"Ollama not reachable at {self.base_url}. "
                "Install from https://ollama.com and run 'ollama serve'.",
            )
        except Exception as e:
            return False, f"Ollama health check failed: {e}"

    def warm_up(self) -> None:
        """
        Send a minimal dummy prompt to Ollama so the model is loaded into
        memory before the first real user query arrives.
        This prevents the first real request from timing out on slow machines.
        Called once at app startup inside load_system().
        """
        logger.info(f"Warming up Ollama model '{self.model}'...")
        try:
            payload = {
                "model": self.model,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1, "temperature": 0},
            }
            requests.post(self._generate_url, json=payload, timeout=TIMEOUT)
            logger.info("Ollama warm-up complete.")
        except Exception as e:
            # Warm-up failure is non-fatal — log and move on
            logger.warning(f"Ollama warm-up skipped: {e}")
