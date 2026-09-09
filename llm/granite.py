"""
IBM Granite LLM provider via watsonx.ai.
Used when IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID are configured.
"""

import logging
from typing import Optional

from llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class GraniteLLM(BaseLLM):
    """
    LLM provider for IBM Granite models accessed through watsonx.ai.
    Requires: ibm-watsonx-ai package and valid IBM Cloud credentials.
    """

    def __init__(
        self,
        model: str = "ibm-granite/granite-3.2-8b-instruct",
        api_key: str = "",
        project_id: str = "",
        url: str = "https://us-south.ml.cloud.ibm.com",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        self._client = None

    def _get_client(self):
        """Lazily initialize the watsonx.ai client."""
        if self._client is not None:
            return self._client
        try:
            from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
            credentials = Credentials(url=self.url, api_key=self.api_key)
            self._client = APIClient(credentials)
            return self._client
        except ImportError:
            raise ImportError(
                "ibm-watsonx-ai package not installed. "
                "Install with: pip install ibm-watsonx-ai"
            )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate text using IBM Granite via watsonx.ai."""
        if not self.api_key or not self.project_id:
            msg = (
                "IBM watsonx.ai credentials not configured. "
                "Set IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID in your .env file."
            )
            return LLMResponse(text="", model=self.model, provider="ibm", error=msg)

        try:
            from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # type: ignore

            client = self._get_client()

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"

            model = ModelInference(
                model_id=self.model,
                api_client=client,
                project_id=self.project_id,
                params={
                    GenParams.MAX_NEW_TOKENS: self.max_tokens,
                    GenParams.TEMPERATURE: self.temperature,
                    GenParams.DECODING_METHOD: "greedy",
                },
            )

            result = model.generate_text(prompt=full_prompt)
            return LLMResponse(
                text=result.strip(),
                model=self.model,
                provider="ibm",
            )

        except ImportError as e:
            msg = f"IBM watsonx-ai library error: {e}"
            logger.error(msg)
            return LLMResponse(text="", model=self.model, provider="ibm", error=msg)

        except Exception as e:
            msg = f"IBM Granite generation error: {e}"
            logger.exception(msg)
            return LLMResponse(text="", model=self.model, provider="ibm", error=msg)

    def health_check(self) -> tuple[bool, str]:
        """Check if IBM watsonx.ai credentials are valid."""
        if not self.api_key:
            return False, "IBM_CLOUD_API_KEY not configured."
        if not self.project_id:
            return False, "WATSONX_PROJECT_ID not configured."

        try:
            client = self._get_client()
            # A lightweight check — list foundation models
            client.foundation_models.get_custom_model_specs()
            return True, f"IBM watsonx.ai connected. Model: {self.model}"
        except ImportError:
            return False, "ibm-watsonx-ai package not installed."
        except Exception as e:
            # Token / connectivity errors will surface here
            return False, f"IBM watsonx.ai connection failed: {e}"
