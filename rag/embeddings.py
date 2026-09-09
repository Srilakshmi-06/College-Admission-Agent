"""
Embedding model wrapper using sentence-transformers.
Generates dense vector embeddings for text chunks.
"""

import logging
import threading
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton cache: model_name -> SentenceTransformer instance
# Shared across all EmbeddingModel instances with the same model name so the
# heavyweight model is only loaded once per process even if multiple
# EmbeddingModel objects are created (e.g. by RAGRetriever inside cache_resource).
_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()


class EmbeddingModel:
    """
    Generates text embeddings using a sentence-transformers model.
    The underlying SentenceTransformer is cached at the module level so it is
    only loaded once per process regardless of how many EmbeddingModel instances
    are created.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # local reference; populated on first use

    def _load_model(self):
        """Load the sentence-transformer model, using the process-level cache."""
        if self._model is not None:
            return self._model

        # Fast path: already in the module cache (no lock needed for read)
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return self._model

        # Slow path: load the model under a lock so only one thread loads it
        with _MODEL_LOCK:
            # Double-check inside the lock
            if self.model_name in _MODEL_CACHE:
                self._model = _MODEL_CACHE[self.model_name]
                return self._model

            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                logger.info(f"Loading embedding model: {self.model_name}")
                model = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = model
                self._model = model
                logger.info("Embedding model loaded successfully")
                return self._model
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {e}"
                )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Args:
            text: Input text.

        Returns:
            1-D numpy array of float32 embeddings.
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of strings in batches.

        Args:
            texts: List of input strings.
            batch_size: Number of texts to embed per batch.

        Returns:
            2-D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)

        model = self._load_model()
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        logger.info(f"Embeddings generated: shape={embeddings.shape}")
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    def health_check(self) -> tuple[bool, str]:
        """Verify the embedding model can be loaded and produces output."""
        try:
            vec = self.embed_text("test sentence")
            return True, f"Embedding model ready. Dim={len(vec)}, Model={self.model_name}"
        except Exception as e:
            return False, f"Embedding model failed: {e}"
