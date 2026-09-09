"""
FAISS vector store — persists chunk embeddings and metadata,
supports similarity search with metadata filtering.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages a FAISS flat-index for dense similarity search.
    Stores chunk texts and metadata alongside the index.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        info_path: str,
        dimension: Optional[int] = None,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.info_path = Path(info_path)
        self.dimension = dimension
        self._index = None
        self._chunks: list[dict] = []   # parallel list: [{text, metadata}]
        self._info: dict = {}

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, chunks) -> None:
        """
        Build a FAISS index from embeddings and TextChunk objects.

        Args:
            embeddings: 2-D float32 array of shape (N, D).
            chunks: List of TextChunk objects (same order as embeddings).
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings to index.")

        n, d = embeddings.shape
        self.dimension = d

        logger.info(f"Building FAISS index: {n} vectors, dim={d}")
        index = faiss.IndexFlatIP(d)   # inner product (cosine after normalisation)
        index.add(embeddings)
        self._index = index

        # Store chunks as serialisable dicts
        self._chunks = [
            {"text": c.text, "metadata": c.metadata} for c in chunks
        ]

        self._info = {
            "num_chunks": n,
            "dimension": d,
            "num_documents": len({c.metadata.get("source") for c in chunks}),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "documents": sorted({c.metadata.get("source", "") for c in chunks}),
        }

        self._save()
        logger.info("FAISS index built and saved.")

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss  # type: ignore
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path) + ".faiss")
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)
        with open(self.info_path, "w", encoding="utf-8") as f:
            json.dump(self._info, f, ensure_ascii=False, indent=2)
        logger.info(f"Index saved to {self.index_path}.faiss")

    def load(self) -> bool:
        """
        Load a previously built index from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        faiss_file = Path(str(self.index_path) + ".faiss")
        if not faiss_file.exists():
            logger.warning(f"FAISS index not found at {faiss_file}")
            return False

        try:
            import faiss  # type: ignore
            self._index = faiss.read_index(str(faiss_file))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)
            if self.info_path.exists():
                with open(self.info_path, "r", encoding="utf-8") as f:
                    self._info = json.load(f)
            self.dimension = self._index.d
            logger.info(
                f"Loaded FAISS index: {self._index.ntotal} vectors, dim={self.dimension}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 4, threshold: float = 0.0
    ) -> list[dict]:
        """
        Search the index for the most similar chunks.

        Args:
            query_embedding: 1-D float32 array (query vector).
            top_k: Number of results to return.
            threshold: Minimum similarity score (0–1 for normalised vectors).

        Returns:
            List of dicts: [{text, metadata, score}], sorted by score descending.
        """
        if self._index is None:
            logger.error("Vector store not loaded. Call load() first.")
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        scores, indices = self._index.search(query, k)
        results: list[dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            if float(score) < threshold:
                continue
            entry = self._chunks[idx]
            results.append({
                "text": entry["text"],
                "metadata": entry["metadata"],
                "score": float(score),
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if an index is loaded and has vectors."""
        return self._index is not None and self._index.ntotal > 0

    @property
    def info(self) -> dict:
        """Return stored index metadata."""
        if self._info:
            return self._info
        if self.is_ready:
            return {
                "num_chunks": self._index.ntotal,
                "dimension": self.dimension,
                "num_documents": len({c["metadata"].get("source") for c in self._chunks}),
            }
        return {}

    def clear(self) -> None:
        """Remove all index files and reset state."""
        for p in [
            Path(str(self.index_path) + ".faiss"),
            self.metadata_path,
            self.info_path,
        ]:
            if p.exists():
                p.unlink()
        self._index = None
        self._chunks = []
        self._info = {}
        logger.info("Vector store cleared.")
