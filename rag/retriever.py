"""
RAG Retriever — combines embedding model + vector store to retrieve
relevant chunks for a query, with deduplication and ranking.
"""

import logging
from typing import Optional

import config
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Orchestrates the retrieval step of the RAG pipeline.
    Accepts a user query and returns ranked, deduplicated text chunks.
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 4,
        similarity_threshold: float = 0.2,
    ):
        self.embedding_model = embedding_model or EmbeddingModel(config.EMBEDDING_MODEL)
        self.vector_store = vector_store or VectorStore(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
            info_path=config.INDEX_INFO_PATH,
        )
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self._loaded = False

    def ensure_loaded(self) -> bool:
        """Load the vector store if not already loaded."""
        if not self._loaded:
            self._loaded = self.vector_store.load()
        return self._loaded

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question.
            top_k: Override the default top-k.
            filter_source: Optionally restrict results to a specific document.

        Returns:
            List of result dicts: [{text, metadata, score}]
        """
        if not self.ensure_loaded():
            logger.warning("Vector store not loaded — returning empty results.")
            return []

        k = top_k or self.top_k

        try:
            query_embedding = self.embedding_model.embed_text(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

        results = self.vector_store.search(
            query_embedding,
            top_k=k * 2,  # fetch extra to allow dedup/filtering
            threshold=self.similarity_threshold,
        )

        # Filter by source if requested
        if filter_source:
            results = [r for r in results if r["metadata"].get("source") == filter_source]

        # Deduplicate by chunk text (keep highest score)
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            text_key = r["text"][:100]
            if text_key not in seen:
                seen.add(text_key)
                deduped.append(r)

        return deduped[:k]

    def retrieve_for_agents(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve chunks and return them with formatted source citations.
        """
        results = self.retrieve(query, top_k=top_k)
        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM prompt.

        Args:
            results: List of result dicts from retrieve().

        Returns:
            Formatted context string.
        """
        if not results:
            return "No relevant information found in the knowledge base."

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = meta.get("source", "Unknown")
            page = meta.get("page", "")
            page_str = f", Page {page}" if page else ""
            parts.append(
                f"[Source {i}: {source}{page_str}]\n{r['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def format_sources(self, results: list[dict]) -> list[dict]:
        """
        Format source citations for display in the UI.

        Returns:
            List of source dicts: [{source, page, section, score, snippet}]
        """
        sources: list[dict] = []
        seen_sources: set[str] = set()
        for r in results:
            meta = r["metadata"]
            source = meta.get("source", "Unknown")
            page = meta.get("page", "")
            key = f"{source}_p{page}"
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({
                    "source": source,
                    "page": page,
                    "section": meta.get("section", ""),
                    "score": round(r["score"], 3),
                    "snippet": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                })
        return sources

    @property
    def is_ready(self) -> bool:
        """True if the retriever has a loaded vector store."""
        return self.ensure_loaded() and self.vector_store.is_ready
