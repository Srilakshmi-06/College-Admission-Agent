"""
Document chunker — splits raw documents into overlapping text chunks
with rich metadata for retrieval.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from rag.loader import RawDocument

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A text chunk ready for embedding and indexing."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


class DocumentChunker:
    """
    Splits RawDocuments into overlapping text chunks.
    Uses a sentence-aware strategy: splits on paragraph/sentence boundaries
    then groups chunks to the target size with overlap.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Target character count per chunk.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: RawDocument) -> list[TextChunk]:
        """
        Chunk a single RawDocument into TextChunks.

        Args:
            doc: A successfully loaded RawDocument.

        Returns:
            List of TextChunk objects with metadata.
        """
        if not doc.success:
            logger.warning(f"Skipping failed document: {doc.file_name}")
            return []

        chunks: list[TextChunk] = []

        for page_data in doc.pages:
            page_num = page_data["page"]
            page_text = page_data["text"]

            if not page_text.strip():
                continue

            page_chunks = self._split_text(page_text)

            for i, chunk_text in enumerate(page_chunks):
                if not chunk_text.strip():
                    continue

                chunk_id = f"{doc.file_name}__p{page_num}__c{i}"
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    metadata={
                        "source": doc.file_name,
                        "source_path": doc.file_path,
                        "doc_type": doc.doc_type,
                        "page": page_num,
                        "chunk_index": i,
                        "total_pages": doc.total_pages,
                        "section": self._infer_section(doc.file_name),
                    },
                )
                chunks.append(chunk)

        logger.info(
            f"Chunked '{doc.file_name}': {len(doc.pages)} pages → {len(chunks)} chunks"
        )
        return chunks

    def chunk_documents(self, docs: list[RawDocument]) -> list[TextChunk]:
        """Chunk a list of documents."""
        all_chunks: list[TextChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Total chunks: {len(all_chunks)} from {len(docs)} documents")
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks of roughly `chunk_size` characters
        with `chunk_overlap` character overlap.
        Tries to break on paragraph / sentence boundaries.
        """
        # Split into sentences/paragraphs
        sentences = self._split_sentences(text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)

            if current_len + s_len > self.chunk_size and current_chunk:
                # Emit current chunk
                chunks.append(" ".join(current_chunk))
                # Carry over overlap
                overlap_text = " ".join(current_chunk)
                overlap_start = max(0, len(overlap_text) - self.chunk_overlap)
                overlap_part = overlap_text[overlap_start:]
                current_chunk = [overlap_part] if overlap_part.strip() else []
                current_len = len(overlap_part)

            current_chunk.append(sentence)
            current_len += s_len + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into paragraphs/sentences for chunking."""
        # First split on double newlines (paragraphs)
        paragraphs = re.split(r"\n{2,}", text)
        sentences: list[str] = []
        for para in paragraphs:
            # Further split on sentence boundaries
            parts = re.split(r"(?<=[.!?])\s+", para.strip())
            sentences.extend(p for p in parts if p.strip())
        return sentences

    @staticmethod
    def _infer_section(file_name: str) -> str:
        """Infer a section label from the document filename."""
        name = file_name.lower().replace("_", " ").replace("-", " ")
        mapping = {
            "course": "Courses",
            "eligib": "Eligibility",
            "fee": "Fees",
            "scholarsh": "Scholarships",
            "document": "Documents Required",
            "admission": "Admission Policy",
            "date": "Important Dates",
            "deadline": "Important Dates",
            "faq": "FAQ",
        }
        for key, section in mapping.items():
            if key in name:
                return section
        return "General"
