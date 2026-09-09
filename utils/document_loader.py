"""
High-level document loading and ingestion utilities.
Used by both the CLI ingest.py script and the Streamlit UI.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import config
from rag.loader import DocumentLoader
from rag.chunker import DocumentChunker
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_documents(
    data_dir: Optional[str] = None,
    extra_files: Optional[list[str]] = None,
    progress_callback=None,
) -> dict:
    """
    Full ingestion pipeline: load → chunk → embed → index.

    Args:
        data_dir: Path to directory of documents. Defaults to config.DATA_DIR.
        extra_files: Additional file paths to include.
        progress_callback: Optional callable(step: str, pct: float) for UI progress.

    Returns:
        Dict with ingestion results and stats.
    """
    def _progress(step: str, pct: float):
        if progress_callback:
            progress_callback(step, pct)
        logger.info(f"[{pct:.0f}%] {step}")

    result = {
        "success": False,
        "num_documents": 0,
        "num_chunks": 0,
        "errors": [],
        "documents": [],
    }

    # 1. Load documents
    _progress("Loading documents...", 5)
    loader = DocumentLoader()
    raw_docs = []

    dir_ = Path(data_dir or config.DATA_DIR)
    if dir_.exists():
        raw_docs.extend(loader.load_directory(str(dir_)))
    else:
        logger.warning(f"Data directory not found: {dir_}")

    if extra_files:
        for f in extra_files:
            doc = loader.load_file(f)
            raw_docs.append(doc)

    successful_docs = [d for d in raw_docs if d.success]
    failed_docs = [d for d in raw_docs if not d.success]

    for fd in failed_docs:
        result["errors"].append(f"{fd.file_name}: {fd.error}")
        logger.warning(f"Failed to load: {fd.file_name} — {fd.error}")

    if not successful_docs:
        msg = "No documents could be loaded."
        result["errors"].append(msg)
        logger.error(msg)
        return result

    result["num_documents"] = len(successful_docs)
    result["documents"] = [d.file_name for d in successful_docs]
    _progress(f"Loaded {len(successful_docs)} document(s)", 20)

    # 2. Chunk documents
    _progress("Splitting documents into chunks...", 35)
    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(successful_docs)

    if not chunks:
        msg = "No text chunks were produced. Documents may be empty."
        result["errors"].append(msg)
        return result

    result["num_chunks"] = len(chunks)
    _progress(f"Created {len(chunks)} chunks", 50)

    # 3. Generate embeddings
    _progress("Generating embeddings...", 60)
    embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
    texts = [c.text for c in chunks]

    try:
        embeddings = embedding_model.embed_texts(texts, batch_size=32)
    except Exception as e:
        msg = f"Embedding generation failed: {e}"
        result["errors"].append(msg)
        logger.error(msg)
        return result

    _progress(f"Embeddings generated: {embeddings.shape}", 80)

    # 4. Build FAISS index
    _progress("Building vector index...", 90)
    vector_store = VectorStore(
        index_path=config.FAISS_INDEX_PATH,
        metadata_path=config.METADATA_PATH,
        info_path=config.INDEX_INFO_PATH,
        dimension=embedding_model.dimension,
    )

    try:
        vector_store.build(embeddings, chunks)
    except Exception as e:
        msg = f"FAISS index build failed: {e}"
        result["errors"].append(msg)
        logger.error(msg)
        return result

    _progress("Knowledge base ready!", 100)
    result["success"] = True
    return result


def get_index_status() -> dict:
    """
    Return the current state of the FAISS index without loading it fully.

    Returns:
        Dict with index info or a 'not found' status.
    """
    import json

    info_path = Path(config.INDEX_INFO_PATH)
    faiss_path = Path(config.FAISS_INDEX_PATH + ".faiss")

    if not faiss_path.exists():
        return {
            "exists": False,
            "num_documents": 0,
            "num_chunks": 0,
            "dimension": None,
            "built_at": None,
            "documents": [],
        }

    info = {}
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            pass

    return {
        "exists": True,
        "num_documents": info.get("num_documents", "?"),
        "num_chunks": info.get("num_chunks", "?"),
        "dimension": info.get("dimension", "?"),
        "built_at": info.get("built_at", "Unknown"),
        "documents": info.get("documents", []),
    }


def save_uploaded_file(uploaded_file, dest_dir: Optional[str] = None) -> tuple[bool, str]:
    """
    Save a Streamlit UploadedFile object to the data directory.

    Args:
        uploaded_file: st.UploadedFile object.
        dest_dir: Destination directory. Defaults to config.DATA_DIR.

    Returns:
        Tuple of (success: bool, message: str)
    """
    from utils.validators import validate_uploaded_file

    valid, msg = validate_uploaded_file(uploaded_file)
    if not valid:
        return False, msg

    dest = Path(dest_dir or config.DATA_DIR) / uploaded_file.name
    try:
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file: {dest}")
        return True, f"File '{uploaded_file.name}' uploaded successfully."
    except Exception as e:
        msg = f"Failed to save file: {e}"
        logger.error(msg)
        return False, msg
