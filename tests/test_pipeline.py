"""
tests/test_pipeline.py — Integration and unit tests for the AI College Admission Agent.

Run with:
    python tests/test_pipeline.py
    python tests/test_pipeline.py --verbose
    python tests/test_pipeline.py --skip-llm   (skip tests that require LLM)

Tests:
  1. Document loading (TXT, PDF structure)
  2. Text chunking
  3. Embedding generation
  4. FAISS index build and search
  5. Full ingestion pipeline
  6. Query router classification
  7. Retriever (requires index)
  8. Agent responses (requires LLM + index)
  9. Anti-hallucination / unknown topic handling
  10. Validator utilities
"""

import sys
import os
import argparse
import time
import traceback
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402

# ── Colours for terminal output ────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"
SEP = "-" * 60


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    # Encode-safe output for Windows terminals
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(f"  [FAIL] {safe}")


def skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"{SEP}")


# ── Test result tracker ────────────────────────────────────────────────────
results = {"passed": 0, "failed": 0, "skipped": 0}


def run_test(name: str, fn, *args, **kwargs):
    """Run a single test function and track results."""
    try:
        fn(*args, **kwargs)
        results["passed"] += 1
    except AssertionError as e:
        fail(f"{name}: {e}")
        results["failed"] += 1
    except Exception as e:
        fail(f"{name}: {type(e).__name__}: {e}")
        if kwargs.get("verbose"):
            traceback.print_exc()
        results["failed"] += 1


def skip_test(name: str, reason: str = ""):
    skip(f"{name}{(' — ' + reason) if reason else ''}")
    results["skipped"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# 1. Document Loader Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_load_txt_file():
    from rag.loader import DocumentLoader
    loader = DocumentLoader()

    # Create a temp TXT file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("This is a test document.\nIt has two lines.\n\nSecond paragraph.")
        tmp_path = f.name

    try:
        doc = loader.load_file(tmp_path)
        assert doc.success, f"Expected success, got error: {doc.error}"
        assert doc.doc_type == "txt"
        assert len(doc.pages) == 1
        assert "test document" in doc.full_text
        ok("Load TXT file")
    finally:
        os.unlink(tmp_path)


def test_load_missing_file():
    from rag.loader import DocumentLoader
    loader = DocumentLoader()
    doc = loader.load_file("/nonexistent/path/file.txt")
    assert not doc.success
    assert "not found" in (doc.error or "").lower()
    ok("Load missing file returns error")


def test_load_unsupported_extension():
    from rag.loader import DocumentLoader
    loader = DocumentLoader()
    doc = loader.load_file("document.xyz")
    assert not doc.success
    assert doc.error is not None
    ok("Unsupported extension returns error")


def test_load_data_directory():
    from rag.loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_directory(str(config.DATA_DIR))
    assert len(docs) > 0, "Expected at least 1 document in data/"
    loaded = [d for d in docs if d.success]
    assert len(loaded) > 0, f"No documents loaded successfully from {config.DATA_DIR}"
    ok(f"Load data directory: {len(loaded)}/{len(docs)} documents loaded")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunker Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_basic_chunking():
    from rag.loader import DocumentLoader, RawDocument
    from rag.chunker import DocumentChunker

    doc = RawDocument(
        file_path="/test/test.txt",
        file_name="test.txt",
        doc_type="txt",
        pages=[{"page": 1, "text": "Sentence one. Sentence two. Sentence three. " * 20}],
        total_pages=1,
    )
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1, "Expected at least 1 chunk"
    for c in chunks:
        assert c.text.strip(), "Chunk text must not be empty"
        assert "source" in c.metadata
        assert c.metadata["source"] == "test.txt"
    ok(f"Basic chunking: {len(chunks)} chunks from 1 document")


def test_chunk_metadata():
    from rag.loader import RawDocument
    from rag.chunker import DocumentChunker

    doc = RawDocument(
        file_path="/test/eligibility.txt",
        file_name="eligibility.txt",
        doc_type="txt",
        pages=[{"page": 1, "text": "Eligibility: Minimum 60% in PCM."}],
        total_pages=1,
    )
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    meta = chunks[0].metadata
    assert meta.get("section") == "Eligibility"
    assert meta.get("page") == 1
    ok("Chunk metadata includes correct section and page")


def test_chunk_real_documents():
    from rag.loader import DocumentLoader
    from rag.chunker import DocumentChunker

    loader = DocumentLoader()
    docs = loader.load_directory(str(config.DATA_DIR))
    loaded = [d for d in docs if d.success]

    if not loaded:
        raise AssertionError("No documents loaded — cannot test chunking")

    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(loaded)
    assert len(chunks) > 0, "Expected chunks from real documents"
    ok(f"Chunk real documents: {len(chunks)} total chunks from {len(loaded)} documents")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_embed_single_text():
    import numpy as np
    from rag.embeddings import EmbeddingModel

    model = EmbeddingModel(config.EMBEDDING_MODEL)
    vec = model.embed_text("Test sentence for embedding")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) > 0
    assert vec.dtype == np.float32
    ok(f"Single text embedding: shape={vec.shape}, dtype={vec.dtype}")


def test_embed_batch():
    import numpy as np
    from rag.embeddings import EmbeddingModel

    model = EmbeddingModel(config.EMBEDDING_MODEL)
    texts = ["First sentence.", "Second sentence.", "Third sentence about AI."]
    vecs = model.embed_texts(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 3
    assert vecs.ndim == 2
    ok(f"Batch embedding: shape={vecs.shape}")


def test_embedding_health_check():
    from rag.embeddings import EmbeddingModel

    model = EmbeddingModel(config.EMBEDDING_MODEL)
    ok_flag, msg = model.health_check()
    assert ok_flag, f"Embedding health check failed: {msg}"
    ok(f"Embedding health check: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FAISS Vector Store Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_faiss_build_and_search():
    import numpy as np
    from rag.vector_store import VectorStore
    from rag.chunker import TextChunk

    # Build a small in-memory-like index in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        vs = VectorStore(
            index_path=os.path.join(tmpdir, "test_index"),
            metadata_path=os.path.join(tmpdir, "test_meta.json"),
            info_path=os.path.join(tmpdir, "test_info.json"),
        )

        # Create fake embeddings and chunks
        dim = 10
        n = 5
        embeddings = np.random.rand(n, dim).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        chunks = [
            TextChunk(
                chunk_id=f"chunk_{i}",
                text=f"Sample text {i} about admission",
                metadata={"source": "test.txt", "page": i, "section": "Test"},
            )
            for i in range(n)
        ]

        vs.build(embeddings, chunks)
        assert vs.is_ready, "Vector store should be ready after build"

        # Search
        query = np.random.rand(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = vs.search(query, top_k=3)
        assert len(results) <= 3
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("score" in r for r in results)

    ok("FAISS build and search")


def test_faiss_persist_and_reload():
    import numpy as np
    from rag.vector_store import VectorStore
    from rag.chunker import TextChunk

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = os.path.join(tmpdir, "persist_index")
        meta_path = os.path.join(tmpdir, "persist_meta.json")
        info_path = os.path.join(tmpdir, "persist_info.json")

        # Build
        dim = 8
        n = 4
        embeddings = np.random.rand(n, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        chunks = [
            TextChunk(
                chunk_id=f"c{i}",
                text=f"Text chunk {i}",
                metadata={"source": "doc.txt", "page": 1, "section": "Test"},
            )
            for i in range(n)
        ]

        vs1 = VectorStore(idx_path, meta_path, info_path)
        vs1.build(embeddings, chunks)

        # Reload from disk
        vs2 = VectorStore(idx_path, meta_path, info_path)
        loaded = vs2.load()
        assert loaded, "Reload from disk should return True"
        assert vs2.is_ready, "Reloaded vector store should be ready"
        assert vs2._index.ntotal == n

    ok("FAISS persist and reload")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Ingestion Pipeline Test
# ─────────────────────────────────────────────────────────────────────────────

def test_full_ingestion():
    from utils.document_loader import ingest_documents

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small TXT document
        doc_path = os.path.join(tmpdir, "test_admission.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "B.Tech Computer Science Engineering\n"
                "Eligibility: Minimum 60% in Class 12 with PCM subjects.\n"
                "Duration: 4 years. Total seats: 60.\n"
                "Scholarship available for top 5 students.\n"
                "Application deadline: April 30, 2025.\n"
            )

        with tempfile.TemporaryDirectory() as idx_dir:
            # Patch config temporarily
            original_faiss = config.FAISS_INDEX_PATH
            original_meta = config.METADATA_PATH
            original_info = config.INDEX_INFO_PATH

            config.FAISS_INDEX_PATH = os.path.join(idx_dir, "faiss_index")
            config.METADATA_PATH = os.path.join(idx_dir, "metadata.json")
            config.INDEX_INFO_PATH = os.path.join(idx_dir, "index_info.json")

            try:
                result = ingest_documents(data_dir=tmpdir)
                assert result["success"], f"Ingestion failed: {result['errors']}"
                assert result["num_documents"] >= 1
                assert result["num_chunks"] >= 1
                ok(f"Full ingestion pipeline: {result['num_documents']} docs, {result['num_chunks']} chunks")
            finally:
                config.FAISS_INDEX_PATH = original_faiss
                config.METADATA_PATH = original_meta
                config.INDEX_INFO_PATH = original_info


# ─────────────────────────────────────────────────────────────────────────────
# 6. Query Router Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_query_router():
    from agents.router import QueryRouter, QueryCategory

    router = QueryRouter()
    test_cases = [
        ("What courses are available?", QueryCategory.COURSE),
        ("Am I eligible for B.Tech?", QueryCategory.ELIGIBILITY),
        ("What is the tuition fee?", QueryCategory.FEES),
        ("When is the application deadline?", QueryCategory.DEADLINE),
        ("What documents do I need?", QueryCategory.DOCUMENTS),
        ("How do I apply?", QueryCategory.APPLICATION),
        ("What scholarships are available?", QueryCategory.SCHOLARSHIP),
        ("Tell me about the college", QueryCategory.GENERAL),
    ]
    errors = []
    for query, expected in test_cases:
        result = router.classify(query)
        if result != expected:
            errors.append(f"'{query}' → got {result.value}, expected {expected.value}")

    if errors:
        raise AssertionError("Router misclassifications:\n" + "\n".join(errors))
    ok(f"Query router: {len(test_cases)}/{len(test_cases)} classifications correct")


def test_router_multi_agent():
    from agents.router import QueryRouter, QueryCategory

    router = QueryRouter()
    # A query touching fees + eligibility + courses should be MULTI
    query = "I studied maths and CS and want an affordable AI course. Am I eligible?"
    category = router.classify(query)
    # Either MULTI or one of the involved categories is acceptable
    assert category in (
        QueryCategory.MULTI, QueryCategory.ELIGIBILITY,
        QueryCategory.COURSE, QueryCategory.FEES,
    ), f"Unexpected category: {category}"
    ok(f"Router multi-agent detection: {category.value}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Retriever Test (requires index)
# ─────────────────────────────────────────────────────────────────────────────

def test_retriever_with_real_index():
    from rag.retriever import RAGRetriever

    retriever = RAGRetriever(top_k=3, similarity_threshold=0.1)
    if not retriever.is_ready:
        raise AssertionError(
            "FAISS index not found. Run 'python ingest.py' first."
        )

    results = retriever.retrieve("what courses are available", top_k=3)
    assert isinstance(results, list), "Expected list of results"
    if results:
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]
        ok(f"Retriever: {len(results)} results for 'what courses are available'")
    else:
        ok("Retriever returned 0 results (index may be empty or threshold too high)")


# ─────────────────────────────────────────────────────────────────────────────
# 8 & 9. Agent Tests (require LLM + index)
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TEST_QUERIES = [
    ("Course query", "Which courses are suitable for a computer science student?"),
    ("Eligibility query", "Am I eligible for B.Tech Computer Science with 85% in PCM?"),
    ("Fees query", "What is the tuition fee for B.Tech?"),
    ("Scholarship query", "What scholarships are available?"),
    ("Documents query", "What documents are required for admission?"),
    ("Deadline query", "When is the application deadline?"),
    ("Application query", "How do I apply for admission?"),
    (
        "Personalized recommendation",
        "I studied Maths and Computer Science and I am interested in AI. Which course should I choose?",
    ),
    ("Unknown topic", "What is the hostel fee for international students from Mars?"),
]


def test_agent_with_llm(query_label: str, query: str, coordinator, verbose: bool = False):
    """Run a single agent query and validate the response."""
    response, category = coordinator.process(query=query)

    # Basic assertions
    assert response is not None, "Response must not be None"

    if not response.success:
        # LLM may be unavailable — mark as warning, not failure
        raise AssertionError(
            f"Agent returned error: {response.error}"
        )

    assert len(response.text) > 10, "Response text too short"
    assert response.agent_name, "Response must have agent_name"

    # Anti-hallucination check for unknown topic
    if "international students from Mars" in query:
        # The system must NOT invent information
        hallucination_phrases = [
            "the hostel fee for international students from mars is",
            "international students from mars pay",
        ]
        text_lower = response.text.lower()
        for phrase in hallucination_phrases:
            assert phrase not in text_lower, (
                f"Possible hallucination detected for unknown query. "
                f"Response: {response.text[:200]}"
            )

    if verbose:
        print(f"\n  Query: {query}")
        print(f"  Agent: {response.agent_name}")
        print(f"  Category: {category.value}")
        print(f"  Sources: {len(response.sources)}")
        print(f"  Response: {response.text[:150]}...")

    ok(f"{query_label}: Agent={response.agent_name}, Sources={len(response.sources)}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Validator Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_query_validation():
    from utils.validators import sanitize_query

    # Valid queries
    valid_cases = [
        "What courses are available?",
        "Am I eligible for B.Tech AI?",
        "Tell me about scholarships",
    ]
    for q in valid_cases:
        valid, result = sanitize_query(q)
        assert valid, f"Expected valid query: '{q}', got error: {result}"

    # Invalid queries
    invalid_cases = [
        ("", "Empty query"),
        ("a", "Too short"),
        ("x" * 3000, "Too long"),
    ]
    for q, label in invalid_cases:
        valid, _ = sanitize_query(q)
        assert not valid, f"Expected invalid query for {label}: '{q[:20]}...'"

    ok("Query validation: valid and invalid cases")


def test_file_validation():
    from utils.validators import validate_uploaded_file

    class MockFile:
        def __init__(self, name, size=1000):
            self.name = name
            self.size = size

    # Valid
    valid = validate_uploaded_file(MockFile("document.pdf"))
    assert valid[0], f"Expected valid: {valid[1]}"

    valid = validate_uploaded_file(MockFile("notes.txt"))
    assert valid[0], f"Expected valid: {valid[1]}"

    valid = validate_uploaded_file(MockFile("report.docx"))
    assert valid[0], f"Expected valid: {valid[1]}"

    # Invalid extension
    invalid = validate_uploaded_file(MockFile("script.py"))
    assert not invalid[0], "Expected invalid extension"

    # Path traversal
    invalid2 = validate_uploaded_file(MockFile("../../etc/passwd"))
    assert not invalid2[0], "Expected path traversal to be rejected"

    ok("File validation: accepted and rejected cases")


# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run AI College Admission Agent tests")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip tests that require an LLM (agent tests)"
    )
    parser.add_argument(
        "--skip-index", action="store_true",
        help="Skip tests that require a built FAISS index"
    )
    args = parser.parse_args()
    verbose = args.verbose

    start_time = time.time()
    banner = "=" * 60
    print(f"\n{banner}")
    print("  AI College Admission Agent - Test Suite")
    print(f"{banner}")

    # 1. Document Loading
    header("1. Document Loader")
    run_test("Load TXT file", test_load_txt_file)
    run_test("Load missing file", test_load_missing_file)
    run_test("Load unsupported extension", test_load_unsupported_extension)
    run_test("Load data directory", test_load_data_directory)

    # ── 2. Chunking ──
    header("2. Document Chunker")
    run_test("Basic chunking", test_basic_chunking)
    run_test("Chunk metadata", test_chunk_metadata)
    run_test("Chunk real documents", test_chunk_real_documents)

    # ── 3. Embeddings ──
    header("3. Embedding Model")
    run_test("Embed single text", test_embed_single_text)
    run_test("Embed batch", test_embed_batch)
    run_test("Embedding health check", test_embedding_health_check)

    # ── 4. FAISS ──
    header("4. FAISS Vector Store")
    run_test("FAISS build and search", test_faiss_build_and_search)
    run_test("FAISS persist and reload", test_faiss_persist_and_reload)

    # ── 5. Ingestion Pipeline ──
    header("5. Full Ingestion Pipeline")
    run_test("Full ingestion pipeline", test_full_ingestion)

    # ── 6. Query Router ──
    header("6. Query Router")
    run_test("Query router classification", test_query_router)
    run_test("Router multi-agent detection", test_router_multi_agent)

    # ── 7. Retriever ──
    header("7. RAG Retriever")
    if args.skip_index:
        skip_test("Retriever with real index", "skipped (--skip-index)")
    else:
        run_test("Retriever with real index", test_retriever_with_real_index)

    # ── 8 & 9. Agent Tests ──
    header("8 & 9. Agent Tests (requires LLM + index)")
    if args.skip_llm:
        for label, _ in AGENT_TEST_QUERIES:
            skip_test(label, "skipped (--skip-llm)")
    else:
        # Check LLM availability first
        from llm.factory import get_llm
        from rag.retriever import RAGRetriever
        from agents.coordinator import AgentCoordinator

        llm = get_llm()
        llm_ok, llm_msg = llm.health_check()

        if not llm_ok:
            print(f"\n  {YELLOW}⚠️  LLM not available: {llm_msg}{RESET}")
            print(f"  {YELLOW}   Skipping agent tests. Start Ollama to run them.{RESET}")
            for label, _ in AGENT_TEST_QUERIES:
                skip_test(label, "LLM unavailable")
        else:
            retriever = RAGRetriever(top_k=config.TOP_K, similarity_threshold=0.1)
            if not retriever.is_ready:
                print(f"\n  {YELLOW}⚠️  Index not ready. Run 'python ingest.py' first.{RESET}")
                for label, _ in AGENT_TEST_QUERIES:
                    skip_test(label, "Index not ready")
            else:
                coordinator = AgentCoordinator(llm=llm, retriever=retriever)
                for label, query in AGENT_TEST_QUERIES:
                    run_test(
                        label,
                        test_agent_with_llm,
                        label, query, coordinator, verbose
                    )

    # ── 10. Validators ──
    header("10. Validators")
    run_test("Query validation", test_query_validation)
    run_test("File validation", test_file_validation)

    # Summary
    elapsed = time.time() - start_time
    total = results["passed"] + results["failed"] + results["skipped"]
    sep2 = "=" * 60
    print(f"\n{sep2}")
    print("  Test Summary")
    print(sep2)
    print(f"  Total   : {total}")
    print(f"  Passed  : {results['passed']}")
    print(f"  Failed  : {results['failed']}")
    print(f"  Skipped : {results['skipped']}")
    print(f"  Time    : {elapsed:.2f}s")
    print(f"{sep2}\n")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
