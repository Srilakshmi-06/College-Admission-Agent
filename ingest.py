"""
ingest.py — CLI script for building/rebuilding the knowledge base.

Usage:
    python ingest.py                    # Ingest all documents in data/
    python ingest.py --dir custom_dir   # Ingest from a different directory
    python ingest.py --file myfile.pdf  # Add a single file
    python ingest.py --status           # Show current index status
    python ingest.py --clear            # Clear the existing index
"""

import argparse
import sys
import io
import logging
from pathlib import Path

# Force UTF-8 output on Windows to handle emoji in log messages
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import config
from utils.document_loader import ingest_documents, get_index_status
from rag.vector_store import VectorStore

logger = logging.getLogger("ingest")


def print_banner():
    print("\n" + "=" * 60)
    print("  AI College Admission Agent — Knowledge Base Ingestion")
    print("=" * 60)


def print_status():
    """Print current index status."""
    status = get_index_status()
    print("\n📊 Knowledge Base Status")
    print("-" * 40)
    if not status["exists"]:
        print("  ❌ No index found. Run 'python ingest.py' to build one.")
        return

    print(f"  ✅ Index found")
    print(f"  📄 Documents : {status['num_documents']}")
    print(f"  🧩 Chunks    : {status['num_chunks']}")
    print(f"  🧬 Embed dim : {status['dimension']}")
    print(f"  🕐 Built at  : {status['built_at']}")
    if status.get("documents"):
        print(f"\n  Indexed documents:")
        for doc in status["documents"]:
            print(f"    • {doc}")


def progress_callback(step: str, pct: float):
    """CLI progress display."""
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:3.0f}%  {step:<45}", end="", flush=True)
    if pct >= 100:
        print()  # newline after completion


def main():
    parser = argparse.ArgumentParser(
        description="Build/rebuild the knowledge base for the AI College Admission Agent"
    )
    parser.add_argument("--dir", default=None, help="Directory containing documents")
    parser.add_argument("--file", default=None, help="Single file to add to the index")
    parser.add_argument("--status", action="store_true", help="Show current index status")
    parser.add_argument("--clear", action="store_true", help="Clear the existing index")
    args = parser.parse_args()

    print_banner()

    # Status check only
    if args.status:
        print_status()
        return 0

    # Clear index
    if args.clear:
        vs = VectorStore(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
            info_path=config.INDEX_INFO_PATH,
        )
        vs.clear()
        print("\n  🗑️  Index cleared.")
        return 0

    # Validate data directory
    data_dir = args.dir or str(config.DATA_DIR)
    if not Path(data_dir).exists():
        print(f"\n  ⚠️  Data directory not found: {data_dir}")
        print(f"  Create the directory and add your PDF/DOCX/TXT documents.")
        return 1

    # Check for documents
    files = [
        f for f in Path(data_dir).iterdir()
        if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".docx"}
    ]
    if not files and not args.file:
        print(f"\n  ⚠️  No supported documents found in: {data_dir}")
        print(f"  Add PDF, DOCX, or TXT files and run again.")
        return 1

    print(f"\n  📁 Source directory : {data_dir}")
    if files:
        print(f"  📄 Documents found  : {len(files)}")
        for f in files:
            print(f"    • {f.name}")

    extra_files = [args.file] if args.file else None
    print("\n  Starting ingestion pipeline...\n")

    result = ingest_documents(
        data_dir=data_dir,
        extra_files=extra_files,
        progress_callback=progress_callback,
    )

    print("\n")
    print("-" * 60)

    if result["success"]:
        print("  ✅ Knowledge base built successfully!")
        print(f"  📄 Documents indexed : {result['num_documents']}")
        print(f"  🧩 Chunks created   : {result['num_chunks']}")
        print(f"\n  Run 'streamlit run app.py' to start the application.")
    else:
        print("  ❌ Ingestion failed!")
        for err in result["errors"]:
            print(f"    • {err}")
        return 1

    if result.get("errors"):
        print(f"\n  ⚠️  Warnings:")
        for err in result["errors"]:
            print(f"    • {err}")

    print("-" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
