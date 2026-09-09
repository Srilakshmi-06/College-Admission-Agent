"""
Central configuration management for the AI College Admission Agent.
All settings are loaded from environment variables with sensible defaults.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress TensorFlow / oneDNN noise that appears when sentence-transformers
# loads its torch backend. These warnings are irrelevant to this application.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Load .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local").lower()  # "local" | "ibm"

# Ollama (local)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# IBM Granite / watsonx.ai
IBM_CLOUD_API_KEY: str = os.getenv("IBM_CLOUD_API_KEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
GRANITE_MODEL: str = os.getenv("GRANITE_MODEL", "ibm-granite/granite-3.2-8b-instruct")

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# RAG Configuration
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# FAISS index paths
FAISS_INDEX_PATH: str = str(VECTORSTORE_DIR / "faiss_index")
METADATA_PATH: str = str(VECTORSTORE_DIR / "metadata.json")
INDEX_INFO_PATH: str = str(VECTORSTORE_DIR / "index_info.json")

# ---------------------------------------------------------------------------
# Generation Configuration
# ---------------------------------------------------------------------------
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("college_agent")


# ---------------------------------------------------------------------------
# Runtime validation helpers
# ---------------------------------------------------------------------------
def is_ibm_configured() -> bool:
    """Return True if IBM watsonx credentials are present."""
    return bool(IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID)


def get_active_provider() -> str:
    """Return the effective LLM provider based on config and available credentials."""
    if LLM_PROVIDER == "ibm":
        if is_ibm_configured():
            return "ibm"
        logger.warning(
            "LLM_PROVIDER=ibm but IBM credentials not found. Falling back to local."
        )
        return "local"
    return "local"


def validate_config() -> dict[str, bool]:
    """Validate and return the status of each configuration section."""
    return {
        "data_dir_exists": DATA_DIR.exists(),
        "vectorstore_dir_exists": VECTORSTORE_DIR.exists(),
        "faiss_index_exists": Path(FAISS_INDEX_PATH + ".faiss").exists(),
        "ibm_configured": is_ibm_configured(),
        "provider": get_active_provider(),
    }
