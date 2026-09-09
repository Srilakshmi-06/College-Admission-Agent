"""
Input validation utilities for file uploads and user queries.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum file size for uploads: 50 MB
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# Max query length (characters)
MAX_QUERY_LENGTH = 2000


def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate a Streamlit UploadedFile before saving.

    Args:
        uploaded_file: st.UploadedFile object.

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if uploaded_file is None:
        return False, "No file provided."

    name = uploaded_file.name
    ext = Path(name).suffix.lower()

    # Check extension
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"File type '{ext}' is not supported. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Check size
    size = uploaded_file.size if hasattr(uploaded_file, "size") else 0
    if size > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"File '{name}' is too large ({size / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB",
        )

    # Check filename for path traversal
    safe_name = Path(name).name
    if safe_name != name or ".." in name or "/" in name or "\\" in name:
        return False, f"Invalid filename: '{name}'"

    return True, "File is valid."


def sanitize_query(query: str) -> tuple[bool, str]:
    """
    Sanitize and validate a user query string.

    Args:
        query: Raw user input.

    Returns:
        Tuple of (is_valid: bool, sanitized_query: str)
        If invalid, the second element is an error message.
    """
    if not query:
        return False, "Please enter a question."

    # Strip leading/trailing whitespace
    query = query.strip()

    if len(query) < 2:
        return False, "Question is too short."

    if len(query) > MAX_QUERY_LENGTH:
        return (
            False,
            f"Question is too long (max {MAX_QUERY_LENGTH} characters). "
            "Please be more concise.",
        )

    # Remove control characters except newlines
    query = re.sub(r"[^\S\n]+", " ", query)
    query = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", query)

    return True, query


def validate_file_path(file_path: str) -> tuple[bool, str]:
    """
    Validate that a file path is safe and within expected bounds.

    Args:
        file_path: File path string.

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    path = Path(file_path)

    # Check for path traversal
    try:
        resolved = path.resolve()
    except Exception as e:
        return False, f"Invalid path: {e}"

    if not path.exists():
        return False, f"File not found: {file_path}"

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"Unsupported file type '{ext}'. Supported: {ALLOWED_EXTENSIONS}",
        )

    return True, "Path is valid."
