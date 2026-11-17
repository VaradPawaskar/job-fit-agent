"""CV loading: PDF / text -> clean text."""

import io
import re

from pypdf import PdfReader

from .config import get_settings


class CVParseError(ValueError):
    """The CV could not be turned into usable text."""


def _clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_cv_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(page.extract_text() or "") for page in reader.pages[:10]]
        except Exception as exc:  # pypdf raises many different error types on bad files
            raise CVParseError("Could not read this PDF. Is it corrupted or password protected?") from exc
        text = _clean("\n".join(pages))
    elif name.endswith((".txt", ".md")):
        text = _clean(data.decode("utf-8", errors="replace"))
    else:
        raise CVParseError("Unsupported file type. Upload a PDF, .txt or .md file.")

    if len(text) < 50:
        raise CVParseError("No readable text found. If this is a scanned PDF, export a text-based PDF instead.")
    return text[: get_settings().max_cv_chars]
