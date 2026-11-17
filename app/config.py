"""Runtime settings, read from environment variables on every call (easy to override in tests)."""

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    model: str
    recursion_limit: int
    max_cv_chars: int
    max_posting_chars: int
    max_upload_bytes: int
    fetch_timeout: float
    allow_private_urls: bool
    api_key: str | None
    model_kwargs: dict = field(default_factory=dict)


def _model_kwargs() -> dict:
    """Extra model options as JSON, e.g. MODEL_KWARGS='{"max_output_tokens": 2048}'."""
    raw = os.getenv("MODEL_KWARGS", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("MODEL_KWARGS must be valid JSON, for example {\"max_output_tokens\": 2048}") from exc
    if not isinstance(value, dict):
        raise ValueError("MODEL_KWARGS must be a JSON object")
    return value


def get_settings() -> Settings:
    return Settings(
        model_kwargs=_model_kwargs(),
        # "<provider>:<model>", e.g. "google_genai:gemini-3.5-flash-lite", "anthropic:<model>", "openai:<model>"
        model=os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite"),
        # Hard cap on agent steps so a confused model cannot loop forever.
        recursion_limit=int(os.getenv("AGENT_RECURSION_LIMIT", "25")),
        max_cv_chars=int(os.getenv("MAX_CV_CHARS", "20000")),
        max_posting_chars=int(os.getenv("MAX_POSTING_CHARS", "12000")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024,
        fetch_timeout=float(os.getenv("FETCH_TIMEOUT_SECONDS", "10")),
        # Only for local testing. Leave false: it blocks requests to private/internal addresses.
        allow_private_urls=os.getenv("ALLOW_PRIVATE_URLS", "false").lower() == "true",
        # If set, /analyze requires an "X-API-Key" header with this value.
        api_key=os.getenv("API_KEY") or None,
    )
