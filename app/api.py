"""FastAPI service exposing the agent."""

import hmac
import logging

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from .agent import AnalysisError, ConfigurationError, analyze
from .config import get_settings
from .cv import CVParseError, extract_cv_text
from .schemas import AnalysisResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Job Fit Agent", version="0.1.0")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(require_api_key)])
def analyze_endpoint(
    cv_file: UploadFile | None = File(default=None, description="CV as PDF, .txt or .md"),
    cv_text: str | None = Form(default=None, description="CV as plain text (alternative to cv_file)"),
    job_url: str | None = Form(default=None, description="Public URL of the job posting"),
    job_text: str | None = Form(default=None, description="Pasted posting text. Used instead of job_url if both are set"),
) -> AnalysisResponse:
    settings = get_settings()

    if bool(cv_file) == bool(cv_text):
        raise HTTPException(422, "Provide exactly one of cv_file or cv_text.")
    if not (job_text or job_url):
        raise HTTPException(422, "Provide job_url or job_text.")

    try:
        if cv_file:
            data = cv_file.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, f"File is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.")
            text = extract_cv_text(cv_file.filename or "", data)
        else:
            text = extract_cv_text("cv.txt", cv_text.encode("utf-8"))
    except CVParseError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        return analyze(text, job_text=job_text, job_url=job_url)
    except ConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc
    except AnalysisError as exc:
        raise HTTPException(422, str(exc)) from exc
