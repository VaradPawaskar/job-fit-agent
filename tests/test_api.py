import pytest
from fastapi.testclient import TestClient

from app import api
from app.agent import AnalysisError, ConfigurationError
from app.schemas import AgentReport, AnalysisResponse

client = TestClient(api.app)

CV = "Alex Example\nPython, Docker, PyTorch. Built and deployed services on AWS for two years. M.Sc. in AI."


def fake_response() -> AnalysisResponse:
    report = AgentReport(job_title="Dev", summary="ok", matches=[], top_gaps=[], cv_suggestions=[])
    return AnalysisResponse(score=70, report=report, steps=[], model="fake")


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_with_text_cv(monkeypatch):
    seen = {}

    def fake_analyze(cv_text, job_text=None, job_url=None):
        seen.update(cv_text=cv_text, job_text=job_text, job_url=job_url)
        return fake_response()

    monkeypatch.setattr(api, "analyze", fake_analyze)
    r = client.post("/analyze", data={"cv_text": CV, "job_text": "Python dev wanted"})
    assert r.status_code == 200 and r.json()["score"] == 70
    assert seen["job_text"] == "Python dev wanted" and "Docker" in seen["cv_text"]


def test_analyze_with_uploaded_txt(monkeypatch):
    monkeypatch.setattr(api, "analyze", lambda *a, **k: fake_response())
    r = client.post("/analyze", files={"cv_file": ("cv.txt", CV.encode())}, data={"job_url": "https://x.example/1"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "files,data,status",
    [
        (None, {"job_text": "x"}, 422),  # no CV
        (None, {"cv_text": CV}, 422),  # no posting
        ({"cv_file": ("cv.txt", CV.encode())}, {"cv_text": CV, "job_text": "x"}, 422),  # both CV inputs
        ({"cv_file": ("cv.docx", b"whatever")}, {"job_text": "x"}, 422),  # unsupported type
        ({"cv_file": ("cv.pdf", b"not a pdf")}, {"job_text": "x"}, 422),  # corrupt pdf
        (None, {"cv_text": "too short", "job_text": "x"}, 422),  # nothing usable
    ],
)
def test_input_validation(files, data, status):
    assert client.post("/analyze", files=files, data=data).status_code == status


def test_upload_size_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    big = b"a" * (1024 * 1024 + 10)
    r = client.post("/analyze", files={"cv_file": ("cv.txt", big)}, data={"job_text": "x"})
    assert r.status_code == 413


def test_agent_errors_map_to_http_codes(monkeypatch):
    def boom(exc):
        def _raise(*a, **k):
            raise exc

        return _raise

    monkeypatch.setattr(api, "analyze", boom(ConfigurationError("no key")))
    assert client.post("/analyze", data={"cv_text": CV, "job_text": "x"}).status_code == 503
    monkeypatch.setattr(api, "analyze", boom(AnalysisError("step limit")))
    r = client.post("/analyze", data={"cv_text": CV, "job_text": "x"})
    assert r.status_code == 422 and r.json()["detail"] == "step limit"


def test_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setattr(api, "analyze", lambda *a, **k: fake_response())
    data = {"cv_text": CV, "job_text": "x"}
    assert client.post("/analyze", data=data).status_code == 401
    assert client.post("/analyze", data=data, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/analyze", data=data, headers={"X-API-Key": "secret"}).status_code == 200
