import ipaddress
import socket

import pytest

from app import tools
from app.tools import UnsafeURLError, build_cv_tools, check_url, fetch_job_posting, html_to_text, search_cv_text


# --- CV search ---------------------------------------------------------------------------------


def test_search_finds_exact_skill(sample_cv):
    hits = search_cv_text(sample_cv, "Docker")
    assert hits and "docker" in hits[0][1].lower()


def test_search_handles_plurals_and_aliases():
    cv = "Built CI/CD pipelines for a bank.\nDeployed on Kubernetes."
    assert search_cv_text(cv, "pipeline")
    assert search_cv_text(cv, "k8s")


def test_search_returns_nothing_for_missing_skill(sample_cv):
    assert search_cv_text(sample_cv, "Kubernetes") == []
    assert search_cv_text(sample_cv, "Power BI") == []


def test_search_tool_messages(sample_cv):
    _, _, search = build_cv_tools(sample_cv)
    assert "NO EVIDENCE FOUND" in search.invoke({"query": "Tableau"})
    assert "[1.00]" in search.invoke({"query": "SHAP"})


def test_batch_search_checks_every_query_in_one_call(sample_cv):
    _, many, _ = build_cv_tools(sample_cv)
    out = many.invoke({"queries": ["Docker", "Kubernetes", "SHAP"]})
    assert "## Docker" in out and "## Kubernetes" in out and "## SHAP" in out
    kube_block = out.split("## Kubernetes")[1].split("##")[0]
    assert "NO EVIDENCE FOUND" in kube_block
    docker_block = out.split("## Docker")[1].split("##")[0]
    assert "NO EVIDENCE" not in docker_block


def test_batch_search_rejects_empty_and_caps_size(sample_cv):
    _, many, _ = build_cv_tools(sample_cv)
    assert many.invoke({"queries": ["", "  "]}).startswith("ERROR:")
    out = many.invoke({"queries": [f"skill{i}" for i in range(50)]})
    assert out.count("## ") == 20


def test_overview_tool_returns_cv_start(sample_cv):
    overview, _, _ = build_cv_tools(sample_cv)
    assert overview.invoke({}).startswith("Alex Example")


# --- URL safety --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://127.0.0.1/admin",
        "http://localhost:8000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://[::1]/",
        "not a url",
    ],
)
def test_unsafe_urls_are_blocked(url):
    with pytest.raises(UnsafeURLError):
        check_url(url)


def test_fetch_tool_returns_error_string_not_exception():
    out = fetch_job_posting.invoke({"url": "http://127.0.0.1/secret"})
    assert out.startswith("ERROR:")


# --- fetching with a mocked network ------------------------------------------------------------


class FakeResponse:
    def __init__(self, body: bytes, status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
        self.is_redirect = "location" in {k.lower() for k in self.headers}
        self.is_permanent_redirect = False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass


@pytest.fixture
def public_dns(monkeypatch):
    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
        # IP literals resolve to themselves (like real DNS); named hosts resolve to a public address
        try:
            ipaddress.ip_address(host.strip("[]"))
            return real_getaddrinfo(host, port, *args, **kwargs)
        except ValueError:
            return [(2, 1, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def test_fetch_extracts_readable_text(monkeypatch, public_dns):
    html = "<html><head><script>evil()</script></head><body><nav>menu</nav><h1>ML Engineer</h1>" + (
        "<p>We need Python, Docker and PyTorch skills for this role.</p>" * 10
    ) + "</body></html>"
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(html.encode()))
    out = fetch_job_posting.invoke({"url": "https://jobs.example.com/1"})
    assert "ML Engineer" in out and "PyTorch" in out
    assert "evil()" not in out and "menu" not in out


def test_fetch_rechecks_redirect_target(monkeypatch, public_dns):
    redirect = FakeResponse(b"", 302, {"location": "http://127.0.0.1/internal", "content-type": "text/html"})
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: redirect)
    out = fetch_job_posting.invoke({"url": "https://jobs.example.com/redirect"})
    assert out.startswith("ERROR:") and "non-public" in out


def test_fetch_reports_empty_pages(monkeypatch, public_dns):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(b"<html><body>Loading...</body></html>"))
    out = fetch_job_posting.invoke({"url": "https://jobs.example.com/spa"})
    assert out.startswith("ERROR:") and "paste" in out


def test_html_to_text_collapses_whitespace():
    assert html_to_text("<p>a</p>\n\n\n<p>b</p>") == "a\nb"
