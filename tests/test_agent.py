"""Agent tests use a scripted fake model, so they run offline and cost nothing."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.agent import AnalysisError, analyze
from app.schemas import RequirementMatch, compute_score


class FakeToolModel(GenericFakeChatModel):
    """Replays scripted AIMessages. bind_tools is a no-op because the script already contains tool calls."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


REPORT = {
    "job_title": "Working Student AI Engineer",
    "company": "Example GmbH",
    "summary": "Good fit on Python and Docker, gap on Kubernetes.",
    "matches": [
        {"requirement": "Python", "importance": "must_have", "status": "match", "evidence": "Python in skills"},
        {"requirement": "Docker", "importance": "must_have", "status": "match", "evidence": "Docker on AWS EC2"},
        {"requirement": "Kubernetes", "importance": "nice_to_have", "status": "missing", "evidence": ""},
    ],
    "top_gaps": ["Kubernetes"],
    "cv_suggestions": ["Move the deployed project above coursework projects."],
}


def test_agent_runs_tools_then_returns_report(sample_cv):
    model = FakeToolModel(
        messages=iter(
            [
                call("get_cv_overview", {}, "c1"),
                call("search_cv", {"query": "Docker"}, "c2"),
                call("AgentReport", REPORT, "c3"),
            ]
        )
    )
    result = analyze(sample_cv, job_text="We need Python, Docker. Kubernetes is a plus.", model=model)

    assert result.report.job_title == "Working Student AI Engineer"
    assert [s.tool for s in result.steps] == ["get_cv_overview", "search_cv"]
    assert result.steps[1].args == {"query": "Docker"}
    # must-haves 2+2 full credit, nice-to-have 1 with no credit -> 4/5
    assert result.score == 80


def test_batched_search_needs_only_four_model_calls(sample_cv):
    """overview -> one batched search -> final report. Fewer round trips = faster."""
    model = FakeToolModel(
        messages=iter(
            [
                call("get_cv_overview", {}, "c1"),
                call("search_cv_many", {"queries": ["Python", "Docker", "Kubernetes"]}, "c2"),
                call("AgentReport", REPORT, "c3"),
            ]
        )
    )
    result = analyze(sample_cv, job_text="Python, Docker, Kubernetes", model=model)
    assert [s.tool for s in result.steps] == ["get_cv_overview", "search_cv_many"]
    assert result.steps[1].args["queries"] == ["Python", "Docker", "Kubernetes"]


def test_model_kwargs_setting(monkeypatch):
    from app.config import get_settings

    assert get_settings().model_kwargs == {}
    monkeypatch.setenv("MODEL_KWARGS", '{"max_output_tokens": 2048}')
    assert get_settings().model_kwargs == {"max_output_tokens": 2048}
    monkeypatch.setenv("MODEL_KWARGS", "not json")
    with pytest.raises(ValueError):
        get_settings()
    monkeypatch.setenv("MODEL_KWARGS", "[1, 2]")
    with pytest.raises(ValueError):
        get_settings()


def test_empty_matches_becomes_a_user_facing_error(sample_cv):
    empty = {**REPORT, "matches": [], "summary": "The posting could not be read."}
    model = FakeToolModel(messages=iter([call("AgentReport", empty, "c1")]))
    with pytest.raises(AnalysisError, match="could not be read"):
        analyze(sample_cv, job_url="https://jobs.example.com/1", model=model)


def test_requires_a_posting(sample_cv):
    with pytest.raises(AnalysisError):
        analyze(sample_cv)


def test_step_limit_stops_a_looping_agent(sample_cv, monkeypatch):
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "6")

    def endless():
        i = 0
        while True:
            i += 1
            yield call("search_cv", {"query": "docker"}, f"c{i}")

    model = FakeToolModel(messages=endless())
    with pytest.raises(AnalysisError, match="step limit"):
        analyze(sample_cv, job_text="anything", model=model)


# --- scoring -----------------------------------------------------------------------------------


def m(importance, status):
    return RequirementMatch(requirement="x", importance=importance, status=status, evidence="")


def test_score_weights_must_haves_and_partials():
    assert compute_score([]) == 0
    assert compute_score([m("must_have", "match"), m("must_have", "match")]) == 100
    assert compute_score([m("must_have", "missing"), m("nice_to_have", "match")]) == 33
    assert compute_score([m("must_have", "partial")]) == 50
