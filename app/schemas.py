"""Data models: what the LLM must produce, and what the API returns."""

from typing import Literal

from pydantic import BaseModel, Field

Importance = Literal["must_have", "nice_to_have"]
Status = Literal["match", "partial", "missing"]


class RequirementMatch(BaseModel):
    requirement: str = Field(description="One concrete requirement from the job posting, e.g. 'Docker'.")
    importance: Importance = Field(description="must_have if the posting treats it as required, else nice_to_have.")
    status: Status = Field(description="match = clear CV evidence, partial = related evidence, missing = none.")
    evidence: str = Field(
        description="Short paraphrase of the CV evidence. Empty string if status is missing. Never invent."
    )


class AgentReport(BaseModel):
    """The structured answer the agent returns when it is done."""

    job_title: str = Field(description="Job title from the posting, or 'Unknown' if not stated.")
    company: str | None = Field(default=None, description="Company name if stated.")
    summary: str = Field(description="2-3 sentence honest verdict on the fit.")
    matches: list[RequirementMatch] = Field(description="6-12 requirements, each checked against the CV.")
    top_gaps: list[str] = Field(description="The most important missing or weak areas, most important first.")
    cv_suggestions: list[str] = Field(
        description="Truthful edits: reorder, reword or emphasise what the CV already shows. Never fabricate."
    )


class ToolStep(BaseModel):
    tool: str
    args: dict


class AnalysisResponse(BaseModel):
    score: int = Field(description="0-100, computed in code from the matches (not chosen by the LLM).")
    report: AgentReport
    steps: list[ToolStep] = Field(description="Tool calls the agent made, for transparency.")
    model: str


# --- scoring -----------------------------------------------------------------------------------

_WEIGHT = {"must_have": 2.0, "nice_to_have": 1.0}
_CREDIT = {"match": 1.0, "partial": 0.5, "missing": 0.0}


def compute_score(matches: list[RequirementMatch]) -> int:
    """Weighted coverage: must-haves count double, partial matches count half."""
    total = sum(_WEIGHT[m.importance] for m in matches)
    if total == 0:
        return 0
    got = sum(_WEIGHT[m.importance] * _CREDIT[m.status] for m in matches)
    return round(100 * got / total)
