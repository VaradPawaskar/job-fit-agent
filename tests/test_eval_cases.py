"""The live eval needs an API key, but its case file and checker must stay valid."""

from eval.run_eval import ROOT, check_case, load_cases
from app.schemas import AgentReport, RequirementMatch


def test_case_files_exist_and_are_well_formed():
    cases = load_cases()
    assert len(cases) >= 3
    for case in cases:
        assert case["expect"]
        assert case.get("job_text") or (ROOT / case["job_file"]).exists()


def test_checker_flags_wrong_and_missing_requirements():
    case = {"expect": {"docker": ["match"], "kubernetes": ["missing"]}}
    report = AgentReport(
        job_title="x",
        summary="x",
        matches=[RequirementMatch(requirement="Docker", importance="must_have", status="missing", evidence="")],
        top_gaps=[],
        cv_suggestions=[],
    )
    failures = check_case(case, report)
    assert len(failures) == 2
