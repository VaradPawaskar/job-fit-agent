"""Tiny evaluation harness. Runs the real agent (needs an API key) on a few cases with known answers.

    python -m eval.run_eval

Each case lists requirements that must appear in the report with an acceptable status. This checks the
agent's judgement against the sample CV, so a prompt or model change that breaks it shows up as a FAIL.
"""

import json
import sys
from pathlib import Path

from app.agent import analyze

ROOT = Path(__file__).resolve().parent.parent


def load_cases(path: Path = ROOT / "eval" / "cases.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_case(case: dict, report) -> list[str]:
    """Return a list of failure messages (empty list = pass)."""
    failures = []
    for needle, allowed in case["expect"].items():
        found = [m for m in report.matches if needle in m.requirement.lower()]
        if not found:
            failures.append(f"'{needle}': requirement not found in report")
        elif not any(m.status in allowed for m in found):
            failures.append(f"'{needle}': got {[m.status for m in found]}, expected one of {allowed}")
    return failures


def main() -> int:
    cv_text = (ROOT / "samples" / "sample_cv.txt").read_text()
    failed = 0
    for case in load_cases():
        job_text = case.get("job_text") or (ROOT / case["job_file"]).read_text()
        result = analyze(cv_text, job_text=job_text)
        failures = check_case(case, result.report)
        status = "PASS" if not failures else "FAIL"
        failed += bool(failures)
        print(f"{status}  {case['id']}  score={result.score}  tool_calls={len(result.steps)}")
        for f in failures:
            print(f"      - {f}")
    print(f"\n{len(load_cases()) - failed}/{len(load_cases())} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
