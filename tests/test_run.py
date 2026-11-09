import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Both ways of starting the program must behave the same.
ENTRY_POINTS = {
    "run.py": [sys.executable, "run.py"],
    "task": [str(Path(sys.executable).parent / "task")],
}


@pytest.fixture(params=ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys())
def run(request):
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*request.param, *args],
            cwd=ROOT, capture_output=True, text=True,
        )

    return _run


def test_default_task_runs_and_exits_zero(run):
    result = run()
    assert result.returncode == 0
    assert "Running task 1 from week 1." in result.stdout


def test_missing_task_prints_one_line_error_and_exits_one(run):
    result = run("-w", "99", "-t", "100")
    assert result.returncode == 1
    assert result.stderr.strip() == (
        "error: Task 100 does not exist in week 99! "
        "Run with --list to see the available tasks."
    )


def test_invalid_week_is_usage_error(run):
    result = run("-w", "0")
    assert result.returncode == 2
    assert "must be 1 or more" in result.stderr
