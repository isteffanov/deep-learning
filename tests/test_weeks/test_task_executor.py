import sys
import types

import pytest

from weeks import task_executor


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], (1, 1)),
        (["-w", "5"], (5, 1)),
        (["-t", "12"], (1, 12)),
        (["--week", "2", "--task", "12"], (2, 12)),
    ],
)
def test_parse_command_line_arguments(argv, expected):
    assert task_executor.parse_command_line_arguments(argv) == expected


def test_parse_reads_sys_argv_when_no_argv_given(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "-w", "3"])
    assert task_executor.parse_command_line_arguments() == (3, 1)


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_parse_rejects_non_positive_or_non_numbers(bad):
    with pytest.raises(SystemExit) as exc:
        task_executor.parse_command_line_arguments(["-w", bad])
    assert exc.value.code == 2


def test_list_prints_available_tasks_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        task_executor.parse_command_line_arguments(["--list"])
    assert exc.value.code == 0
    assert "uv run task -w 1 -t 1" in capsys.readouterr().out


def test_available_tasks_finds_week01_task01():
    assert (1, 1) in task_executor.available_tasks()


def test_main_runs_week1_task1_by_default(capsys):
    assert task_executor.main([]) == 0
    assert "Running task 1 from week 1." in capsys.readouterr().out


def test_missing_task_raises_value_error():
    with pytest.raises(ValueError, match="Task 100 does not exist in week 99!"):
        task_executor.main(["-w", "99", "-t", "100"])


def test_missing_dependency_inside_task_is_not_hidden(monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(task_executor.importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="torch"):
        task_executor.load_task(1, 1)


def test_task_without_main_raises_value_error(monkeypatch):
    monkeypatch.setattr(
        task_executor.importlib, "import_module", lambda name: types.ModuleType(name)
    )
    with pytest.raises(ValueError, match="has no main"):
        task_executor.load_task(1, 1)
