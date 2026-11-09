"""Run one course task: ``uv run task -w <week> -t <task>``"""

import argparse
import importlib
import pkgutil
import sys
from collections.abc import Callable, Sequence

import weeks


def available_tasks() -> list[tuple[int, int]]:
    """Find every weeks/weekNN/taskNN.py on disk, sorted."""
    found: list[tuple[int, int]] = []
    for week in pkgutil.iter_modules(weeks.__path__):
        if not (week.ispkg and week.name.startswith("week")):
            continue
        package = importlib.import_module(f"weeks.{week.name}")
        for task in pkgutil.iter_modules(package.__path__):
            if task.name.startswith("task"):
                found.append((int(week.name[4:]), int(task.name[4:])))
    return sorted(found)


class ListTasks(argparse.Action):
    """Like --help: print something, then exit during parsing."""

    def __call__(self, parser, namespace, values, option_string=None):
        for week, task in available_tasks():
            print(f"uv run task -w {week} -t {task}")
        parser.exit()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def parse_command_line_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[int, int]:
    parser = argparse.ArgumentParser(
        prog="uv run task",
        description="Run one task from the course.",
        epilog="example: uv run task -w 1 -t 1",
    )
    parser.add_argument("-w", "--week", type=positive_int, default=1,
                        help="week number (default: 1)")
    parser.add_argument("-t", "--task", type=positive_int, default=1,
                        help="task number (default: 1)")
    parser.add_argument("-l", "--list", action=ListTasks, nargs=0,
                        default=argparse.SUPPRESS,
                        help="list the available tasks and exit")
    args = parser.parse_args(argv)
    return (args.week, args.task)


def load_task(week: int, task: int) -> Callable[[], int]:
    name = f"weeks.week{week:02d}.task{task:02d}"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        # Only "the task file is missing" becomes a ValueError. A missing
        # dependency inside the task (e.g. torch) is re-raised unchanged.
        if e.name is None or not (name == e.name or name.startswith(e.name + ".")):
            raise
        raise ValueError(
            f"Task {task:02d} does not exist in week {week:02d}! "
            "Run with --list to see the available tasks."
        ) from e
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise ValueError(f"{name} has no main() function.")
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    week, task = parse_command_line_arguments(argv)
    return load_task(week, task)()


def cli() -> None:
    """Entry point for `uv run task` and run.py: errors become exit codes."""
    try:
        sys.exit(main())
    except ValueError as e:
        sys.exit(f"error: {e}")
