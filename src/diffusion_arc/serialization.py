"""NVARC-compatible and provider-neutral serialization for ARC tasks.

The NVARC serializer intentionally produces the exact chat-shaped prefix used
by the frozen checkpoint. The provider-neutral serializer produces a prompt
that can be pasted into a general-purpose LLM. Test outputs are never read or
included by either serializer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _validate_grid(grid: Any, *, name: str) -> list[list[int]]:
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)):
        raise TypeError(f"{name} must be a rectangular sequence of rows")
    if not 1 <= len(grid) <= 30:
        raise ValueError(f"{name} height must be in [1, 30]")

    rows: list[list[int]] = []
    width: int | None = None
    for row_index, row in enumerate(grid):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name}[{row_index}] must be a sequence")
        values = list(row)
        if width is None:
            width = len(values)
            if not 1 <= width <= 30:
                raise ValueError(f"{name} width must be in [1, 30]")
        elif len(values) != width:
            raise ValueError(f"{name} must be rectangular")

        checked_row: list[int] = []
        for column_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name}[{row_index}][{column_index}] must be an integer"
                )
            if not 0 <= value <= 9:
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] must be in [0, 9]"
                )
            checked_row.append(value)
        rows.append(checked_row)

    return rows


def serialize_grid(grid: Any, *, name: str = "grid") -> str:
    """Serialize an ARC grid using one digit per cell and one row per line."""

    rows = _validate_grid(grid, name=name)
    return "\n".join("".join(str(value) for value in row) for row in rows)


def _serialize_llm_grid(grid: Any, *, name: str, space_separated: bool) -> str:
    if not isinstance(space_separated, bool):
        raise TypeError("space_separated must be a boolean")
    separator = " " if space_separated else ""
    rows = _validate_grid(grid, name=name)
    return "\n".join(separator.join(str(value) for value in row) for row in rows)


def serialize_arc_task_for_llm(
    task: Mapping[str, Any],
    *,
    space_separated: bool = False,
) -> str:
    """Serialize all demonstrations and test inputs as a copy-ready LLM prompt.

    The prompt uses plain-text grid rows and requests JSON-only predictions.
    ``space_separated`` inserts one ASCII space between adjacent cells. Any
    ``output`` fields on test examples are deliberately ignored.
    """

    if not isinstance(task, Mapping):
        raise TypeError("task must be a mapping")
    train = task.get("train")
    test = task.get("test")
    if not isinstance(train, Sequence) or isinstance(train, (str, bytes)):
        raise TypeError("task['train'] must be a sequence")
    if not isinstance(test, Sequence) or isinstance(test, (str, bytes)):
        raise TypeError("task['test'] must be a sequence")
    if not test:
        raise ValueError("task['test'] must contain at least one example")
    if not isinstance(space_separated, bool):
        raise TypeError("space_separated must be a boolean")

    row_format = (
        "each line is one row and cells are separated by one space"
        if space_separated
        else "each line is one row and each character is one cell"
    )
    if len(test) == 1:
        response_format = "a single grid represented as a list of row lists"
    else:
        response_format = (
            "a list of grids in test-input order, where each grid is a list of row lists"
        )

    sections = [
        "\n".join(
            [
                "You are solving one ARC-AGI-2 task.",
                "Infer the transformation rule from the training examples and apply it "
                "to every test input.",
                "Each cell is an integer color from 0 through 9. Grid dimensions may change.",
                f"The grids below are plain text: {row_format}.",
                f"Return only valid JSON: {response_format}.",
                "Do not include explanations, Markdown, or code fences.",
            ]
        )
    ]

    for example_index, example in enumerate(train):
        if not isinstance(example, Mapping):
            raise TypeError(f"task['train'][{example_index}] must be a mapping")
        if "input" not in example or "output" not in example:
            raise ValueError(
                f"task['train'][{example_index}] must contain input and output"
            )
        input_grid = _serialize_llm_grid(
            example["input"],
            name=f"task['train'][{example_index}]['input']",
            space_separated=space_separated,
        )
        output_grid = _serialize_llm_grid(
            example["output"],
            name=f"task['train'][{example_index}]['output']",
            space_separated=space_separated,
        )
        sections.append(
            f"Training example {example_index + 1} input:\n{input_grid}\n\n"
            f"Training example {example_index + 1} output:\n{output_grid}"
        )

    for test_index, example in enumerate(test):
        if not isinstance(example, Mapping):
            raise TypeError(f"task['test'][{test_index}] must be a mapping")
        if "input" not in example:
            raise ValueError(f"task['test'][{test_index}] must contain input")
        input_grid = _serialize_llm_grid(
            example["input"],
            name=f"task['test'][{test_index}]['input']",
            space_separated=space_separated,
        )
        sections.append(f"Test input {test_index + 1}:\n{input_grid}")

    return "\n\n".join(sections)


def _user_turn(grid: Any, *, name: str) -> str:
    return f"{_IM_START}user\n{serialize_grid(grid, name=name)}{_IM_END}"


def _assistant_turn(grid: Any, *, name: str) -> str:
    return f"{_IM_START}assistant\n{serialize_grid(grid, name=name)}{_IM_END}"


def serialize_arc_task(task: Mapping[str, Any], test_index: int = 0) -> str:
    """Serialize demonstrations and one test input for frozen NVARC.

    Args:
        task: ARC challenge mapping containing ``train`` and ``test`` lists.
        test_index: Index of the test input to condition on.

    Returns:
        NVARC's exact prompt prefix.  It ends immediately after the assistant
        header for the selected test input.

    Raises:
        TypeError or ValueError: If the ARC task violates the expected schema.

    Any ``output`` field present in a test example is deliberately ignored.
    This makes leakage prevention structural rather than dependent on callers
    removing ground-truth test outputs.
    """

    if not isinstance(task, Mapping):
        raise TypeError("task must be a mapping")
    train = task.get("train")
    test = task.get("test")
    if not isinstance(train, Sequence) or isinstance(train, (str, bytes)):
        raise TypeError("task['train'] must be a sequence")
    if not isinstance(test, Sequence) or isinstance(test, (str, bytes)):
        raise TypeError("task['test'] must be a sequence")
    if isinstance(test_index, bool) or not isinstance(test_index, int):
        raise TypeError("test_index must be an integer")
    if not 0 <= test_index < len(test):
        raise IndexError(
            f"test_index {test_index} is outside task['test'] of length {len(test)}"
        )

    parts: list[str] = []
    for example_index, example in enumerate(train):
        if not isinstance(example, Mapping):
            raise TypeError(f"task['train'][{example_index}] must be a mapping")
        if "input" not in example or "output" not in example:
            raise ValueError(
                f"task['train'][{example_index}] must contain input and output"
            )
        parts.append(
            _user_turn(
                example["input"],
                name=f"task['train'][{example_index}]['input']",
            )
        )
        parts.append(
            _assistant_turn(
                example["output"],
                name=f"task['train'][{example_index}]['output']",
            )
        )

    selected_test = test[test_index]
    if not isinstance(selected_test, Mapping):
        raise TypeError(f"task['test'][{test_index}] must be a mapping")
    if "input" not in selected_test:
        raise ValueError(f"task['test'][{test_index}] must contain input")
    parts.append(
        _user_turn(
            selected_test["input"],
            name=f"task['test'][{test_index}]['input']",
        )
    )
    parts.append(f"{_IM_START}assistant\n")
    return "".join(parts)
