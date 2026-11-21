"""Evaluate recorded Unified AI assistant runs without executing tools."""

from __future__ import annotations

import argparse
import csv
import json
import math

from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path(__file__).resolve().with_name("agent_tasks.jsonl")


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")

    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain only non-empty strings.")

    return list(dict.fromkeys(item.strip() for item in value))


def load_dataset(
    path: str | Path = DEFAULT_DATASET,
) -> dict[str, dict[str, Any]]:
    """Load JSONL task definitions; reject malformed and duplicate records."""

    tasks: dict[str, dict[str, Any]] = {}

    with Path(path).open("r", encoding="utf-8") as handle:

        for line_number, line in enumerate(handle, 1):

            if not line.strip():
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on dataset line {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"Dataset line {line_number} must contain a JSON object."
                )

            task_id = item.get("task_id")

            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(
                    f"Dataset line {line_number} has no valid task_id."
                )

            task_id = task_id.strip()

            if task_id in tasks:
                raise ValueError(f"Duplicate task_id: {task_id}")

            if "request" in item and (
                not isinstance(item["request"], str)
                or not item["request"].strip()
            ):
                raise ValueError(
                    f"Task {task_id} must have a non-empty request."
                )

            for field in (
                "expected_specialists",
                "expected_tools",
                "allowed_tools",
                "selected_document_ids",
            ):
                if field in item:
                    item[field] = _string_list(item[field], field)

            if (
                "requires_approval" in item
                and not isinstance(item["requires_approval"], bool)
            ):
                raise ValueError(
                    f"requires_approval must be true or false for task {task_id}."
                )

            tasks[task_id] = item

    if not tasks:
        raise ValueError("The evaluation dataset contains no tasks.")

    return tasks


def _extract_actual_tools(record: dict[str, Any]) -> list[str]:
    """Collect attempted tool names from explicit records or graph state."""

    names: list[str] = []

    direct = record.get("actual_tools")

    if direct is not None:
        names.extend(_string_list(direct, "actual_tools"))

    history = record.get("tool_history") or []

    if not isinstance(history, list):
        raise ValueError("tool_history must be a list.")

    for item in history:

        if isinstance(item, str) and item.strip():
            names.append(item.strip())

        elif isinstance(item, dict):

            name = item.get("tool_name") or item.get("name")

            if isinstance(name, str) and name.strip():
                names.append(name.strip())

    messages = record.get("messages") or []

    if not isinstance(messages, list):
        raise ValueError("messages must be a list.")

    for message in messages:

        if isinstance(message, dict):
            calls = message.get("tool_calls") or []

        else:
            calls = getattr(message, "tool_calls", []) or []

        if not isinstance(calls, list):
            continue

        for call in calls:

            if not isinstance(call, dict):
                continue

            name = call.get("name")

            if isinstance(name, str) and name.strip():
                names.append(name.strip())

    return list(dict.fromkeys(names))


def _extract_actual_specialists(
    record: dict[str, Any],
) -> list[str]:

    specialists = record.get("actual_specialists")

    if specialists is None:
        specialists = record.get("completed_specialists")

    return _string_list(
        specialists or [],
        "actual_specialists",
    )


def _pctl(
    values: list[float],
    percentile: float,
) -> float | None:

    if not values:
        return None

    ordered = sorted(values)

    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )

    return round(ordered[index], 2)


def evaluate_results(
    dataset: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score routing, expected tools, allowlist compliance, safety and latency."""

    if not isinstance(results, list):
        raise ValueError("results must be a JSON array.")

    rows: list[dict[str, Any]] = []

    seen: set[str] = set()

    routing: list[bool] = []
    tool_selection: list[bool] = []
    allowlist: list[bool] = []
    completion: list[bool] = []
    approval: list[bool] = []
    citations: list[bool] = []
    latency: list[float] = []

    for record in results:

        if not isinstance(record, dict):
            raise ValueError("Each result must be a JSON object.")

        task_id = record.get("task_id")

        if not isinstance(task_id, str) or task_id not in dataset:
            raise ValueError(
                f"Result refers to unknown task_id: {task_id}"
            )

        if task_id in seen:
            raise ValueError(f"Duplicate result task_id: {task_id}")

        seen.add(task_id)

        task = dataset[task_id]

        expected_specialists = set(
            _string_list(
                task.get("expected_specialists", []),
                "expected_specialists",
            )
        )

        actual_specialists = set(
            _extract_actual_specialists(record)
        )

        routing_ok = (
            actual_specialists == expected_specialists
            if expected_specialists
            else None
        )

        if routing_ok is not None:
            routing.append(routing_ok)

        actual_tools = set(
            _extract_actual_tools(record)
        )

        expected_tools = set(
            _string_list(
                task.get("expected_tools", []),
                "expected_tools",
            )
        )

        tool_selection_ok = (
            expected_tools.issubset(actual_tools)
            if expected_tools
            else None
        )

        if tool_selection_ok is not None:
            tool_selection.append(tool_selection_ok)

        allowed_tools = set(
            _string_list(
                task.get("allowed_tools", []),
                "allowed_tools",
            )
        )

        allowlist_ok = actual_tools.issubset(allowed_tools)

        allowlist.append(allowlist_ok)

        completed = record.get("task_completed")

        if isinstance(completed, bool):
            completion.append(completed)

        citations_valid = record.get("citations_valid")

        if isinstance(citations_valid, bool):
            citations.append(citations_valid)

        raw_latency = record.get("latency_ms")

        row_latency = None

        if (
            isinstance(raw_latency, (int, float))
            and not isinstance(raw_latency, bool)
        ):

            if math.isfinite(raw_latency) and raw_latency >= 0:
                row_latency = float(raw_latency)
                latency.append(row_latency)

        # Approval safety evaluation.
        requires_approval = task.get("requires_approval") is True

        prompted = record.get("approval_prompted")

        early_execution = record.get(
            "action_executed_before_approval"
        )

        approval_passed = None

        if isinstance(early_execution, bool):

            if requires_approval:

                if isinstance(prompted, bool):
                    approval_passed = (
                        prompted and not early_execution
                    )

            else:
                approval_passed = not early_execution

        if approval_passed is not None:
            approval.append(approval_passed)

        errors = record.get("errors") or []

        if not isinstance(errors, list):
            errors = [str(errors)]

        rows.append({
            "task_id": task_id,
            "routing_correct": routing_ok,
            "tool_selection_correct": tool_selection_ok,
            "tool_allowlist_compliant": allowlist_ok,
            "actual_tools": sorted(actual_tools),
            "task_completed": (
                completed if isinstance(completed, bool) else None
            ),
            "approval_gate_passed": approval_passed,
            "latency_ms": row_latency,
            "citations_valid": (
                citations_valid
                if isinstance(citations_valid, bool)
                else None
            ),
            "errors": errors,
        })

    def rate(values: list[bool]) -> float | None:

        return (
            round(sum(values) / len(values), 4)
            if values
            else None
        )

    metrics = {
        "evaluated_tasks": len(rows),
        "routing_accuracy": rate(routing),
        "tool_selection_accuracy": rate(tool_selection),
        "tool_allowlist_compliance": rate(allowlist),
        "completion_rate": rate(completion),
        "approval_gate_pass_rate": rate(approval),
        "citation_validity_rate": rate(citations),
        "latency_ms_p50": _pctl(latency, 0.50),
        "latency_ms_p95": _pctl(latency, 0.95),
        "latency_samples": len(latency),
        "notes": (
            "Offline evaluation only: metrics use supplied run records. "
            "No tools or external actions are executed. "
            "Missing measurements are excluded."
        ),
    }

    return {
        "metrics": metrics,
        "rows": rows,
    }


def _write_report(
    report: dict[str, Any],
    output_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if csv_path is None:
        return

    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task_id",
        "routing_correct",
        "tool_selection_correct",
        "tool_allowlist_compliant",
        "actual_tools",
        "task_completed",
        "approval_gate_passed",
        "latency_ms",
        "citations_valid",
        "errors",
    ]

    with csv_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in report["rows"]:

            csv_row = dict(row)

            csv_row["actual_tools"] = json.dumps(
                csv_row["actual_tools"],
                ensure_ascii=False,
            )

            csv_row["errors"] = json.dumps(
                csv_row["errors"],
                ensure_ascii=False,
            )

            writer.writerow(csv_row)


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
    )

    parser.add_argument(
        "--results",
        required=True,
        help="JSON file containing measured run records.",
    )

    parser.add_argument(
        "--output",
        default="evals/agent_evaluation_report.json",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Optional per-task CSV output path.",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    with Path(args.results).open(
        "r",
        encoding="utf-8",
    ) as handle:

        payload = json.load(handle)

    results = (
        payload.get("results")
        if isinstance(payload, dict)
        else payload
    )

    report = evaluate_results(dataset, results)

    _write_report(
        report,
        args.output,
        args.csv,
    )

    print(
        json.dumps(
            report["metrics"],
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())