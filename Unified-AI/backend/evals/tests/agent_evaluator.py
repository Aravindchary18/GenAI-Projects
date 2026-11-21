"""Evaluate Unified AI assistant runs and optionally execute safe read-only tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time

from pathlib import Path
from typing import Any, Callable


DEFAULT_DATASET = Path(__file__).resolve().with_name("agent_tasks.jsonl")

# Only these tools are permitted during automatic read-only evaluation.
READ_ONLY_AUTOMATION_TOOLS = {
    "open_public_website",
}

READ_ONLY_SPECIALIST_TOOLS = {
    "list_uploaded_documents",
    "search_uploaded_documents",
    "research_web_fresh",
    "search_user_memory",
}


def _normalise_string_list(value: Any, field: str) -> list[str]:
    """Validate and normalise a list of non-empty strings."""
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
    """Load JSONL task definitions and reject malformed or duplicate tasks."""
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

            # Validate optional task lists.
            for field in (
                "expected_specialists",
                "expected_tools",
                "allowed_tools",
                "selected_document_ids",
            ):
                if field in item:
                    item[field] = _normalise_string_list(
                        item[field], field
                    )

            for field in ("requires_approval", "safe_to_run"):
                if field in item and not isinstance(item[field], bool):
                    raise ValueError(
                        f"{field} must be true or false for task {task_id}."
                    )

            execution_mode = item.get("execution_mode")

            if execution_mode is not None and execution_mode not in {
                "read_only",
                "manual",
                "approval_required",
            }:
                raise ValueError(
                    f"Invalid execution_mode for task {task_id}."
                )

            tasks[task_id] = item

    if not tasks:
        raise ValueError("The evaluation dataset contains no tasks.")

    return tasks


def _extract_actual_tools(record: dict[str, Any]) -> list[str]:
    """Extract tool names from supplied run records and LangGraph state."""
    tools: list[str] = []

    # Explicit measurements take priority.
    direct = record.get("actual_tools")

    if isinstance(direct, list):
        tools.extend(
            item for item in direct
            if isinstance(item, str) and item.strip()
        )

    # Extract tools from execution history.
    history = record.get("tool_history") or []

    if isinstance(history, list):
        for item in history:

            if isinstance(item, str):
                tools.append(item)

            elif isinstance(item, dict):
                name = item.get("tool_name") or item.get("name")

                if isinstance(name, str) and name.strip():
                    tools.append(name)

    # Also inspect model-selected tool calls.
    messages = record.get("messages") or []

    if isinstance(messages, list):
        for message in messages:

            if isinstance(message, dict):
                calls = message.get("tool_calls") or []

            else:
                calls = getattr(message, "tool_calls", []) or []

            for call in calls:

                if not isinstance(call, dict):
                    continue

                name = call.get("name")

                if isinstance(name, str) and name.strip():
                    tools.append(name)

    return list(dict.fromkeys(tools))


def _extract_actual_specialists(record: dict[str, Any]) -> list[str]:
    """Support both recorded specialist names and LangGraph state."""
    specialists = record.get("actual_specialists")

    if specialists is None:
        specialists = record.get("completed_specialists")

    if specialists is None:
        specialists = []

    return _normalise_string_list(
        specialists,
        "actual_specialists",
    )


def _pctl(values: list[float], percentile: float) -> float | None:
    """Calculate a nearest-rank percentile."""
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


def _record_from_run(
    task: dict[str, Any],
    run: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    """Convert a LangGraph result into a standard evaluation record."""
    return {
        "task_id": task["task_id"],
        "actual_specialists": _extract_actual_specialists(run),
        "actual_tools": _extract_actual_tools(run),
        "task_completed": run.get("task_status") == "completed",
        "approval_prompted": bool(run.get("pending_approval")),
        "action_executed_before_approval": False,
        "citations_valid": None,
        "latency_ms": round(latency_ms, 2),
        "errors": run.get("errors") or [],
    }


def evaluate_results(
    dataset: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate routing, tool, completion, approval and latency metrics."""
    if not isinstance(results, list):
        raise ValueError("results must be a JSON array.")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    routing_scores: list[bool] = []
    tool_selection_scores: list[bool] = []
    allowlist_scores: list[bool] = []
    completion_scores: list[bool] = []
    approval_scores: list[bool] = []
    citation_scores: list[bool] = []
    latency_values: list[float] = []

    for record in results:

        if not isinstance(record, dict):
            raise ValueError("Each result must be a JSON object.")

        task_id = record.get("task_id")

        if task_id not in dataset:
            raise ValueError(
                f"Result refers to unknown task_id: {task_id}"
            )

        if task_id in seen:
            raise ValueError(f"Duplicate result task_id: {task_id}")

        seen.add(task_id)

        task = dataset[task_id]

        actual_specialists = set(
            _extract_actual_specialists(record)
        )

        actual_tools = set(
            _extract_actual_tools(record)
        )

        # Backward-compatible specialist routing evaluation.
        expected_specialists = set(
            task.get("expected_specialists", [])
        )

        routing_ok = (
            actual_specialists == expected_specialists
            if expected_specialists
            else None
        )

        if routing_ok is not None:
            routing_scores.append(routing_ok)

        # New expected-tools evaluation.
        expected_tools = set(task.get("expected_tools", []))

        tool_selection_ok = (
            expected_tools.issubset(actual_tools)
            if expected_tools
            else None
        )

        if tool_selection_ok is not None:
            tool_selection_scores.append(tool_selection_ok)

        allowed_tools = set(task.get("allowed_tools", []))

        allowlist_ok = actual_tools.issubset(allowed_tools)
        allowlist_scores.append(allowlist_ok)

        # Task completion.
        completed = record.get("task_completed")

        if isinstance(completed, bool):
            completion_scores.append(completed)

        # Citation validity is excluded when not manually evaluated.
        citations_valid = record.get("citations_valid")

        if isinstance(citations_valid, bool):
            citation_scores.append(citations_valid)

        # Latency.
        latency = record.get("latency_ms")

        if (
            isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(latency)
            and latency >= 0
        ):
            latency = float(latency)
            latency_values.append(latency)

        # Approval safety.
        requires_approval = bool(task.get("requires_approval"))

        prompted = record.get("approval_prompted")
        early_execution = record.get(
            "action_executed_before_approval"
        )

        approval_passed = None

        if isinstance(early_execution, bool):

            if requires_approval:

                if isinstance(prompted, bool):
                    approval_passed = (
                        prompted is True
                        and early_execution is False
                    )

            else:
                approval_passed = early_execution is False

        if approval_passed is not None:
            approval_scores.append(approval_passed)

        rows.append({
            "task_id": task_id,
            "routing_correct": routing_ok,
            "tool_selection_correct": tool_selection_ok,
            "tool_allowlist_compliant": allowlist_ok,
            "actual_tools": sorted(actual_tools),
            "task_completed": completed,
            "approval_gate_passed": approval_passed,
            "latency_ms": latency,
            "citations_valid": citations_valid,
            "errors": record.get("errors") or [],
        })

    def rate(values: list[bool]) -> float | None:
        if not values:
            return None

        return round(sum(values) / len(values), 4)

    metrics = {
        "evaluated_tasks": len(rows),
        "routing_accuracy": rate(routing_scores),
        "tool_selection_accuracy": rate(tool_selection_scores),
        "tool_allowlist_compliance": rate(allowlist_scores),
        "completion_rate": rate(completion_scores),
        "approval_gate_pass_rate": rate(approval_scores),
        "citation_validity_rate": rate(citation_scores),
        "latency_ms_p50": _pctl(latency_values, 0.50),
        "latency_ms_p95": _pctl(latency_values, 0.95),
        "latency_samples": len(latency_values),
        "notes": (
            "Metrics use supplied measurements only. "
            "Missing measurements are excluded, not treated as successes."
        ),
    }

    return {
        "metrics": metrics,
        "rows": rows,
    }


def run_read_only_tasks(
    dataset: dict[str, dict[str, Any]],
    invoke: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute only explicitly marked read-only tasks.

    Gmail, form submission, memory writes and other side-effecting tasks
    are never run by this function.
    """
    if invoke is None:
        from agents.graph import invoke_assistant

        invoke = invoke_assistant

    results: list[dict[str, Any]] = []

    for task in dataset.values():

        if task.get("safe_to_run") is not True:
            continue

        if task.get("execution_mode") != "read_only":
            continue

        automation_request = task.get("automation_request") or {}

        if automation_request:

            if not isinstance(automation_request, dict):
                continue

            tool_name = automation_request.get("tool_name")

            if tool_name not in READ_ONLY_AUTOMATION_TOOLS:
                continue

        started = time.perf_counter()

        try:
            run = invoke(
                task["request"],
                selected_document_ids=task.get(
                    "selected_document_ids"
                ),
                automation_request=automation_request or None,
            )

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000

            results.append(
                _record_from_run(task, run, elapsed_ms)
            )

        except Exception as exc:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000

            results.append({
                "task_id": task["task_id"],
                "actual_specialists": [],
                "actual_tools": [],
                "task_completed": False,
                "approval_prompted": False,
                "action_executed_before_approval": False,
                "citations_valid": None,
                "latency_ms": round(elapsed_ms, 2),
                "errors": [str(exc)],
            })

    return results


def _write_report(
    report: dict[str, Any],
    output_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write JSON metrics and an optional per-task CSV."""
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
        default=None,
        help="Existing JSON file containing measured result records.",
    )

    parser.add_argument(
        "--run-read-only",
        action="store_true",
        help="Execute only dataset tasks explicitly marked safe_to_run.",
    )

    parser.add_argument(
        "--output",
        default="evals/agent_evaluation_report.json",
    )

    parser.add_argument(
        "--csv",
        default=None,
    )

    args = parser.parse_args()

    if not args.results and not args.run_read_only:
        parser.error(
            "Provide --results, --run-read-only, or both."
        )

    dataset = load_dataset(args.dataset)

    results: list[dict[str, Any]] = []

    if args.results:

        with Path(args.results).open(
            "r",
            encoding="utf-8",
        ) as handle:

            payload = json.load(handle)

        if isinstance(payload, dict):
            results.extend(payload.get("results", []))

        elif isinstance(payload, list):
            results.extend(payload)

        else:
            raise ValueError(
                "Results file must contain a JSON array or an object with results."
            )

    if args.run_read_only:
        results.extend(
            run_read_only_tasks(dataset)
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