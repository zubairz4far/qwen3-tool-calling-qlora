"""Evaluate tool-call JSONL with explicit input validation and abstention metrics."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JSONL tool-call predictions")
    parser.add_argument("predictions", help="JSONL file with expected/predicted tool and args")
    return parser.parse_args()


def exact_dict_match(expected, predicted):
    """Compare JSON values without Python's bool/int equivalence or missing-null ambiguity."""
    if type(expected) is not type(predicted):
        return False
    if isinstance(expected, dict):
        return expected.keys() == predicted.keys() and all(
            exact_dict_match(value, predicted[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(predicted) and all(
            exact_dict_match(left, right) for left, right in zip(expected, predicted)
        )
    return expected == predicted


def kv_accuracy(expected, predicted):
    if not expected:
        return 1.0 if not predicted else 0.0
    correct = sum(
        1
        for key, value in expected.items()
        if key in predicted and exact_dict_match(value, predicted[key])
    )
    return correct / len(expected)


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def validate_row(row, line_number):
    """Refuse malformed evidence rather than silently counting missing labels as correct."""
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")
    for field in ("expected_tool", "predicted_tool"):
        if field not in row:
            raise ValueError(f"line {line_number}: missing required field {field}")
        value = row[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"line {line_number}: {field} must be a tool name or null")
    for field in ("expected_args", "predicted_args"):
        value = row.get(field)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"line {line_number}: {field} must be a JSON object or null")
        if row[field.replace("_args", "_tool")] is not None and field not in row:
            raise ValueError(f"line {line_number}: tool rows require {field}")
    return row


def evaluate(rows):
    if not rows:
        raise ValueError("No prediction rows found")

    tool_correct = 0
    exact_call_correct = 0
    arg_exact_correct = 0
    kv_scores = []
    non_tool_examples = 0
    unexpected_tools = 0

    for row in rows:
        expected_tool = row["expected_tool"]
        predicted_tool = row["predicted_tool"]
        expected_args = row.get("expected_args") or {}
        predicted_args = row.get("predicted_args") or {}

        tool_match = expected_tool == predicted_tool
        args_match = exact_dict_match(expected_args, predicted_args)

        tool_correct += int(tool_match)
        arg_exact_correct += int(args_match)
        exact_call_correct += int(tool_match and args_match)
        kv_scores.append(kv_accuracy(expected_args, predicted_args))
        if expected_tool is None:
            non_tool_examples += 1
            unexpected_tools += int(predicted_tool is not None)

    n = len(rows)
    return {
        "examples": n,
        "tool_selection_accuracy": tool_correct / n,
        "strict_exact_call_accuracy": exact_call_correct / n,
        "argument_exact_accuracy": arg_exact_correct / n,
        "argument_kv_accuracy": sum(kv_scores) / n,
        "non_tool_examples": non_tool_examples,
        "unexpected_tool_calls_on_non_tool_cases": unexpected_tools,
        "hallucinated_tool_rate_on_non_tool_cases": (
            unexpected_tools / non_tool_examples if non_tool_examples else 0.0
        ),
    }


def main():
    args = parse_args()
    rows = []
    try:
        with Path(args.predictions).open("r", encoding="utf-8") as f:
            for line_number, raw in enumerate(f, start=1):
                if raw.strip():
                    try:
                        row = json.loads(
                            raw,
                            object_pairs_hook=_reject_duplicate_keys,
                            parse_constant=_reject_nonfinite,
                        )
                    except ValueError as exc:
                        raise ValueError(f"line {line_number}: {exc}") from exc
                    rows.append(validate_row(row, line_number))
        metrics = evaluate(rows)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid prediction evidence: {exc}") from exc
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
