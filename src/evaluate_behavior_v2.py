import argparse
import json
from collections import defaultdict
from pathlib import Path


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate V2 behavioral predictions")
    parser.add_argument("predictions", help="JSONL with expected_behavior/predicted_behavior and optional tool/args")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No rows found")

    behavior_correct = 0
    tool_correct = 0
    strict_correct = 0
    tool_rows = 0
    category = defaultdict(lambda: [0, 0])
    hallucinated_tool_non_tool = 0
    non_tool_rows = 0

    for row in rows:
        expected_behavior = row.get("expected_behavior")
        predicted_behavior = row.get("predicted_behavior")
        expected_tool = row.get("expected_tool")
        predicted_tool = row.get("predicted_tool")
        expected_args = row.get("expected_args") or {}
        predicted_args = row.get("predicted_args") or {}
        cat = row.get("category", "unknown")

        ok_behavior = expected_behavior == predicted_behavior
        behavior_correct += int(ok_behavior)
        category[cat][0] += int(ok_behavior)
        category[cat][1] += 1

        if expected_behavior == "tool":
            tool_rows += 1
            tool_match = expected_tool == predicted_tool
            tool_correct += int(tool_match)
            strict_correct += int(tool_match and expected_args == predicted_args)
        else:
            non_tool_rows += 1
            hallucinated_tool_non_tool += int(predicted_behavior == "tool")

    metrics = {
        "examples": len(rows),
        "behavior_accuracy": safe_div(behavior_correct, len(rows)),
        "tool_selection_accuracy_on_tool_cases": safe_div(tool_correct, tool_rows),
        "strict_exact_call_accuracy_on_tool_cases": safe_div(strict_correct, tool_rows),
        "hallucinated_tool_rate_on_non_tool_cases": safe_div(hallucinated_tool_non_tool, non_tool_rows),
        "category_accuracy": {k: safe_div(v[0], v[1]) for k, v in sorted(category.items())},
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
