import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JSONL tool-call predictions")
    parser.add_argument("predictions", help="JSONL file with expected/predicted tool and args")
    return parser.parse_args()


def exact_dict_match(expected, predicted):
    return expected == predicted


def kv_accuracy(expected, predicted):
    if not expected:
        return 1.0 if not predicted else 0.0
    correct = sum(1 for key, value in expected.items() if predicted.get(key) == value)
    return correct / len(expected)


def main():
    args = parse_args()
    rows = []
    with Path(args.predictions).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise SystemExit("No prediction rows found")

    tool_correct = 0
    exact_call_correct = 0
    arg_exact_correct = 0
    kv_scores = []

    for row in rows:
        expected_tool = row.get("expected_tool")
        predicted_tool = row.get("predicted_tool")
        expected_args = row.get("expected_args") or {}
        predicted_args = row.get("predicted_args") or {}

        tool_match = expected_tool == predicted_tool
        args_match = exact_dict_match(expected_args, predicted_args)

        tool_correct += int(tool_match)
        arg_exact_correct += int(args_match)
        exact_call_correct += int(tool_match and args_match)
        kv_scores.append(kv_accuracy(expected_args, predicted_args))

    n = len(rows)
    metrics = {
        "examples": n,
        "tool_selection_accuracy": tool_correct / n,
        "strict_exact_call_accuracy": exact_call_correct / n,
        "argument_exact_accuracy": arg_exact_correct / n,
        "argument_kv_accuracy": sum(kv_scores) / n,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
