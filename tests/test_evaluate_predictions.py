import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "src" / "evaluate_predictions.py"

spec = importlib.util.spec_from_file_location("evaluate_predictions", EVALUATOR_PATH)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class EvaluatePredictionsUnitTests(unittest.TestCase):
    def test_exact_dict_match_is_strict(self):
        self.assertTrue(evaluator.exact_dict_match({"id": 7}, {"id": 7}))
        self.assertFalse(evaluator.exact_dict_match({"id": 7}, {"id": "7"}))
        self.assertFalse(evaluator.exact_dict_match({"id": 7}, {"id": 7, "extra": True}))

    def test_kv_accuracy_scores_partial_arguments(self):
        expected = {"order_id": "A-1", "limit": 5}
        predicted = {"order_id": "A-1", "limit": 10}
        self.assertEqual(evaluator.kv_accuracy(expected, predicted), 0.5)

    def test_kv_accuracy_handles_empty_expected_args(self):
        self.assertEqual(evaluator.kv_accuracy({}, {}), 1.0)
        self.assertEqual(evaluator.kv_accuracy({}, {"unexpected": 1}), 0.0)


class EvaluatePredictionsCliTests(unittest.TestCase):
    def test_cli_reports_expected_metrics(self):
        rows = [
            {
                "expected_tool": "get_order",
                "predicted_tool": "get_order",
                "expected_args": {"order_id": "A-1"},
                "predicted_args": {"order_id": "A-1"},
            },
            {
                "expected_tool": "check_inventory",
                "predicted_tool": "check_inventory",
                "expected_args": {"sku": "SKU-9", "warehouse": "PK"},
                "predicted_args": {"sku": "SKU-9", "warehouse": "US"},
            },
            {
                "expected_tool": "search_orders",
                "predicted_tool": "get_order",
                "expected_args": {},
                "predicted_args": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            completed = subprocess.run(
                [sys.executable, str(EVALUATOR_PATH), str(path)],
                check=True,
                capture_output=True,
                text=True,
            )

        metrics = json.loads(completed.stdout)
        self.assertEqual(metrics["examples"], 3)
        self.assertAlmostEqual(metrics["tool_selection_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["strict_exact_call_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["argument_exact_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["argument_kv_accuracy"], (1.0 + 0.5 + 1.0) / 3)


if __name__ == "__main__":
    unittest.main()
