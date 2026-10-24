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


class EvaluatePredictionsEvidenceValidationTests(unittest.TestCase):
    def test_rejects_missing_tool_labels_that_previously_counted_as_correct(self):
        with self.assertRaisesRegex(ValueError, "missing required field expected_tool"):
            evaluator.validate_row({"predicted_tool": None}, 3)
        with self.assertRaisesRegex(ValueError, "missing required field predicted_tool"):
            evaluator.validate_row({"expected_tool": None}, 3)

    def test_rejects_malformed_argument_payloads(self):
        row = {
            "expected_tool": "get_order",
            "predicted_tool": "get_order",
            "expected_args": {"order_id": "A"},
            "predicted_args": ["not", "a", "dict"],
        }
        with self.assertRaisesRegex(ValueError, "predicted_args"):
            evaluator.validate_row(row, 2)

    def test_rejects_tool_rows_with_missing_argument_field(self):
        row = {"expected_tool": "get_order", "predicted_tool": "get_order"}
        with self.assertRaisesRegex(ValueError, "expected_args"):
            evaluator.validate_row(row, 1)

    def test_measures_unexpected_calls_separately_from_strict_accuracy(self):
        rows = [
            evaluator.validate_row(
                {
                    "expected_tool": None,
                    "predicted_tool": None,
                    "expected_args": {},
                    "predicted_args": {},
                },
                1,
            ),
            evaluator.validate_row(
                {
                    "expected_tool": None,
                    "predicted_tool": "get_order",
                    "expected_args": {},
                    "predicted_args": {"order_id": "invented"},
                },
                2,
            ),
        ]
        metrics = evaluator.evaluate(rows)
        self.assertEqual(metrics["non_tool_examples"], 2)
        self.assertEqual(metrics["unexpected_tool_calls_on_non_tool_cases"], 1)
        self.assertEqual(metrics["hallucinated_tool_rate_on_non_tool_cases"], 0.5)

    def test_rejects_non_object_jsonl_row(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            evaluator.validate_row(["not", "an", "object"], 1)


class StrictJsonEvidenceTests(unittest.TestCase):
    def test_boolean_is_not_integer_for_exact_match_or_kv(self):
        expected = {"limit": 1, "options": [{"enabled": True}]}
        predicted = {"limit": True, "options": [{"enabled": 1}]}
        self.assertFalse(evaluator.exact_dict_match(expected, predicted))
        self.assertEqual(evaluator.kv_accuracy(expected, predicted), 0.0)

    def test_missing_null_key_does_not_get_credit(self):
        self.assertFalse(evaluator.exact_dict_match({"note": None}, {}))
        self.assertEqual(evaluator.kv_accuracy({"note": None}, {}), 0.0)
        self.assertEqual(evaluator.kv_accuracy({"note": None}, {"note": None}), 1.0)

    def test_cli_rejects_duplicate_jsonl_keys_and_reports_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            path.write_text('{"expected_tool": null, "expected_tool": "get_order", "predicted_tool": null}\n')
            completed = subprocess.run(
                [sys.executable, str(EVALUATOR_PATH), str(path)],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("line 1", completed.stderr)
        self.assertIn("duplicate JSON key", completed.stderr)

    def test_cli_rejects_nonfinite_metric_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            path.write_text('{"expected_tool": null, "predicted_tool": null, "expected_args": {"score": NaN}}\n')
            completed = subprocess.run(
                [sys.executable, str(EVALUATOR_PATH), str(path)],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("non-finite JSON constant", completed.stderr)
