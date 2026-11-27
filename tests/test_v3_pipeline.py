import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from v3_data import build, validate, write_dataset, read_verified, REGISTRY
from v3_protocol import score, target, parse_output

class V3Tests(unittest.TestCase):
    def test_determinism_and_split(self):
        train, dev=build("train"),build("dev")
        self.assertEqual(train,build("train"))
        self.assertFalse(validate(train,"train") & validate(dev,"dev"))
        self.assertEqual(set(r["expected_tool"] for r in train if r["expected_tool"]),set(REGISTRY))
        self.assertEqual(len(train),1260)
        self.assertEqual(len(dev),420)
    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_dataset(root)
            self.assertEqual(len(read_verified(root,"train")),1260)
            (root/"train.jsonl").write_text("tampered")
            with self.assertRaises(ValueError):
                read_verified(root,"train")
    def test_targets_and_guard(self):
        for row in build("dev"):
            self.assertTrue(score(row,target(row))["strict_pass"],row["id"])
        row=next(r for r in build("dev") if r["category"]=="missing")
        self.assertTrue(score(row,'<tool_call>{"name":"get_order","arguments":{"order_id":"X"}}</tool_call>')["unexpected_tool_attempt"])
    def test_strict_parser(self):
        for bad in ['<tool_call>{"name":"get_order","arguments":{"order_id":1}}</tool_call>',
                    '<tool_call>{"name":"get_order","name":"get_order","arguments":{"order_id":"X"}}</tool_call>',
                    '<tool_call>{"name":"get_order","arguments":{"order_id":"X","extra":"Y"}}</tool_call>',
                    '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
                    'prefix <tool_call>{"name":"get_order","arguments":{"order_id":"X"}}</tool_call>']:
            self.assertEqual(parse_output(bad)["kind"],"invalid")
        self.assertEqual(parse_output('Example JSON: {"name":"get_order"}')["kind"],"text")
if __name__=="__main__":
    unittest.main()
