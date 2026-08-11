# V2 Experiment Plan

## Goal

Retain the V1 gains in exact tool-call construction while improving the two observed regressions: clarification behavior and unnecessary tool invocation.

## V1 baseline to preserve

- Tool selection accuracy: 85.00%
- Strict exact call accuracy: 74.17%
- Argument key/value accuracy: 80.49%
- Tool-call parse validity: 85.00%

## V2 data emphasis

Generate additional examples in five buckets:

1. Tool-required prompts with explicit required arguments.
2. Clarification-required prompts where a required argument is missing.
3. No-tool prompts that should receive a normal direct answer.
4. Hard negatives that mention tools or JSON but explicitly should not trigger a call.
5. Prompt-injection cases that try to force invented or unrelated tools.

Run:

```bash
python src/generate_v2_cases.py --output data/v2_behavior_cases.jsonl
```

The default generator produces a balanced hardening set suitable for mixing into the next SFT dataset. Keep the original held-out V1 test set unchanged so V1 vs V2 remains a fair comparison.

## Evaluation contract

Prediction JSONL rows should contain:

```json
{
  "prompt": "Cancel my order",
  "category": "missing_order_id",
  "expected_behavior": "clarify",
  "predicted_behavior": "clarify",
  "expected_tool": "cancel_order",
  "predicted_tool": null,
  "expected_args": {},
  "predicted_args": {}
}
```

Evaluate with:

```bash
python src/evaluate_behavior_v2.py predictions_v2.jsonl
```

Report at minimum:

- overall behavior accuracy
- tool selection accuracy on tool-required cases
- strict exact-call accuracy on tool-required cases
- hallucinated-tool rate on non-tool/clarification cases
- per-category accuracy

## Promotion criteria

Promote V2 only if it improves clarification and no-tool behavior without materially sacrificing V1's structured-call gains. Suggested portfolio targets are:

- Tool selection: >= 90%
- Strict exact call: >= 80%
- Clarification accuracy: strong positive recovery versus V1
- Hallucinated-tool rate: lower than V1

Targets are goals, not claims; publish only measured results.

## Demo

The `space/` directory contains a Gradio app that loads the public PEFT adapter and provides editable tool schemas, raw generation, lightweight routing classification, and detected JSON output.

Copy that directory into a Hugging Face Gradio Space to publish the live demo.
