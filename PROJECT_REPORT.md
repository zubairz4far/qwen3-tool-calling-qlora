# Experiment Report

## Objective

Fine-tune Qwen3-1.7B to improve reliable tool/function calling while measuring not only exact-call accuracy but also routing behavior, clarification handling, no-tool decisions, parse validity, and prompt-injection robustness.

## Dataset

The experiment used 1,600 synthetic examples split into:

- 1,200 training examples
- 160 validation examples
- 240 held-out test examples

The examples were designed around support-style tool use and included positive tool calls, non-tool requests, missing-information cases, and adversarial instructions.

## Training setup

The model was adapted using QLoRA / parameter-efficient fine-tuning on a Kaggle NVIDIA T4 GPU.

Observed training summary:

- Global step: 38
- Final training loss: ~0.193
- Runtime: ~1,373 seconds
- Checkpoints saved as adapter/tokenizer artifacts

The working stack included:

- PyTorch 2.10.0+cu128
- Transformers 5.14.1
- TRL 1.9.2
- PEFT 0.19.1
- bitsandbytes 0.50.0

## Held-out results

### Fine-tuned model

| Metric | Score |
|---|---:|
| Tool selection accuracy | 0.8500 |
| Strict exact call accuracy | 0.7417 |
| Argument exact accuracy | 0.7417 |
| Argument key/value accuracy | 0.8049 |
| Tool-call parse validity | 0.8500 |

### Improvement versus baseline

| Metric | Percentage-point change |
|---|---:|
| Tool selection accuracy | +13.33 |
| Strict exact call accuracy | +21.11 |
| Argument KV accuracy | +15.93 |
| Tool parse validity | +13.33 |
| No-tool routing accuracy | 0.00 |
| Clarification accuracy | -36.67 |
| Prompt-injection exact accuracy | +40.00 |
| Hallucinated-tool rate on non-tool prompts | +15.00 |
| Overall routing accuracy | +2.00 |
| Overall strict behavior accuracy | +5.33 |

## Interpretation

The strongest gains came from structured call construction and prompt-injection exact behavior. The model became better at selecting tools and forming arguments, which directly supports practical agent workflows.

However, clarification accuracy dropped substantially and the hallucinated-tool rate on non-tool examples increased. This indicates that the fine-tune shifted the model toward being more eager to call tools. A single aggregate tool-call score would have obscured that failure mode.

## Next iteration

The next training round should prioritize:

1. More clarification-required samples where required arguments are deliberately omitted.
2. More hard negative examples where a plausible tool exists but should not be called.
3. Contrastive examples pairing nearly identical prompts with different routing outcomes.
4. A weighted behavioral objective/evaluation score that penalizes unnecessary tool use.
5. Error buckets by intent/tool to identify whether regressions are concentrated in specific schemas.

## Engineering takeaway

For production agents, exact-call accuracy is necessary but insufficient. The model must also know when **not** to call a tool and when to ask the user for missing information. Behavioral evaluation should therefore be treated as part of model development, not as a final benchmark-only step.
