---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- qwen3
- qlora
- peft
- trl
- sft
- tool-calling
- function-calling
- structured-output
---

# Qwen3-1.7B Tool-Calling QLoRA (V1)

PEFT/QLoRA adapter fine-tuned from `Qwen/Qwen3-1.7B` for structured calls across seven e-commerce and operations tools.

This is the **selected V1 release**. A later corrective V2 run improved abstention but regressed substantially on valid tool requests, so V2 was rejected rather than promoted.

- **Interactive showcase:** [Qwen3 Tool Calling Lab](https://huggingface.co/spaces/zubairz4far/qwen3-tool-calling-demo)
- **Code, evaluation, and experiment record:** [zubairz4far/qwen3-tool-calling-qlora](https://github.com/zubairz4far/qwen3-tool-calling-qlora)
- **Rejected experiment:** `zubairz4far/qwen3-1.7b-tool-calling-v2`

## Training details

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-1.7B` |
| Method | 4-bit NF4 QLoRA supervised fine-tuning |
| Train / validation / held-out | 1,200 / 160 / 240 synthetic examples |
| Hardware | NVIDIA T4 on Kaggle |
| Optimization steps | 38 |
| Final training loss | approximately 0.193 |
| Runtime | approximately 1,373 seconds |

## Controlled held-out benchmark (240 examples)

The 240 held-out cases used paraphrases separate from the training template pool.

| Metric | Base | V1 | Delta |
|---|---:|---:|---:|
| Tool selection accuracy | 85.00% | **100.00%** | **+15.00 pp** |
| Strict exact-call accuracy | 74.17% | **95.00%** | **+20.83 pp** |
| Argument key/value accuracy | 80.49% | **98.33%** | **+17.85 pp** |
| Tool-call parse validity | 85.00% | **100.00%** | **+15.00 pp** |

## Locked final benchmark (150 unseen cases)

The final benchmark was frozen before comparing the base model, V1, and V2. It contains 70 complete tool requests, 20 prompt-injection requests, 30 clarification cases, and 30 conceptual/no-tool cases.

SHA-256: `5e1cbede7c2f80ee712c36dfee1dcf26b6a4b03e75c5ddc4d8dd3b9e3c5e0b02`

| Metric | Base | V1 | Delta |
|---|---:|---:|---:|
| Tool selection accuracy | 85.56% | **98.89%** | **+13.33 pp** |
| Strict exact-call accuracy | 73.33% | **94.44%** | **+21.11 pp** |
| Argument KV accuracy | 81.48% | **97.41%** | **+15.93 pp** |
| Tool parse validity | 85.56% | **98.89%** | **+13.33 pp** |
| No-tool routing accuracy | 100.00% | **100.00%** | 0.00 pp |
| Clarification accuracy | **36.67%** | 0.00% | **-36.67 pp** |
| Prompt-injection exact accuracy | 55.00% | **95.00%** | **+40.00 pp** |
| Overall strict behavior accuracy | 71.33% | **76.67%** | **+5.33 pp** |

V1 is strong at valid structured tool execution but too eager to invoke tools when required information is missing. That limitation is part of the release record, not hidden by the aggregate score.

## V2 decision

V2 continued training from V1 on 600 corrective examples. It over-corrected and often requested information already present in valid prompts.

| V2 diagnostic | Result |
|---|---:|
| Tool selection accuracy | 44.44% |
| Strict exact-call accuracy | 42.22% |
| Argument KV accuracy | 43.70% |
| Tool parse validity | 44.44% |
| Overall strict behavior accuracy | 57.33% |

**Release decision:** retain V1 and reject V2. The locked final benchmark was not reused for another training round.

## Loading the adapter

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen3-1.7B"
adapter_id = "zubairz4far/qwen3-1.7b-tool-calling"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```

For reproducible prompting with tool schemas, use [`src/inference.py`](https://github.com/zubairz4far/qwen3-tool-calling-qlora/blob/main/src/inference.py).

```bash
python src/inference.py \
  --adapter zubairz4far/qwen3-1.7b-tool-calling \
  --tools examples/tools.json \
  --prompt "Check inventory for SKU GLM-001"
```

Generation is deterministic (`do_sample=False`). Applications must validate the selected tool and arguments before performing real-world actions.

## Intended use and limitations

This adapter is intended for portfolio, research, and experimentation with function calling, tool routing, and structured generation. It is not a generally safe autonomous agent or a production action-execution system.

- The datasets and benchmarks are synthetic and domain-specific.
- Reported results are controlled benchmark scores, not general real-world accuracy.
- Clarification behavior is the principal known V1 weakness.
- Free-form support-ticket text can make strict argument matching brittle.
- Downstream systems should schema-validate every generated call.

## Author

**Zubair Zafar** — AI/ML Engineer focused on LLM fine-tuning, agentic systems, and automation.
