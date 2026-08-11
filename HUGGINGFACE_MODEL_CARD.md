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
- agents
- structured-output
---

# Qwen3-1.7B Tool-Calling QLoRA

QLoRA adapter fine-tuned from `Qwen/Qwen3-1.7B` for structured tool/function calling, with evaluation covering tool selection, argument generation, clarification behavior, no-tool routing, parse validity, prompt-injection robustness, and tool hallucination.

**Portfolio showcase:** [Qwen3 Tool Calling Lab](https://huggingface.co/spaces/zubairz4far/qwen3-tool-calling-demo)  
**Source and evaluation:** [zubairz4far/qwen3-tool-calling-qlora](https://github.com/zubairz4far/qwen3-tool-calling-qlora)

## Model details

- **Base model:** `Qwen/Qwen3-1.7B`
- **Fine-tuning:** Supervised fine-tuning with QLoRA / PEFT
- **Training framework:** Transformers + TRL + PEFT + bitsandbytes
- **Training hardware:** NVIDIA T4 on Kaggle
- **Dataset size:** 1,600 synthetic tool-calling/support examples
- **Split:** 1,200 train / 160 validation / 240 held-out test
- **Final training loss:** ~0.193
- **Training steps:** 38
- **Training runtime:** ~1,373 seconds

## Held-out evaluation

Evaluation was run on 240 held-out examples.

| Metric | Result |
|---|---:|
| Tool selection accuracy | **85.00%** |
| Strict exact call accuracy | **74.17%** |
| Argument exact accuracy | **74.17%** |
| Argument key/value accuracy | **80.49%** |
| Tool-call parse validity | **85.00%** |

## Behavioral evaluation

The evaluation suite includes more than ordinary function-name accuracy. It measures:

- correct tool selection
- exact tool name + argument construction
- individual argument key/value accuracy
- machine-readable parse validity
- no-tool routing
- clarification when required information is missing
- prompt-injection behavior
- hallucinated/unnecessary tool calls

The fine-tune substantially improved structured call construction and prompt-injection exact behavior, while error analysis showed that clarification behavior and unnecessary tool invocation require additional hard-negative training examples. These regressions are intentionally documented because the goal of the project is reliable agent behavior, not only a single aggregate benchmark score.

## Intended use

This adapter is intended for experimentation and portfolio/research work involving:

- function calling
- tool routing
- structured JSON-style outputs
- small-model agent systems
- tool-use reliability evaluation

It is not presented as a production safety model. Applications that trigger real-world actions should validate generated tool calls and arguments before execution.

## Loading the adapter

```python
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

## Example tool-calling flow

```text
User request
    ↓
Qwen3-1.7B + QLoRA adapter
    ↓
Routing decision
    ├── Tool required → choose tool → construct arguments → structured call
    ├── Missing information → clarification
    └── No tool required → direct response
```

## Limitations and next iteration

The held-out evaluation identified two important areas for improvement:

1. **Clarification-required prompts** — add more examples where required arguments are unavailable and the model must ask instead of guessing.
2. **No-tool / hard-negative routing** — add examples designed to reduce unnecessary tool invocation.

A V2 experiment should retain the gains in strict call accuracy while improving these behavioral dimensions.

## Reproducibility and evaluation

The companion GitHub repository contains the project report, inference script, evaluation script, and structured evaluation results:

`zubairz4far/qwen3-tool-calling-qlora`

## Author

**Zubair Zafar**  
AI / ML Engineer — LLM fine-tuning, agentic systems, and automation
