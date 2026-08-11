# Reliable Function Calling on a 1.7B LLM
## QLoRA fine-tuning, locked evaluation, and failure analysis

This project fine-tunes **Qwen3-1.7B** for structured tool calling across seven e-commerce/operations functions using **4-bit QLoRA**, then evaluates the base model and fine-tuned adapters at the function-call level.

The main lesson is not only that fine-tuning improved tool execution. It also exposed a real post-training trade-off: the strongest tool-calling model became too eager to call tools when required information was missing. A corrective V2 experiment then over-corrected and became too conservative. The final release decision therefore keeps **V1** and documents the remaining clarification limitation instead of hiding it.

## Selected model

**Release:** `zubairz4far/qwen3-1.7b-tool-calling`  
**Rejected corrective experiment:** `zubairz4far/qwen3-1.7b-tool-calling-v2`  
**Demo:** [Qwen3 Tool Calling Lab](https://huggingface.co/spaces/zubairz4far/qwen3-tool-calling-demo)

V1 is retained because it provides the best balance on the locked final benchmark, especially for valid structured tool requests and adversarial prompts.

## What the model can call

1. `search_orders`
2. `get_order`
3. `calculate_sales`
4. `search_customers`
5. `check_inventory`
6. `create_support_ticket`
7. `schedule_callback`

The training/evaluation system prompt instructs the model to use a tool only when needed, never invent arguments, and ask for missing required arguments.

## Training setup

- Base model: Qwen3-1.7B
- Method: 4-bit NF4 QLoRA
- Training data: 1,200 synthetic training examples
- Validation: 160 examples
- Controlled held-out test: 240 examples
- GPU: NVIDIA T4
- Tool catalog: 7 operations functions
- Evaluation: deterministic generation (`do_sample=False`)
- Primary outputs: tool selection, exact arguments, key-value argument accuracy, parse validity, and exact-call accuracy

The original one-epoch V1 run completed at 38 optimization steps with training loss about `0.193`.

## Controlled held-out benchmark (240 examples)

The first controlled benchmark used held-out paraphrases that were separate from the training template pool.

| Metric | Base | V1 | Improvement |
|---|---:|---:|---:|
| Tool selection accuracy | 85.00% | **100.00%** | **+15.00 pp** |
| Strict argument exact match | 74.17% | **95.00%** | **+20.83 pp** |
| Argument KV accuracy | 80.49% | **98.33%** | **+17.85 pp** |
| Parse validity | 85.00% | **100.00%** | **+15.00 pp** |
| Exact function-call accuracy | 74.17% | **95.00%** | **+20.83 pp** |

The remaining strict V1 failures were concentrated in `create_support_ticket`, where free-form issue wording made strict text matching brittle.

## Locked final benchmark (150 unseen cases)

After development and failure analysis, a separate **150-case final benchmark** was created and locked before evaluating Base, V1, and V2.

Benchmark composition:

- 70 complete normal tool requests
- 20 adversarial / prompt-injection tool requests
- 30 missing-required-argument clarification cases
- 30 conceptual / no-tool cases

SHA-256:

```text
5e1cbede7c2f80ee712c36dfee1dcf26b6a4b03e75c5ddc4d8dd3b9e3c5e0b02
```

### Base vs V1

| Metric | Base | V1 | Delta |
|---|---:|---:|---:|
| Tool selection accuracy | 85.56% | **98.89%** | **+13.33 pp** |
| Strict exact-call accuracy | 73.33% | **94.44%** | **+21.11 pp** |
| Argument KV accuracy | 81.48% | **97.41%** | **+15.93 pp** |
| Tool parse validity | 85.56% | **98.89%** | **+13.33 pp** |
| No-tool routing accuracy | 100.00% | **100.00%** | 0.00 pp |
| Clarification accuracy | **36.67%** | 0.00% | **-36.67 pp** |
| Prompt-injection exact accuracy | 55.00% | **95.00%** | **+40.00 pp** |
| Hallucinated-tool rate on non-tool cases | **25.00%** | 40.00% | **+15.00 pp worse** |
| Overall routing accuracy | 81.33% | **83.33%** | +2.00 pp |
| Overall strict behavior accuracy | 71.33% | **76.67%** | +5.33 pp |

## Corrective V2 experiment: rejected

V1's main weakness was over-eager tool invocation when a required argument was missing. A corrective dataset of 600 examples was created:

- 300 missing-argument clarification examples
- 100 noisy `get_order` examples
- 100 normal complete tool calls
- 100 conceptual/no-tool examples

V2 was continued from the V1 adapter with a small corrective learning rate. The intervention improved the model's tendency to abstain, but it **over-corrected**: V2 frequently asked for information that was already present in valid requests.

Known locked-benchmark diagnostics:

| Metric | V2 |
|---|---:|
| Tool selection accuracy | 44.44% |
| Strict exact-call accuracy | 42.22% |
| Argument KV accuracy | 43.70% |
| Tool parse validity | 44.44% |
| Strict behavior failures | 64 / 150 |
| Overall strict behavior from failure count | 57.33% |

**Release decision:** reject V2 and keep V1. The final test was not used for another round of training.

## Why this project is portfolio-worthy

This project demonstrates more than a single fine-tuning run:

- supervised fine-tuning for structured generation
- PEFT / LoRA and 4-bit quantization
- tool-schema design
- chat-template handling
- controlled synthetic data generation
- held-out paraphrase evaluation
- deterministic function-call parsing
- exact and partial argument metrics
- adversarial prompt testing
- missing-information / clarification testing
- failure analysis
- corrective data design
- regression detection
- model release selection based on a locked final benchmark

## Reproducing evaluation

1. Use one GPU for the 4-bit model. Set `CUDA_VISIBLE_DEVICES=0` before importing PyTorch in multi-GPU Kaggle sessions.
2. Load Qwen3-1.7B with NF4 4-bit quantization.
3. Load the V1 PEFT adapter from `zubairz4far/qwen3-1.7b-tool-calling`.
4. Load the locked final benchmark.
5. Use deterministic generation.
6. Parse Qwen `<tool_call>...</tool_call>` blocks.
7. Compute tool selection, exact-call, argument KV, parse validity, clarification, no-tool routing, and adversarial metrics.

## Limitations

V1 should **not** be described as a generally safe autonomous agent.

The locked benchmark shows that V1 is excellent at valid tool execution but weak at clarification: it may call tools too aggressively when a required argument is missing. This is the main remaining limitation and should be addressed with a more carefully balanced future dataset rather than another corrective pass on the locked final benchmark.

The datasets are synthetic and domain-specific. Reported metrics are therefore **controlled benchmark results**, not general real-world accuracy.

## Resume-ready summary

> Fine-tuned Qwen3-1.7B with 4-bit QLoRA for seven operations tools, improving strict exact function-call accuracy from 73.3% to 94.4%, tool-selection accuracy from 85.6% to 98.9%, and adversarial exact-call accuracy from 55% to 95% on a locked 150-case unseen benchmark; performed failure analysis and corrective SFT experiments to quantify routing and clarification trade-offs.

## Stack

Python, PyTorch, Transformers, TRL, PEFT, bitsandbytes, Hugging Face, pandas, NumPy, Kaggle
