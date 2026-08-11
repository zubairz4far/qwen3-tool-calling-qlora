# Experiment Report

## Objective

Adapt Qwen3-1.7B for reliable structured tool calling and evaluate not only valid calls, but also abstention, clarification, no-tool routing, parse validity, and prompt-injection robustness.

## V1 setup

- Base model: `Qwen/Qwen3-1.7B`
- Method: supervised fine-tuning with 4-bit NF4 QLoRA
- Domain: seven e-commerce and operations tools
- Data: 1,200 train, 160 validation, 240 controlled held-out cases
- Hardware: Kaggle NVIDIA T4
- Global step: 38
- Final training loss: approximately 0.193
- Runtime: approximately 1,373 seconds

The observed environment used PyTorch 2.10.0+cu128, Transformers 5.14.1, TRL 1.9.2, PEFT 0.19.1, and bitsandbytes 0.50.0.

## Controlled held-out benchmark

| Metric | Base | V1 | Delta |
|---|---:|---:|---:|
| Tool selection accuracy | 85.00% | **100.00%** | **+15.00 pp** |
| Strict exact-call accuracy | 74.17% | **95.00%** | **+20.83 pp** |
| Argument KV accuracy | 80.49% | **98.33%** | **+17.85 pp** |
| Tool parse validity | 85.00% | **100.00%** | **+15.00 pp** |

The remaining strict V1 errors were concentrated in `create_support_ticket`, where free-form issue wording made exact text comparison brittle.

## Locked final benchmark

A separate set of 150 unseen cases was frozen before the Base/V1/V2 comparison:

- 70 complete normal tool requests
- 20 adversarial or prompt-injection tool requests
- 30 missing-required-argument clarification cases
- 30 conceptual or no-tool cases

Benchmark SHA-256: `5e1cbede7c2f80ee712c36dfee1dcf26b6a4b03e75c5ddc4d8dd3b9e3c5e0b02`

| Metric | Base | V1 | Delta |
|---|---:|---:|---:|
| Tool selection accuracy | 85.56% | **98.89%** | **+13.33 pp** |
| Strict exact-call accuracy | 73.33% | **94.44%** | **+21.11 pp** |
| Argument KV accuracy | 81.48% | **97.41%** | **+15.93 pp** |
| Tool parse validity | 85.56% | **98.89%** | **+13.33 pp** |
| No-tool routing accuracy | 100.00% | **100.00%** | 0.00 pp |
| Clarification accuracy | **36.67%** | 0.00% | **-36.67 pp** |
| Prompt-injection exact accuracy | 55.00% | **95.00%** | **+40.00 pp** |
| Hallucinated-tool rate on non-tool/clarification cases | **25.00%** | 40.00% | **+15.00 pp worse** |
| Overall routing accuracy | 81.33% | **83.33%** | +2.00 pp |
| Overall strict behavior accuracy | 71.33% | **76.67%** | +5.33 pp |

## Failure analysis

V1 materially improved valid structured calls and adversarial exact behavior. It also learned an overly aggressive tool-use policy: when a required value was absent, it tended to call a tool instead of asking a clarifying question. Reporting only exact-call accuracy would have concealed this regression.

## Corrective V2 experiment

V2 continued from the V1 adapter with a small learning rate and 600 corrective examples: 300 clarification cases, 100 noisy `get_order` cases, 100 complete tool calls, and 100 conceptual/no-tool prompts.

| V2 metric | Result |
|---|---:|
| Tool selection accuracy | 44.44% |
| Strict exact-call accuracy | 42.22% |
| Argument KV accuracy | 43.70% |
| Tool parse validity | 44.44% |
| Strict behavior failures | 64 / 150 |
| Overall strict behavior accuracy | 57.33% |

V2 over-corrected and often asked for details already present in complete requests. It failed the promotion criteria and was rejected.

## Release decision

V1 remains the selected adapter because it offers the best measured balance on the locked benchmark. The final set was not used for another training iteration. Future work should use a new development set and a more balanced mix of complete/clarification contrast pairs.

## Engineering takeaway

Exact-call accuracy is necessary but insufficient for agent reliability. Release evaluation must also measure when a model should abstain, ask for missing information, and resist unnecessary or injected calls.
