---
title: Qwen3 1.7B Tool Calling QLoRA
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
---

# Qwen3-1.7B Tool-Calling QLoRA Demo

Interactive demo for `zubairz4far/qwen3-1.7b-tool-calling`.

The demo exposes the model's raw generation, a lightweight behavior classification, and detected JSON so recruiters and reviewers can inspect tool-use behavior directly.

## Example behaviors

- Tool required: `Check the status of order 12345`
- Clarification required: `Cancel my order`
- Tool required: `What is the weather in Lahore?`
- No tool: `Explain what a neural network is in one sentence`

This is a portfolio/research demo, not a production action-execution service. Tool calls should be validated before real-world execution.
