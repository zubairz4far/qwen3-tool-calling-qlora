import json
import re

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER = "zubairz4far/qwen3-1.7b-tool-calling"

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get the current status of an order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


def classify_output(text: str):
    lowered = text.lower()
    if any(token in lowered for token in ["<tool_call>", '"name"', "function"]):
        decision = "Tool call"
    elif any(token in lowered for token in ["which order", "order id", "please provide", "need the", "what city"]):
        decision = "Clarification"
    else:
        decision = "Direct response / no tool"
    return decision


def extract_json(text: str):
    candidates = re.findall(r"\{.*\}", text, re.DOTALL)
    for candidate in candidates:
        try:
            return json.dumps(json.loads(candidate), indent=2)
        except Exception:
            continue
    return "No standalone JSON object detected."


def run_demo(prompt, tools_json):
    try:
        tools = json.loads(tools_json) if tools_json.strip() else DEFAULT_TOOLS
    except json.JSONDecodeError as exc:
        return "Invalid tools JSON", str(exc), ""

    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "tools": tools}
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return classify_output(text), text, extract_json(text)


with gr.Blocks(title="Qwen3-1.7B Tool-Calling QLoRA") as demo:
    gr.Markdown(
        "# Qwen3-1.7B Tool-Calling QLoRA\n"
        "Interactive portfolio demo for routing, clarification, no-tool behavior, and structured function calls."
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="User request",
            value="Check the status of order 12345",
            lines=3,
        )
        tools = gr.Code(
            label="Tool schemas (JSON)",
            language="json",
            value=json.dumps(DEFAULT_TOOLS, indent=2),
        )

    run = gr.Button("Run model", variant="primary")
    decision = gr.Textbox(label="Behavior classification")
    raw = gr.Textbox(label="Raw model output", lines=8)
    parsed = gr.Code(label="Detected JSON", language="json")

    gr.Examples(
        examples=[
            ["Check the status of order 12345"],
            ["Cancel my order"],
            ["What is the weather in Lahore?"],
            ["Explain what a neural network is in one sentence"],
        ],
        inputs=[prompt],
    )

    run.click(run_demo, inputs=[prompt, tools], outputs=[decision, raw, parsed])


demo.queue().launch()
