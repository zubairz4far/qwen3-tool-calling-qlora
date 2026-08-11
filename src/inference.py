import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a Qwen3-1.7B QLoRA adapter")
    parser.add_argument("--adapter", required=True, help="Hugging Face adapter repo or local path")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tools", help="Optional JSON file containing tool schemas")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_tools(path):
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    tools = load_tools(args.tools)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools is not None:
        template_kwargs["tools"] = tools

    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
