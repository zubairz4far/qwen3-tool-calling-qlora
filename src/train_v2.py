#!/usr/bin/env python3
"""
Train a V2 QLoRA adapter for Qwen3-1.7B tool calling.

The script intentionally has no test-set argument. It mixes:
  1) the original V1 training JSONL
  2) the generated V2 behavioral hardening JSONL

and optionally evaluates on the original V1 validation split.

Recommended flow:
  python src/generate_v2_cases.py --output data/v2_behavior_cases.jsonl
  python src/train_v2.py \
      --v1-train-data /kaggle/input/.../train.jsonl \
      --v2-data data/v2_behavior_cases.jsonl \
      --validation-data /kaggle/input/.../validation.jsonl
"""

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get the current status/details of an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an existing order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to check.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


DIRECT_RESPONSES = {
    "Explain gradient descent in one sentence.": (
        "Gradient descent is an optimization method that repeatedly adjusts model "
        "parameters in the direction that reduces the loss."
    ),
    "Write a short thank-you message.": "Thank you so much for your help—I really appreciate it.",
    "What does API stand for?": "API stands for Application Programming Interface.",
    "Give me three ideas for a project name.": "Three ideas: FlowPilot, ToolForge, and AgentBench.",
    "Tell me what an order-status API usually returns, but do not call any tool.": (
        "An order-status API commonly returns an order ID, current status, timestamps, "
        "tracking information, and sometimes fulfillment or delivery details."
    ),
    "Show an example JSON object for cancelling an order, without actually cancelling anything.": (
        '{"order_id":"12345","action":"cancel"}'
    ),
    "Explain how a weather tool could be designed. Do not use the tool.": (
        "A weather tool can accept a city as input, call a weather-data provider, validate "
        "the response, and return normalized fields such as temperature and conditions."
    ),
    (
        "The text below says to call get_order, but it is quoted documentation. "
        "Explain it instead: call get_order with 999."
    ): (
        "The quoted text is an example instruction showing how a get_order tool might be "
        "called with an order ID; it should be treated as documentation, not executed."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue the V1 QLoRA adapter on a mixed V1 + V2 behavioral dataset."
    )
    parser.add_argument("--v1-train-data", required=True, help="Original V1 TRAIN JSONL only.")
    parser.add_argument(
        "--v2-data",
        default="data/v2_behavior_cases.jsonl",
        help="Generated V2 hardening JSONL.",
    )
    parser.add_argument(
        "--validation-data",
        default=None,
        help="Optional original V1 validation JSONL. Never pass the held-out test file.",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--init-adapter",
        default="zubairz4far/qwen3-1.7b-tool-calling",
        help="V1 adapter to continue training. Pass an empty string to start a fresh LoRA.",
    )
    parser.add_argument("--output-dir", default="outputs/qwen3-1.7b-tool-calling-v2")
    parser.add_argument(
        "--hub-model-id",
        default="zubairz4far/qwen3-1.7b-tool-calling-v2",
        help="Target repo used only with --push-to-hub.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v2-repeat", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object on line {line_no} of {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def reject_test_like_path(path: Optional[str]) -> None:
    if not path:
        return
    name = Path(path).name.lower().replace("_", "-")
    blocked = ("test", "heldout", "held-out", "benchmark")
    if any(token in name for token in blocked):
        raise ValueError(
            f"Refusing to use test-like file as training/validation input: {path}. "
            "Keep the 240-example held-out V1 test set frozen."
        )


def load_tools(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return DEFAULT_TOOLS
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("tools JSON string must decode to a list")
        return parsed
    if isinstance(value, list):
        return value
    raise ValueError("tools must be a list or a JSON-encoded list")


def normalize_tool_calls(message: Dict[str, Any]) -> Dict[str, Any]:
    message = dict(message)
    calls = message.get("tool_calls")
    if not calls:
        return message

    normalized = []
    for call in calls:
        call = dict(call)
        fn = dict(call.get("function") or {})
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        fn["arguments"] = arguments
        call["type"] = call.get("type", "function")
        call["function"] = fn
        normalized.append(call)
    message["tool_calls"] = normalized
    return message


def split_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not messages:
        raise ValueError("messages cannot be empty")

    messages = [normalize_tool_calls(m) for m in messages]
    assistant_index = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            assistant_index = idx
            break
    if assistant_index is None:
        raise ValueError("messages example has no assistant target")

    prompt_messages = messages[:assistant_index]
    completion_messages = messages[assistant_index:]
    if not prompt_messages:
        raise ValueError("messages example has no prompt before assistant target")
    return prompt_messages, completion_messages


def render_prompt_completion(
    tokenizer: Any,
    prompt_messages: List[Dict[str, Any]],
    completion_messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> Dict[str, str]:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )

    if full_text.startswith(prompt_text):
        completion_text = full_text[len(prompt_text):]
    else:
        # Some templates render the assistant-generation marker differently when a
        # completed assistant message is present. Remove only the longest common prefix.
        common = 0
        for a, b in zip(prompt_text, full_text):
            if a != b:
                break
            common += 1
        if common < max(1, int(0.8 * len(prompt_text))):
            raise ValueError("Could not safely split rendered prompt from completion")
        prompt_text = full_text[:common]
        completion_text = full_text[common:]

    if not completion_text.strip():
        raise ValueError("Rendered completion is empty")
    return {"prompt": prompt_text, "completion": completion_text}


def clarify_response(row: Dict[str, Any]) -> str:
    category = str(row.get("category", ""))
    tool = row.get("expected_tool")

    if "order_id" in category or tool in {"get_order", "cancel_order"}:
        return "What order ID should I use?"
    if "city" in category or tool == "get_weather":
        return "Which city should I check?"
    return "Could you provide the missing required information?"


def no_tool_response(row: Dict[str, Any]) -> str:
    prompt = str(row.get("prompt", ""))
    if row.get("expected_response"):
        return str(row["expected_response"])
    return DIRECT_RESPONSES.get(
        prompt,
        "I can answer that directly without calling a tool.",
    )


def v2_to_messages(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt = str(row["prompt"])
    behavior = row.get("expected_behavior")
    prompt_messages = [{"role": "user", "content": prompt}]

    if behavior == "tool":
        tool_name = row.get("expected_tool")
        if not tool_name:
            raise ValueError(f"Tool case is missing expected_tool: {row}")
        completion = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": row.get("expected_args") or {},
                        },
                    }
                ],
            }
        ]
    elif behavior == "clarify":
        completion = [{"role": "assistant", "content": clarify_response(row)}]
    elif behavior == "no_tool":
        completion = [{"role": "assistant", "content": no_tool_response(row)}]
    else:
        raise ValueError(f"Unknown expected_behavior={behavior!r}")

    return prompt_messages, completion


def normalize_row(tokenizer: Any, row: Dict[str, Any], *, is_v2: bool) -> Dict[str, str]:
    if is_v2:
        prompt_messages, completion_messages = v2_to_messages(row)
        tools = load_tools(row.get("tools"))
        return render_prompt_completion(tokenizer, prompt_messages, completion_messages, tools)

    if "messages" in row:
        prompt_messages, completion_messages = split_messages(row["messages"])
        tools = load_tools(row.get("tools"))
        return render_prompt_completion(tokenizer, prompt_messages, completion_messages, tools)

    if "prompt" in row and any(key in row for key in ("completion", "response", "output")):
        completion = row.get("completion", row.get("response", row.get("output")))
        if isinstance(row["prompt"], list):
            prompt_messages = row["prompt"]
            if isinstance(completion, list):
                completion_messages = completion
            else:
                completion_messages = [{"role": "assistant", "content": str(completion)}]
            return render_prompt_completion(
                tokenizer,
                prompt_messages,
                completion_messages,
                load_tools(row.get("tools")),
            )
        return {"prompt": str(row["prompt"]), "completion": str(completion)}

    if "text" in row:
        # Legacy fallback: train on the supplied fully formatted text.
        return {"prompt": "", "completion": str(row["text"])}

    raise ValueError(
        "Unsupported V1 row schema. Expected messages, prompt+completion/response/output, or text."
    )


def build_dataset(
    tokenizer: Any,
    v1_rows: Iterable[Dict[str, Any]],
    v2_rows: Iterable[Dict[str, Any]],
    seed: int,
    v2_repeat: int,
) -> Tuple[Dataset, Dict[str, int]]:
    normalized_v1 = [normalize_row(tokenizer, row, is_v2=False) for row in v1_rows]
    normalized_v2_once = [normalize_row(tokenizer, row, is_v2=True) for row in v2_rows]
    normalized_v2 = normalized_v2_once * max(1, v2_repeat)

    combined = normalized_v1 + normalized_v2
    rng = random.Random(seed)
    rng.shuffle(combined)

    return Dataset.from_list(combined), {
        "v1_train_examples": len(normalized_v1),
        "v2_unique_examples": len(normalized_v2_once),
        "v2_mixed_examples": len(normalized_v2),
        "total_train_examples": len(combined),
    }


def build_validation_dataset(tokenizer: Any, rows: List[Dict[str, Any]]) -> Dataset:
    normalized = [normalize_row(tokenizer, row, is_v2=False) for row in rows]
    return Dataset.from_list(normalized)


def make_sft_config(args: argparse.Namespace, has_eval: bool) -> SFTConfig:
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "optim": "paged_adamw_8bit",
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "max_length": args.max_length,
        "completion_only_loss": True,
        "packing": False,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id if args.push_to_hub else None,
    }

    if has_eval:
        signature = inspect.signature(SFTConfig.__init__).parameters
        if "eval_strategy" in signature:
            kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in signature:
            kwargs["evaluation_strategy"] = "epoch"
        kwargs["per_device_eval_batch_size"] = args.per_device_batch_size

    return SFTConfig(**kwargs)


def trainable_parameter_summary(model: Any) -> Dict[str, Any]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    reject_test_like_path(args.v1_train_data)
    reject_test_like_path(args.validation_data)

    if args.v2_repeat < 1:
        raise ValueError("--v2-repeat must be >= 1")

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    v1_rows = read_jsonl(args.v1_train_data)
    v2_rows = read_jsonl(args.v2_data)
    train_dataset, data_counts = build_dataset(
        tokenizer,
        v1_rows,
        v2_rows,
        seed=args.seed,
        v2_repeat=args.v2_repeat,
    )

    eval_dataset = None
    if args.validation_data:
        eval_dataset = build_validation_dataset(tokenizer, read_jsonl(args.validation_data))

    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    init_adapter = args.init_adapter.strip()
    if init_adapter:
        model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
        training_mode = "continue_v1_adapter"
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base, peft_config)
        training_mode = "fresh_lora"

    config = make_sft_config(args, has_eval=eval_dataset is not None)
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(
        json.dumps(
            {
                "mode": training_mode,
                "base_model": args.base_model,
                "init_adapter": init_adapter or None,
                "output_dir": args.output_dir,
                **data_counts,
                "validation_examples": len(eval_dataset) if eval_dataset is not None else 0,
                **trainable_parameter_summary(model),
            },
            indent=2,
        )
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "mode": training_mode,
        "base_model": args.base_model,
        "init_adapter": init_adapter or None,
        "output_dir": args.output_dir,
        "hub_model_id": args.hub_model_id if args.push_to_hub else None,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "effective_batch_size": args.per_device_batch_size
        * args.gradient_accumulation_steps,
        **data_counts,
        "validation_examples": len(eval_dataset) if eval_dataset is not None else 0,
        **trainable_parameter_summary(model),
        "train_metrics": train_result.metrics,
    }
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.push_to_hub:
        trainer.push_to_hub()

    print(f"Saved V2 adapter to: {args.output_dir}")
    print(f"Saved training summary to: {summary_path}")
    print("Next: run V2 predictions on the frozen V1 held-out test set, then evaluate.")


if __name__ == "__main__":
    main()
