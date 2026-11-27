"""Shared V3 training/evaluation prompt and strict output protocol."""
import json
import re
from v3_data import EXPERIMENT, TOOLS, REGISTRY, dumps
from evaluate_predictions import exact_dict_match, _reject_duplicate_keys, _reject_nonfinite

def render_prompt(tokenizer, row):
    messages = [{"role":"system","content":EXPERIMENT["system_prompt"]}, *row["messages"]]
    return tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)

def target(row):
    if row["expected_behavior"] == "tool":
        return "<tool_call>\n" + dumps({"name": row["expected_tool"], "arguments":row["expected_args"]}) + "\n</tool_call>"
    return row["expected_response"]

def parse_output(text):
    text = text.strip()
    # No substring detection: documentation JSON is not a tool call.
    if "<tool_call>" not in text and "</tool_call>" not in text:
        return {"kind":"text", "text":text} if text else {"kind":"invalid"}
    m = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return {"kind":"invalid"}
    try:
        obj = json.loads(m.group(1), object_pairs_hook=_reject_duplicate_keys,
                         parse_constant=_reject_nonfinite)
        if not isinstance(obj, dict) or set(obj) != {"name","arguments"}:
            raise ValueError("Invalid call envelope")
        name, args = obj["name"], obj["arguments"]
        if not isinstance(name,str) or not isinstance(args,dict) or name not in REGISTRY:
            raise ValueError("Unknown tool or arguments")
        spec = REGISTRY[name]
        if set(args) != set(spec["required"]) or any(type(v) is not str or not v for v in args.values()):
            raise ValueError("Invalid arguments")
        return {"kind":"tool","name":name,"args":args}
    except (ValueError,TypeError):
        return {"kind":"invalid"}

def score(row, text):
    p = parse_output(text)
    expected = row["expected_behavior"]
    attempted = "<tool_call>" in text or "</tool_call>" in text
    if expected == "tool":
        correct = p["kind"] == "tool" and p["name"] == row["expected_tool"] and exact_dict_match(p["args"],row["expected_args"])
    else:
        # Conservative exact-text proxy, not semantic or human-judged answer quality.
        correct = p["kind"] == "text" and " ".join(p["text"].casefold().split()) == " ".join(row["expected_response"].casefold().split())
    return {"strict_pass":correct, "unexpected_tool_attempt":expected!="tool" and attempted,
            "parse_valid":p["kind"]!="invalid", "predicted_kind":p["kind"]}
