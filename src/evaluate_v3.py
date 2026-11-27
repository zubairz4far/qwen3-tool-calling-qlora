"""Run a real model on V3 development data; save raw predictions and category metrics."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from v3_data import EXPERIMENT as C, read_verified
from v3_protocol import render_prompt, score

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--base-revision",required=True)
    p.add_argument("--adapter",help="Local candidate or Hugging Face adapter ID; omit for base")
    p.add_argument("--adapter-revision",help="Required for remote adapter")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    import re
    if not re.fullmatch(r"[0-9a-f]{40}",a.base_revision):
        p.error("Base revision must be immutable SHA")
    if a.adapter and not Path(a.adapter).is_dir() and not re.fullmatch(r"[0-9a-f]{40}",a.adapter_revision or ""):
        p.error("Remote adapter requires immutable revision")
    if a.output.exists():
        p.error("Refusing to overwrite evidence")
    rows=read_verified(a.data,"dev")
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    tok=AutoTokenizer.from_pretrained(C["base_model"],revision=a.base_revision)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(C["base_model"],revision=a.base_revision,quantization_config=quant,device_map={"":0})
    if a.adapter:
        model=PeftModel.from_pretrained(model,a.adapter,revision=a.adapter_revision)
    model.eval()
    counts=defaultdict(lambda:{"n":0,"strict_pass":0,"unexpected_tool_attempt":0,"parse_valid":0})
    predictions=[]
    for row in rows:
        ids=tok(render_prompt(tok,row),return_tensors="pt",add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            out=model.generate(**ids,max_new_tokens=C["max_new_tokens"],do_sample=False,pad_token_id=tok.eos_token_id)
        generated=out[0,ids["input_ids"].shape[1]:].tolist()
        while generated and generated[-1] in {tok.eos_token_id,tok.pad_token_id}:
            generated.pop()
        text=tok.decode(generated,skip_special_tokens=False)
        s=score(row,text)
        for category in ("all",row["category"]):
            c=counts[category]
            c["n"]+=1
            for k in ("strict_pass","unexpected_tool_attempt","parse_valid"):
                c[k]+=int(s[k])
        predictions.append({"id":row["id"],"category":row["category"],"raw_output":text,**s})
    a.output.mkdir(parents=True)
    (a.output/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in predictions))
    report={"split":"development","base_revision":a.base_revision,"adapter":a.adapter,"adapter_revision":a.adapter_revision,
            "non_tool_metric":"normalized reference exact match, not semantic quality",
            "categories":{k:{**v,"strict_pass_rate":v["strict_pass"]/v["n"]} for k,v in counts.items()}}
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
