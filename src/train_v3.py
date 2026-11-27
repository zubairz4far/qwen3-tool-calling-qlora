"""Train a candidate only; no final evaluation or publication occurs here."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from v3_data import EXPERIMENT as C, read_verified, validate
from v3_protocol import render_prompt, target

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-revision", required=True, help="Full immutable Hugging Face commit SHA")
    p.add_argument("--adapter-revision", required=True, help="Full immutable V1 adapter commit SHA")
    a = p.parse_args()
    import re
    for revision in (a.base_revision,a.adapter_revision):
        if not re.fullmatch(r"[0-9a-f]{40}",revision):
            p.error("Model revisions must be full 40-character lowercase commit SHAs")
    train, dev = read_verified(a.data,"train"), read_verified(a.data,"dev")
    if validate(train,"train") & validate(dev,"dev"):
        raise ValueError("Split overlap")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Output directory must be empty")
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Expose exactly one CUDA GPU (CUDA_VISIBLE_DEVICES=0)")
    set_seed(C["seed"])
    tokenizer = AutoTokenizer.from_pretrained(C["base_model"],revision=a.base_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    def convert(rows):
        records=[]
        for r in rows:
            prompt=render_prompt(tokenizer,r)
            completion=target(r)+tokenizer.eos_token
            # Fail before training rather than silently truncating supervision.
            prompt_ids=tokenizer(prompt,add_special_tokens=False)["input_ids"]
            full_ids=tokenizer(prompt+completion,add_special_tokens=False)["input_ids"]
            if full_ids[:len(prompt_ids)]!=prompt_ids or len(full_ids)>C["max_length"]:
                raise ValueError("Prompt boundary or context limit violated: "+r["id"])
            records.append({"prompt":prompt,"completion":completion})
        return Dataset.from_list(records)
    training, validation = convert(train), convert(dev)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModelForCausalLM.from_pretrained(C["base_model"],revision=a.base_revision,
                                            quantization_config=quant,device_map={"":0})
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    model=PeftModel.from_pretrained(base,C["initial_adapter"],revision=a.adapter_revision,is_trainable=True)
    model.print_trainable_parameters()
    config=SFTConfig(output_dir=str(a.output),num_train_epochs=C["epochs"],learning_rate=C["learning_rate"],
        per_device_train_batch_size=C["batch_size"],per_device_eval_batch_size=C["batch_size"],
        gradient_accumulation_steps=C["gradient_accumulation"],max_length=C["max_length"],
        completion_only_loss=True,packing=False,gradient_checkpointing=True,fp16=True,bf16=False,
        optim="paged_adamw_8bit",lr_scheduler_type="cosine",warmup_ratio=0.05,
        eval_strategy="epoch",save_strategy="epoch",report_to="none",seed=C["seed"],data_seed=C["seed"])
    trainer=SFTTrainer(model=model,args=config,train_dataset=training,eval_dataset=validation,processing_class=tokenizer)
    result=trainer.train()
    trainer.save_model(str(a.output))
    tokenizer.save_pretrained(str(a.output))
    summary={"status":"candidate_unvalidated","base_revision":a.base_revision,"adapter_revision":a.adapter_revision,
        "dataset_manifest_sha256":hashlib.sha256((a.data/"manifest.json").read_bytes()).hexdigest(),
        "experiment":C,"gpu":torch.cuda.get_device_name(0),"metrics":result.metrics,
        "versions":{n:importlib.metadata.version(n) for n in ["torch","transformers","trl","peft","bitsandbytes","datasets","accelerate"]}}
    (a.output/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__=="__main__":
    main()
