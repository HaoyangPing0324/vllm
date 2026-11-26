"""Run vLLM's public offline generation API on the fixed full-architecture fixture."""
import argparse
import json
import random
import time
import os
from pathlib import Path

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("mode", choices=["base", "adapter", "merged", "unpatched"])
    parser.add_argument("report", type=Path)
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    weights = fixture / ("merged" if args.mode == "merged" else "base")
    print("RUNTIME", vllm.__version__, torch.__version__, torch.cuda.get_device_name(0),
          "mode", args.mode, flush=True)
    trace = os.getenv("TRACE") == "1"
    options = dict(model=str(weights), dtype="bfloat16", enforce_eager=True, tensor_parallel_size=int(os.environ.get("TP_SIZE", "2")),
                   seed=51, max_model_len=64, max_logprobs=128, max_num_seqs=4, max_num_batched_tokens=64,
                   gpu_memory_utilization=0.25, kv_cache_memory_bytes=64 << 20,
                   skip_mm_profiling=True, enable_prefix_caching=trace, enable_return_routed_experts=trace,
                   limit_mm_per_prompt={"image": 1, "audio": 1, "video": 1},
                   enable_lora=args.mode in ("adapter", "unpatched"))
    if options["enable_lora"]:
        options.update(lora_dtype=getattr(torch, os.getenv("FACTOR_DTYPE", "bfloat16")), max_loras=1, max_lora_rank=8, lora_target_modules=["experts"])
    llm = LLM(**options)
    req = LoRARequest("experts",1,str(fixture/"adapter")) if args.mode=="adapter" else None
    def prompts(seed,n):
        rng=random.Random(seed)
        return [[rng.randrange(4,124) for _ in range(rng.randint(4,12))] for _ in range(n)]
    def generate(batch, params, request):
        return llm.generate([{"prompt_token_ids": p} for p in batch],params,lora_request=request,use_tqdm=False)
    report={}
    def save():
        args.report.write_text(json.dumps(report))
    # Warm and measure fixed-length generation, excluding engine startup and adapter load.
    params=SamplingParams(temperature=0,min_tokens=8,max_tokens=8,ignore_eos=True)
    pool=prompts(275094,256)
    for i in range(0,32,4):
        generate(pool[i:i+4],params,req)
    for repeat in range(3):
        elapsed=[]
        decoded=[]
        for i in range(0,len(pool),4):
            start=time.perf_counter()
            outs=generate(pool[i:i+4],params,req)
            elapsed.append(time.perf_counter()-start)
            decoded += [[out.prompt_token_ids,list(out.outputs[0].token_ids)] for out in outs]
        report["perf"+str(repeat)]=elapsed
        if repeat==0: report["decode"]=decoded
        else: assert report["decode"]==decoded
        save()
    if os.getenv("PERF_ONLY")=="1": return
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=128)
    for seed in (70094,170094):
        pool=prompts(seed,2048)
        rows=[]
        for i in range(0,len(pool),4):
            for out in generate(pool[i:i+4],params,req):
                item=out.outputs[0]
                rows.append([out.prompt_token_ids,item.token_ids[0],[item.logprobs[0][j].logprob for j in range(128)]])
        report[str(seed)]=rows
        save()
        print("Scored",seed,len(rows),flush=True)
if __name__=="__main__":
    main()
