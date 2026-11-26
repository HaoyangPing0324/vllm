"""Create full, small HF Muse Spark weights, a genuine PEFT adapter, and a merged oracle."""
import argparse
import json
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import MuseSparkConfig, MuseSparkForConditionalGeneration, PreTrainedTokenizerFast

PROMPTS = [[4, 9, 15, 26], [5, 11, 19], [22, 7, 30, 9, 25], [31, 19, 12, 5]]
NAMES = ["q_proj", "k_proj", "v_proj", "gate_proj"]


@torch.inference_mode()
def greedy(model):
    out = []
    for prompt in PROMPTS:
        ids = torch.tensor([prompt])
        for _ in range(5):
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            assert logits.isfinite().all()
            ids = torch.cat([ids, logits.argmax().reshape(1, 1)], dim=1)
        out.append(ids[0, len(prompt):].tolist())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    torch.manual_seed(51)
    torch.set_num_threads(1)
    text = dict(vocab_size=128, hidden_size=256, num_hidden_layers=2, head_dim=64,
                num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
                num_local_experts=16, moe_hidden_size=128, moe_intermediate_size=128,
                num_shared_experts=0, num_experts_per_tok=2,
                layer_types=["sliding_attention", "full_attention"], sliding_window=32,
                layer_rope_theta=[10000, 0], max_position_embeddings=128,
                mlp_layer_types=["sparse", "sparse"], qk_scale_factor=2.0,
                output_multiplier=1.0, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    vision = dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=2, patch_size=14, patch_temporal=2, merge_size=2,
                  pos_emb_height=8, pos_emb_width=8, layer_types=["window_attention","full_attention"],
                  rope_parameters={"rope_type":"axial","rope_theta":10000})
    audio = dict(hidden_sizes=[16, 32, 64, 128], depths=[2]*4, patch_sizes=[16, 6, 5, 4],
                 num_hidden_layers=4, out_hidden_size=256)
    config = MuseSparkConfig(text_config=text, vision_config=vision, audio_config=audio,
                             image_token_id=124, video_token_id=125, audio_token_id=126,
                             out_hidden_size=512, projector_hidden_size=128)
    config._attn_implementation = "eager"
    model = MuseSparkForConditionalGeneration(config).eval()
    fp32_weights = {name: p.detach().clone() for name, p in model.named_parameters()
                    if any(key in name for key in model._keep_in_fp32_modules_strict)}
    model.to(torch.bfloat16)
    for name, p in model.named_parameters():
        if name in fp32_weights:
            p.data = fp32_weights[name]
    words = {f"t{i}": i for i in range(128)}
    for i, token in ((0, "<pad>"), (1, "<bos>"), (2, "<eos>"), (3, "<unk>"),
                     (124,"<|patch|>"),(125,"<|video|>"),(126,"<|audio|>")):
        del words[f"t{i}"]
        words[token] = i
    tok = Tokenizer(WordLevel(words, unk_token="<unk>"))
    tok.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                                        bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    model.save_pretrained(root/"base")
    tokenizer.save_pretrained(root/"base")
    def reload(subdir):
        return MuseSparkForConditionalGeneration.from_pretrained(
            root/subdir, dtype=torch.bfloat16, attn_implementation="eager").eval()

    base_ids = greedy(reload("base"))
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=8, bias="none", target_modules=[], target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])).eval()
    generator = torch.Generator().manual_seed(62)
    for name, p in model.named_parameters():
        if ".lora_" in name:
            with torch.no_grad():
                p.normal_(std=0.25, generator=generator)
    model.save_pretrained(root/"adapter", save_embedding_layers=False)
    merged = model.merge_and_unload().eval()
    merged.save_pretrained(root/"merged")
    tokenizer.save_pretrained(root/"merged")
    result = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                  params=sum(p.numel() for p in merged.parameters()), prompts=PROMPTS,
                  base_ids=base_ids, merged_ids=greedy(reload("merged")),
                  adapter_ids=greedy(PeftModel.from_pretrained(reload("base"), root/"adapter").eval()))
    assert all(i != j for i, j in zip(result["base_ids"], result["merged_ids"]))
    (root/"reference.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
