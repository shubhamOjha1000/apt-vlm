"""TTFT (time to first token) of LLaVA-1.5-7B: normal vision encoder vs APT, eager attention.

No training, no accuracy. For each image + prompt we time, on the GPU:
    vision encoder -> projector -> LLM prefill -> first token
for
  * Baseline : LLaVA's own CLIP ViT-L/14-336 (576 image tokens), and
  * APT      : the same CLIP weights run through the APT ViT (fewer image tokens), for the
               two APT settings used in forward_check.py.
APT time includes computing the entropy maps and building the patch groups.

Both pipelines share everything after the vision encoder (projector, splicing the image
tokens into the prompt, LLM prefill), so the TTFT difference comes only from the encoder
and the number of image tokens.

Sanity check before timing: APT with merging turned off must give the same 576 image
features as LLaVA's own encoder and the same answer.

Usage (Colab):  python scripts/llava_ttft.py           # fp16 if the GPU has >= 20 GB, else 4-bit LLM
                python scripts/llava_ttft.py --load-4bit
"""
import argparse
import os
import sys
import time

os.environ.setdefault("APT_ATTN_IMPL", "eager")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import timm
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration

from scripts.forward_check import (
    APT_SETTINGS, IMAGE_URLS, TIMM_NAME, PATCH, IMG_SIZE, build_apt, build_tokenizer, load_images,
)
from src.models import vit_components

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PROMPT = "USER: <image>\nDescribe this image in one sentence. ASSISTANT:"
REAL_IMAGES = list(IMAGE_URLS)  # the synthetic gradient is reported but left out of the mean


def part(model, name):
    """LLaVA submodule across transformers versions (model.X or model.model.X)."""
    m = getattr(model, name, None)
    return m if m is not None else getattr(model.model, name)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


class Pipeline:
    def __init__(self, model):
        self.model = model
        self.vision = part(model, "vision_tower")
        self.projector = part(model, "multi_modal_projector")
        self.embed_tokens = model.get_input_embeddings()
        cfg = model.config
        self.layer = cfg.vision_feature_layer  # -2 for LLaVA-1.5
        assert cfg.vision_feature_select_strategy == "default"  # drop CLS
        self.image_token_id = getattr(cfg, "image_token_id", None) or cfg.image_token_index

    # Vision encoders: both return (num_image_tokens, 1024) features from the same layer, CLS dropped.
    def encode_baseline(self, pixel_values):
        out = self.vision(pixel_values, output_hidden_states=True)
        return out.hidden_states[self.layer][0, 1:]

    def encode_apt(self, net, tokenizer, pixel_values):
        t0 = sync()
        maps = tokenizer.compute_importance_maps(pixel_values.float())
        d = tokenizer(pixel_values, importance_maps=maps)
        self.tokenize_time = sync() - t0
        x, cu_seqlens, max_seqlen, _, _ = net.mixed_patch(pixel_values, net.pos_embed, d)
        x = net.norm_pre(x)
        # hidden_states[h] = output after h blocks (h = 0 is after the pre-norm), as in HF CLIP.
        # Run every block, like LLaVA's own encoder does, so both sides do the same number of layers.
        keep = len(net.blocks) + 1 + self.layer if self.layer < 0 else self.layer
        feats = x if keep == 0 else None
        for i, blk in enumerate(net.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            if i + 1 == keep:
                feats = x
        feats = feats[0][d["output_mask"] != -1]
        return feats[raster_order(d, tokenizer)]

    def splice(self, input_ids, image_feats):
        """Prompt embeddings with the image-token run replaced by `image_feats` (any length)."""
        pos = (input_ids[0] == self.image_token_id).nonzero().squeeze(1)
        start, end = pos[0].item(), pos[-1].item() + 1
        assert end - start == len(pos), "image tokens are not one contiguous run"
        text = self.embed_tokens(input_ids)
        image = self.projector(image_feats.unsqueeze(0)).to(text.dtype)
        return torch.cat([text[:, :start], image, text[:, end:]], dim=1)

    @torch.inference_mode()
    def timed_first_token(self, encode, input_ids, pixel_values):
        """Returns (first token id, number of image tokens, prefill length, {stage: seconds}).
        TTFT = vision + projector + prefill; "tokenize" (APT entropy + patch grouping) is part of vision."""
        self.tokenize_time = 0.
        t0 = sync()
        feats = encode(pixel_values)
        t1 = sync()
        embeds = self.splice(input_ids, feats)
        t2 = sync()
        logits = self.model(inputs_embeds=embeds, use_cache=True).logits
        first = logits[0, -1].argmax().item()
        t3 = sync()
        stages = {"vision": t1 - t0, "projector": t2 - t1, "prefill": t3 - t2, "tokenize": self.tokenize_time}
        return first, feats.shape[0], embeds.shape[1], stages

    @torch.inference_mode()
    def answer(self, encode, input_ids, pixel_values, max_new_tokens=40):
        embeds = self.splice(input_ids, encode(pixel_values))
        out = self.model.generate(
            inputs_embeds=embeds, attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        return out[0]


def raster_order(d, tokenizer):
    """Order APT tokens by the top-left 14 px cell they cover (row-major), like the baseline's raster order.
    The tokenizer emits all 14 px tokens first, then 28 px, then 56 px."""
    grid = tokenizer.image_size // tokenizer.base_patch_size
    keys = []
    for i in range(tokenizer.num_scales):
        s = tokenizer.base_patch_size * 2 ** i
        g, k = tokenizer.image_size // s, 2 ** i
        flat = d[f"pos_embed_mask_{s}"][0].nonzero().squeeze(1)
        keys.append((flat // g) * k * grid + (flat % g) * k)
    return torch.cat(keys).argsort()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-4bit", action="store_true", help="4-bit LLM (vision tower and projector stay fp16)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU runtime"
    device = "cuda"
    gpu = torch.cuda.get_device_properties(0)
    load_4bit = args.load_4bit or gpu.total_memory < 20 * 2 ** 30
    print(f"GPU={gpu.name} ({gpu.total_memory / 2 ** 30:.0f} GB)  LLM={'4-bit' if load_4bit else 'fp16'}  "
          f"attention=eager (LLaVA) / {vit_components.ATTN_IMPL} (APT)  "
          f"torch={torch.__version__} transformers={transformers.__version__} timm={timm.__version__}")

    kwargs = dict(torch_dtype=torch.float16, attn_implementation="eager", device_map={"": 0})
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
        )
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    pipe = Pipeline(model)

    images = load_images()
    names = list(images)
    inputs = {}
    for n in names:
        enc = processor(images=images[n], text=PROMPT, return_tensors="pt")
        inputs[n] = (enc["input_ids"].to(device), enc["pixel_values"].to(device, torch.float16))
    assert inputs[names[0]][1].shape[-1] == IMG_SIZE

    # APT encoders: OpenAI CLIP ViT-L/14-336 weights (the encoder LLaVA-1.5 uses), fp16.
    ref = timm.create_model(TIMM_NAME, pretrained=True, num_classes=0).eval()
    apt = {}
    for setting, (num_scales, thresholds) in APT_SETTINGS.items():
        net = build_apt(ref, num_scales, thresholds, "cpu").to(device, torch.float16)
        apt[setting] = (net, build_tokenizer(num_scales, thresholds))
    no_merge_tok = build_tokenizer(2, [-1.0])
    no_merge_net = apt[next(iter(APT_SETTINGS))][0]  # 2-scale net, merging off via the tokenizer
    del ref

    encoders = {"Baseline": pipe.encode_baseline}
    for setting, (net, tok) in apt.items():
        encoders[setting] = (lambda net, tok: (lambda pv: pipe.encode_apt(net, tok, pv)))(net, tok)

    # Sanity check: APT with merging off == LLaVA's own encoder.
    print("\n=== Sanity check: APT with merging off vs LLaVA's own vision encoder ===")
    no_merge = lambda pv: pipe.encode_apt(no_merge_net, no_merge_tok, pv)
    all_ok = True
    for n in REAL_IMAGES:
        ids, pv = inputs[n]
        with torch.inference_mode():
            a, b = no_merge(pv).float(), pipe.encode_baseline(pv).float()
        cos = F.cosine_similarity(a, b, dim=-1).min().item()
        same = torch.equal(pipe.answer(no_merge, ids, pv), pipe.answer(pipe.encode_baseline, ids, pv))
        ok = a.shape == b.shape == (576, 1024) and cos > 0.999
        all_ok &= ok
        print(f"[{'ok' if ok else 'FAIL'}] {n:<18} tokens={a.shape[0]}  min cosine per token={cos:.5f}  "
              f"rel. max diff={(a - b).abs().max().item() / b.abs().max().item():.1e}  same answer={same}")
    assert all_ok, "APT with merging off does not reproduce LLaVA's vision features"

    # TTFT.
    print(f"\n=== TTFT: {args.warmup} warm-up + {args.repeats} timed runs per image, median reported ===")
    results = {e: {} for e in encoders}
    for n in names:
        ids, pv = inputs[n]
        for e, encode in encoders.items():
            for _ in range(args.warmup):
                pipe.timed_first_token(encode, ids, pv)
            runs = [pipe.timed_first_token(encode, ids, pv) for _ in range(args.repeats)]
            first, n_img, n_prefill, _ = runs[0]
            stages = {k: float(np.median([r[3][k] for r in runs])) for k in runs[0][3]}
            ttft = float(np.median([r[3]["vision"] + r[3]["projector"] + r[3]["prefill"] for r in runs]))
            results[e][n] = dict(tokens=n_img, prefill=n_prefill, ttft=ttft, stages=stages, first=first)
    torch.cuda.empty_cache()

    # Report.
    cols = list(encoders)
    print("\n" + "=" * 110)
    print("IMAGE TOKENS and TTFT (ms).  APT columns also show speedup vs baseline.")
    print("=" * 110)
    print(f"{'image':<20}" + "".join(f"{c[:34]:>36}" for c in cols))
    for n in names:
        base = results["Baseline"][n]["ttft"]
        row = f"{n:<20}"
        for c in cols:
            r = results[c][n]
            cell = f"{r['tokens']} tok  {1000 * r['ttft']:.1f} ms"
            if c != "Baseline":
                cell += f"  ({base / r['ttft']:.2f}x)"
            row += f"{cell:>36}"
        print(row)
    row = f"{'MEAN (real photos)':<20}"
    for c in cols:
        tok = np.mean([results[c][n]["tokens"] for n in REAL_IMAGES])
        ttft = np.mean([results[c][n]["ttft"] for n in REAL_IMAGES])
        base = np.mean([results["Baseline"][n]["ttft"] for n in REAL_IMAGES])
        cell = f"{tok:.0f} tok  {1000 * ttft:.1f} ms" + ("" if c == "Baseline" else f"  ({base / ttft:.2f}x)")
        row += f"{cell:>36}"
    print(row)

    print("\nTTFT breakdown, mean over real photos (ms):")
    print(f"{'':<36}{'vision':>10}{'(entropy+tok)':>15}{'projector':>11}{'LLM prefill':>13}{'TTFT':>9}{'prefill len':>13}")
    for c in cols:
        st = {k: 1000 * np.mean([results[c][n]["stages"][k] for n in REAL_IMAGES])
              for k in ("vision", "tokenize", "projector", "prefill")}
        plen = np.mean([results[c][n]["prefill"] for n in REAL_IMAGES])
        total = st["vision"] + st["projector"] + st["prefill"]
        print(f"{c:<36}{st['vision']:>10.1f}{st['tokenize']:>15.1f}{st['projector']:>11.1f}{st['prefill']:>13.1f}"
              f"{total:>9.1f}{plen:>13.0f}")

    print("\nAnswers (greedy, for a quick look only; no accuracy is measured):")
    for n in REAL_IMAGES:
        ids, pv = inputs[n]
        print(f"  {n}:")
        for c, encode in encoders.items():
            text = processor.decode(pipe.answer(encode, ids, pv), skip_special_tokens=True).strip()
            print(f"    {c[:34]:<34} {text}")


if __name__ == "__main__":
    main()
