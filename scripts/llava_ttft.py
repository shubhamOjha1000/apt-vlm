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
import copy
import os
import sys
import time
from functools import partial

os.environ.setdefault("APT_ATTN_IMPL", "eager")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoProcessor, CLIPVisionModel, LlavaForConditionalGeneration

from scripts.forward_check import APT_SETTINGS, IMAGE_URLS, PATCH, IMG_SIZE, build_tokenizer, load_images
from src.models import vit_components
from src.models.patch_embed import TokenizedZeroConvPatchAttn
from src.models.vision_transformer import VisionTransformer

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CLIP_ID = "openai/clip-vit-large-patch14-336"  # LLaVA-1.5 keeps this encoder frozen: same weights
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

    def encode_apt(self, net, tokenizer, pixel_values, input_dict=None):
        keep = len(net.blocks) + 1 + self.layer if self.layer < 0 else self.layer
        return apt_hidden_states(self, net, tokenizer, pixel_values, {keep}, input_dict)[keep]

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


def apt_hidden_states(pipe, net, tokenizer, pixel_values, keep, input_dict=None):
    """APT image features (CLS dropped, raster order) after h blocks, for each h in `keep`.
    h = 0 is right after the pre-norm, matching HF CLIP's hidden_states[h].
    Every block is run, like LLaVA's own encoder, so both sides do the same number of layers.
    Patch sizes come from entropy unless a ready-made tokenizer output `input_dict` is given."""
    t0 = sync()
    if input_dict is None:
        maps = tokenizer.compute_importance_maps(pixel_values.float())
        input_dict = tokenizer(pixel_values, importance_maps=maps)
    d = input_dict
    pipe.tokenize_time = sync() - t0
    x, cu_seqlens, max_seqlen, _, _ = net.mixed_patch(pixel_values, net.pos_embed, d)
    x = net.norm_pre(x)
    is_image = d["output_mask"] != -1
    order = raster_order(d, tokenizer)
    out = {0: x[0][is_image][order]} if 0 in keep else {}
    for i, blk in enumerate(net.blocks):
        x = blk(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        if i + 1 in keep:
            out[i + 1] = x[0][is_image][order]
    return out


def hf_hidden_states(vision, pixel_values):
    """HF CLIP image features (CLS dropped) for every hidden state 0..depth."""
    return [h[0, 1:] for h in vision(pixel_values, output_hidden_states=True).hidden_states]


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def is_plain_fp16(module):
    """True if no layer of `module` is quantized and every weight is fp16."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return all(type(m) is nn.Linear for m in linears) and all(p.dtype == torch.float16 for p in module.parameters())


def apt_from_hf_clip(vision, num_scales, thresholds, dtype=torch.float16):
    """APT ViT carrying exactly the weights of an HF CLIPVisionModel (LLaVA's vision tower)."""
    cfg = vision.config
    assert cfg.hidden_act == "quick_gelu" and cfg.patch_size == PATCH and cfg.image_size == IMG_SIZE
    sd = {k: v.detach().float().cpu() for k, v in vision.state_dict().items()}
    anchor = "embeddings.patch_embedding.weight"
    prefix = next(k for k in sd if k.endswith(anchor))[:-len(anchor)]
    g = lambda k: sd[prefix + k]
    D, L = cfg.hidden_size, cfg.num_hidden_layers

    new = {
        "patch_embed.proj.weight": g("embeddings.patch_embedding.weight"),
        "cls_token": g("embeddings.class_embedding").view(1, 1, D),
        "pos_embed": g("embeddings.position_embedding.weight").unsqueeze(0),
        "norm_pre.weight": g("pre_layrnorm.weight"), "norm_pre.bias": g("pre_layrnorm.bias"),
        "norm.weight": g("post_layernorm.weight"), "norm.bias": g("post_layernorm.bias"),
    }
    for i in range(L):
        hf, apt = f"encoder.layers.{i}.", f"blocks.{i}."
        for a, h in [("norm1", "layer_norm1"), ("norm2", "layer_norm2"), ("attn.proj", "self_attn.out_proj"),
                     ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")]:
            for t in ("weight", "bias"):
                new[f"{apt}{a}.{t}"] = g(f"{hf}{h}.{t}")
        for t in ("weight", "bias"):
            new[f"{apt}attn.qkv.{t}"] = torch.cat([g(f"{hf}self_attn.{x}_proj.{t}") for x in "qkv"])

    net = VisionTransformer(
        img_size=IMG_SIZE, patch_size=PATCH, embed_dim=D, depth=L, num_heads=cfg.num_attention_heads,
        mlp_ratio=cfg.intermediate_size / D, num_classes=0, pre_norm=True,
        norm_layer=partial(nn.LayerNorm, eps=cfg.layer_norm_eps), act_layer=QuickGELU,
        mixed_patch_embed=partial(TokenizedZeroConvPatchAttn, patch_size=PATCH),
        num_scales=num_scales, thresholds=thresholds, weight_init="skip",
    )
    missing, unexpected = net.load_state_dict(new, strict=False)
    bad_missing = [k for k in missing if not k.startswith("mixed_patch.")]
    assert not bad_missing and not unexpected, f"weight mismatch: missing={bad_missing} unexpected={unexpected}"
    net.init_multiscale_patch_embed()  # after loading, as in ViTLitModule
    assert torch.count_nonzero(net.mixed_patch.zero_conv.weight) == 0
    return net.to(vision.device, dtype).eval()


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


def load_llava(load_4bit=False, device="cuda", auto_4bit=True):
    """LLaVA-1.5-7B (eager attention) + processor + Pipeline whose vision encoder is full fp16 CLIP.
    The LLM is 4-bit if asked, or (auto_4bit) if the GPU has < 20 GB. With auto_4bit=False and
    load_4bit=False everything is fp16 and a too-small GPU is an error."""
    assert torch.cuda.is_available(), "needs a GPU runtime"
    gpu = torch.cuda.get_device_properties(0)
    small = gpu.total_memory < 20 * 2 ** 30
    if small and not (load_4bit or auto_4bit):
        raise RuntimeError(f"{gpu.name} has {gpu.total_memory / 2 ** 30:.0f} GB; LLaVA-7B in fp16 needs ~16 GB "
                           "plus activations. Use an A100 or L4 runtime (or pass --load-4bit).")
    load_4bit = load_4bit or (auto_4bit and small)
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

    # The vision encoder must be full fp16 CLIP for both pipelines. If 4-bit loading quantized it anyway,
    # use a fresh fp16 copy of the same frozen OpenAI CLIP encoder instead.
    if is_plain_fp16(pipe.vision):
        print("vision encoder: LLaVA's own, fp16, not quantized")
    else:
        print("vision encoder: LLaVA's copy was quantized by the 4-bit load -> using fp16 " + CLIP_ID)
        pipe.vision = CLIPVisionModel.from_pretrained(
            CLIP_ID, torch_dtype=torch.float16, attn_implementation="eager").to(device).eval()
        assert is_plain_fp16(pipe.vision)

    report = {"vision encoder": pipe.vision, "projector": pipe.projector,
              "decoder (LLM)": part(model, "language_model"), "lm_head": model.get_output_embeddings()}
    print("precision: " + ", ".join(f"{k} {'fp16' if is_plain_fp16(m) else 'QUANTIZED/mixed'}" for k, m in report.items()))
    if not load_4bit:
        assert all(is_plain_fp16(m) for m in report.values()), "expected every part in plain fp16"
    return model, processor, pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-4bit", action="store_true", help="4-bit LLM (vision tower and projector stay fp16)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    device = "cuda"
    model, processor, pipe = load_llava(args.load_4bit, device)

    images = load_images()
    names = list(images)
    inputs = {}
    for n in names:
        enc = processor(images=images[n], text=PROMPT, return_tensors="pt")
        inputs[n] = (enc["input_ids"].to(device), enc["pixel_values"].to(device, torch.float16))
    assert inputs[names[0]][1].shape[-1] == IMG_SIZE

    # APT encoders, built from the exact weights of the vision encoder above, fp16.
    apt = {}
    for setting, (num_scales, thresholds) in APT_SETTINGS.items():
        apt[setting] = (apt_from_hf_clip(pipe.vision, num_scales, thresholds), build_tokenizer(num_scales, thresholds))

    encoders = {"Baseline": pipe.encode_baseline}
    for setting, (net, tok) in apt.items():
        encoders[setting] = (lambda net, tok: (lambda pv: pipe.encode_apt(net, tok, pv)))(net, tok)

    # Sanity check: APT with merging off == LLaVA's vision encoder.
    print("\n=== Sanity check: APT with merging off vs LLaVA's vision encoder ===")
    print("Pass/fail is decided in fp32. The fp16 rows are for information: compare APT's fp16 error")
    print("with LLaVA's own fp16-vs-fp32 error (last row); they should be about the same size.")
    no_merge_tok = build_tokenizer(2, [-1.0])
    net16 = apt[next(iter(APT_SETTINGS))][0]  # 2-scale net; merging is turned off by the tokenizer
    net32 = apt_from_hf_clip(pipe.vision, 2, [-1.0], dtype=torch.float32)
    vision32 = copy.deepcopy(pipe.vision).float()
    no_merge = lambda pv: pipe.encode_apt(net16, no_merge_tok, pv)
    depth = len(net16.blocks)
    target = depth + 1 + pipe.layer if pipe.layer < 0 else pipe.layer  # 23: the layer LLaVA uses
    show = sorted({0, 1, depth // 4, depth // 2, 3 * depth // 4, target, depth})
    min_cos = lambda a, b: F.cosine_similarity(a.float(), b.float(), dim=-1).min().item()

    all_ok = True
    for n in REAL_IMAGES:
        ids, pv = inputs[n]
        with torch.inference_mode():
            hf32 = hf_hidden_states(vision32, pv.float())
            hf16 = hf_hidden_states(pipe.vision, pv)
            apt32 = apt_hidden_states(pipe, net32, no_merge_tok, pv.float(), set(show))
            apt16 = apt_hidden_states(pipe, net16, no_merge_tok, pv, set(show))
        ok = apt32[target].shape == hf32[target].shape == (576, net16.embed_dim) \
            and min_cos(apt32[target], hf32[target]) > 0.999
        all_ok &= ok
        same = torch.equal(pipe.answer(no_merge, ids, pv), pipe.answer(pipe.encode_baseline, ids, pv))
        print(f"[{'ok' if ok else 'FAIL'}] {n}: {apt32[target].shape[0]} tokens, same answer in fp16 = {same}")
        print(f"   {'min cosine per token, after block':<36}" + "".join(f"{h:>8}" for h in show) + f"   (LLaVA uses {target})")
        for label, a, b in [("APT fp32   vs LLaVA fp32", apt32, hf32),
                            ("APT fp16   vs LLaVA fp32", apt16, hf32),
                            ("LLaVA fp16 vs LLaVA fp32", hf16, hf32)]:
            print(f"   {label:<36}" + "".join(f"{min_cos(a[h], b[h]):>8.4f}" for h in show))
    del vision32, net32
    torch.cuda.empty_cache()
    assert all_ok, "APT with merging off does not reproduce LLaVA's vision features in fp32"

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
