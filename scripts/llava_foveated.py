"""LLaVA-1.5-7B answers with gaze-foveated patching (Steps 1-3) vs normal LLaVA. No training.

For each image:
  * Baseline : LLaVA's own CLIP ViT-L/14-336, 576 image tokens.
  * Foveated : same CLIP weights through APT (patch aggregation Eq. 2 + positional interpolation),
               patch sizes from gaze (14 px fovea, 28 px middle ring, 56 px periphery), for a few gaze points.
Everything after the vision encoder (projector, LLM) is LLaVA's own and shared by both.

Before that, a check: with the rings pushed to infinity (every patch 14 px) the foveated path must give
LLaVA's own image features.

Usage (Colab):  python scripts/llava_foveated.py [--a 10 40 160] [--fov 110] [--num-scales 3] [--max-new-tokens 60]
                python scripts/llava_foveated.py --image-dir uploads   # your own images instead of the COCO samples
"""
import argparse
import json
import os
import sys

os.environ.setdefault("APT_ATTN_IMPL", "eager")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.forward_check import CLIP_MEAN, CLIP_STD, load_images
from scripts.foveation_demo import hstack, overlay
from scripts.llava_ttft import apt_from_hf_clip, load_llava
from src.models.foveation import FoveaConfig, FoveatedTokenizer, layout_from_input_dict

GAZES = {"centre": (0.5, 0.5), "corner": (0.1, 0.1), "right": (0.75, 0.4)}
PROMPTS = {
    "describe": "USER: <image>\nDescribe this image in one sentence. ASSISTANT:",
    "objects": "USER: <image>\nList the objects you can see in this image. ASSISTANT:",
}
OUT_DIR = "outputs/llava_foveated"


def vstack(images, pad=6):
    out = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images) + pad * (len(images) - 1)), "white")
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height + pad
    return out


def load_gazes(path):
    """Gaze points from a JSON file: {image name or "*": {label: [u, v], ...}}.
    u, v in [0, 1] of the 336x336 crop LLaVA sees (u to the right, v down). "*" applies to images
    without their own entry; images with neither use the default centre / corner / right."""
    if path is None:
        return {"*": GAZES}
    with open(path) as f:
        spec = json.load(f)
    for name, gazes in spec.items():
        assert isinstance(gazes, dict) and gazes, f"{name}: expected {{label: [u, v]}}"
        for label, uv in gazes.items():
            assert len(uv) == 2 and all(0 <= c <= 1 for c in uv), f"{name}/{label}: u, v must be in [0, 1], got {uv}"
    return spec


def gazes_for(spec, name):
    return {g: tuple(uv) for g, uv in spec.get(name, spec.get("*", GAZES)).items()}


def load_image_dir(path):
    """All images in a folder, by file name (without extension). LLaVA's processor resizes and centre-crops them."""
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")
    try:  # iPhone / Mac photos are often HEIC
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
    assert files, f"no images ({', '.join(exts)}) found in {path}"
    return {os.path.splitext(f)[0]: Image.open(os.path.join(path, f)).convert("RGB") for f in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-4bit", action="store_true", help="4-bit LLM, only for GPUs under 20 GB (default: all fp16)")
    ap.add_argument("--num-scales", type=int, default=3)
    ap.add_argument("--a", type=float, nargs="+", default=[40.0], help="one or more fall-off values, e.g. --a 10 40 160")
    ap.add_argument("--fov", type=float, default=110.0, help="horizontal field of view of the images, degrees")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--image-dir", default=None, help="folder of your own images (default: built-in COCO samples)")
    ap.add_argument("--gazes", default=None, help="JSON file of gaze points (see load_gazes); default: centre, corner, right")
    args = ap.parse_args()
    device = "cuda"

    model, processor, pipe = load_llava(args.load_4bit, device, auto_4bit=False)  # fp16 everywhere unless --load-4bit

    # Foveated encoder: APT ViT carrying LLaVA's vision weights; layout from gaze (Steps 1-3), one per a.
    net = apt_from_hf_clip(pipe.vision, args.num_scales, [0.5] * (args.num_scales - 1))
    img_size = pipe.vision.config.image_size
    a_values = list(dict.fromkeys(args.a))  # keep order, drop repeats
    fov = {}
    for a in a_values:
        cfg = FoveaConfig.for_encoder(net, num_scales=args.num_scales, img_size=img_size, fov_deg=args.fov, a=a)
        fov[a] = (cfg, FoveatedTokenizer(cfg, mean=CLIP_MEAN, std=CLIP_STD))
        print(f"Foveation a={a:g}: p={cfg.p}, patch sizes {cfg.sizes} px, R={cfg.R_deg:.0f} deg, "
              f"rings at " + ", ".join(f"{r:.1f}" for r in cfg.rings_deg) + " deg")

    def foveated(tok, gaze):
        return lambda pv: pipe.encode_apt(net, tok, pv, input_dict=tok(pv, [gaze]))

    images = load_image_dir(args.image_dir) if args.image_dir else load_images()
    gaze_spec = load_gazes(args.gazes)
    unknown = [k for k in gaze_spec if k != "*" and k not in images]
    assert not unknown, f"gaze entries for unknown images {unknown}; images are {list(images)}"
    names = list(images)
    inputs = {}
    for n in names:
        for p, text in PROMPTS.items():
            enc = processor(images=images[n], text=text, return_tensors="pt")
            inputs[n, p] = (enc["input_ids"].to(device), enc["pixel_values"].to(device, torch.float16))

    # Check: no foveation (rings at infinity -> all 14 px) == LLaVA's own image features.
    cfg_off = FoveaConfig.for_encoder(net, num_scales=args.num_scales, img_size=img_size, fov_deg=args.fov)
    cfg_off.rings_deg = [float("inf")] * (args.num_scales - 1)
    tok_off = FoveatedTokenizer(cfg_off, mean=CLIP_MEAN, std=CLIP_STD)
    print("\n=== Check: foveation turned off vs LLaVA's own vision encoder (fp16) ===")
    for n in names:
        _, pv = inputs[n, "describe"]
        with torch.inference_mode():
            a = pipe.encode_apt(net, tok_off, pv, input_dict=tok_off(pv, [(0.5, 0.5)])).float()
            b = pipe.encode_baseline(pv).float()
        cos = F.cosine_similarity(a, b, dim=-1)
        print(f"  {n:<20} tokens {a.shape[0]} vs {b.shape[0]}   token cosine mean {cos.mean():.4f}, min {cos.min():.4f}")
        assert a.shape == b.shape and cos.mean() > 0.95, "foveated path with foveation off does not match LLaVA"

    # Answers, per image at its own gaze points, for every a.
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n=== Answers (greedy, max {args.max_new_tokens} new tokens) ===")
    for n in names:
        gazes = gazes_for(gaze_spec, n)
        _, pv = inputs[n, "describe"]
        layouts = {}
        with torch.inference_mode():
            for a, (cfg, tok) in fov.items():
                for g, gaze in gazes.items():
                    layouts[a, g] = sorted(layout_from_input_dict(tok(pv, [gaze]), cfg)[0])
        sizes = fov[a_values[0]][0].sizes
        counts = {k: [sum(1 for *_, s in l if s == size) for size in sizes] for k, l in layouts.items()}

        print(f"\n### {n}")
        print(f"  image tokens (baseline 576), per size {sizes} px:")
        for g, (u, v) in gazes.items():
            print(f"    gaze {g} at (u={u:.2f}, v={v:.2f}): " + ", ".join(
                f"a={a:g} -> {sum(counts[a, g])} [" + "/".join(map(str, counts[a, g])) + "]" for a in a_values))

        for p in PROMPTS:
            ids, pv = inputs[n, p]
            print(f"  [{p}]")
            rows = [("baseline (576)", pipe.encode_baseline, None)]
            rows += [(f"gaze {g}, a={a:g} ({sum(counts[a, g])})", foveated(fov[a][1], gaze), (g, tuple(layouts[a, g])))
                     for g, gaze in gazes.items() for a in a_values]
            width = max(len(label) for label, *_ in rows) + 2
            done = {}  # same gaze + same layout -> same answer: reuse it
            for label, encode, key in rows:
                if key is not None and key in done:
                    text = done[key][0] + f"   (same layout as {done[key][1]})"
                else:
                    out = pipe.answer(encode, ids, pv, max_new_tokens=args.max_new_tokens)
                    text = processor.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
                    if key is not None:
                        done[key] = (text, label.split(" (")[0])
                print(f"    {label:<{width}}{text}")

        # What the model was given: one row per a, one panel per gaze.
        _, pv = inputs[n, "describe"]
        shown = Image.fromarray((pv[0].float().cpu() * torch.tensor(CLIP_STD).view(3, 1, 1)
                                 + torch.tensor(CLIP_MEAN).view(3, 1, 1)).clamp(0, 1).mul(255).byte()
                                .permute(1, 2, 0).numpy())
        grid = [hstack([overlay(shown, fov[a][0], gaze, layouts[a, g]) for g, gaze in gazes.items()]) for a in a_values]
        vstack(grid).save(os.path.join(OUT_DIR, f"{n}.png"))
        with open(os.path.join(OUT_DIR, f"{n}.gazes.txt"), "w") as f:
            f.write("rows: " + ", ".join(f"a={a:g}" for a in a_values) + "; columns: " + ", ".join(gazes))
    print(f"\nSaved overlays to {OUT_DIR}/ (rows = a values, columns = gaze points, in the order listed above)")


if __name__ == "__main__":
    main()
