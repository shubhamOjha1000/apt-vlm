"""LLaVA-1.5-7B answers with gaze-foveated patching (Steps 1-3) vs normal LLaVA. No training.

For each image:
  * Baseline : LLaVA's own CLIP ViT-L/14-336, 576 image tokens.
  * Foveated : same CLIP weights through APT (patch aggregation Eq. 2 + positional interpolation),
               patch sizes from gaze (14 px fovea, 28 px middle ring, 56 px periphery), for a few gaze points.
Everything after the vision encoder (projector, LLM) is LLaVA's own and shared by both.

Before that, a check: with the rings pushed to infinity (every patch 14 px) the foveated path must give
LLaVA's own image features.

Usage (Colab):  python scripts/llava_foveated.py [--a 40] [--fov 110] [--num-scales 3] [--max-new-tokens 60]
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
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--num-scales", type=int, default=3)
    ap.add_argument("--a", type=float, default=40.0)
    ap.add_argument("--fov", type=float, default=110.0, help="horizontal field of view of the images, degrees")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--image-dir", default=None, help="folder of your own images (default: built-in COCO samples)")
    ap.add_argument("--gazes", default=None, help="JSON file of gaze points (see load_gazes); default: centre, corner, right")
    args = ap.parse_args()
    device = "cuda"

    model, processor, pipe = load_llava(args.load_4bit, device)

    # Foveated encoder: APT ViT carrying LLaVA's vision weights; layout from gaze (Steps 1-3).
    net = apt_from_hf_clip(pipe.vision, args.num_scales, [0.5] * (args.num_scales - 1))
    cfg = FoveaConfig.for_encoder(net, num_scales=args.num_scales, img_size=pipe.vision.config.image_size,
                                  fov_deg=args.fov, a=args.a)
    tok = FoveatedTokenizer(cfg, mean=CLIP_MEAN, std=CLIP_STD)
    print(f"Foveation: p={cfg.p}, patch sizes {cfg.sizes} px, R={cfg.R_deg:.0f} deg, a={cfg.a:g}, "
          f"rings at " + ", ".join(f"{r:.1f}" for r in cfg.rings_deg) + " deg")

    def foveated(gaze):
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
    cfg_off = FoveaConfig.for_encoder(net, num_scales=args.num_scales, img_size=cfg.img_size, fov_deg=args.fov, a=args.a)
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

    # Answers, per image at its own gaze points.
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n=== Answers (greedy, max {args.max_new_tokens} new tokens) ===")
    for n in names:
        gazes = gazes_for(gaze_spec, n)
        _, pv = inputs[n, "describe"]
        with torch.inference_mode():
            layouts = {g: layout_from_input_dict(tok(pv, [gaze]), cfg)[0] for g, gaze in gazes.items()}
        counts = {g: [sum(1 for *_, s in l if s == size) for size in cfg.sizes] for g, l in layouts.items()}
        print(f"\n### {n}")
        print("  image tokens: baseline 576; " + ", ".join(
            f"{g} at (u={u:.2f}, v={v:.2f}) -> {sum(counts[g])} [" + "/".join(map(str, counts[g])) + "]"
            for g, (u, v) in gazes.items()) + f"  (per size {cfg.sizes} px)")
        for p in PROMPTS:
            ids, pv = inputs[n, p]
            print(f"  [{p}]")
            rows = [("baseline (576)", pipe.encode_baseline)]
            rows += [(f"gaze {g} ({sum(counts[g])})", foveated(gaze)) for g, gaze in gazes.items()]
            width = max(len(label) for label, _ in rows) + 2
            for label, encode in rows:
                out = pipe.answer(encode, ids, pv, max_new_tokens=args.max_new_tokens)
                text = processor.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
                print(f"    {label:<{width}}{text}")

        # What the model was given at each gaze.
        _, pv = inputs[n, "describe"]
        shown = Image.fromarray((pv[0].float().cpu() * torch.tensor(CLIP_STD).view(3, 1, 1)
                                 + torch.tensor(CLIP_MEAN).view(3, 1, 1)).clamp(0, 1).mul(255).byte()
                                .permute(1, 2, 0).numpy())
        panels = [overlay(shown, cfg, gaze, layouts[g]) for g, gaze in gazes.items()]
        hstack(panels).save(os.path.join(OUT_DIR, f"{n}.png"))
        with open(os.path.join(OUT_DIR, f"{n}.gazes.txt"), "w") as f:
            f.write(", ".join(gazes))
    print(f"\nSaved overlays to {OUT_DIR}/ (one panel per gaze, in the order listed above)")


if __name__ == "__main__":
    main()
