"""LLaVA-1.5-7B answers with gaze-foveated patching (Steps 1-3) vs normal LLaVA. No training.

For each image:
  * Baseline : LLaVA's own CLIP ViT-L/14-336, 576 image tokens.
  * Foveated : same CLIP weights through APT (patch aggregation Eq. 2 + positional interpolation),
               patch sizes from gaze (14 px fovea, 28 px middle ring, 56 px periphery), for a few gaze points.
Everything after the vision encoder (projector, LLM) is LLaVA's own and shared by both.

Before that, a check: with the rings pushed to infinity (every patch 14 px) the foveated path must give
LLaVA's own image features.

Usage (Colab):  python scripts/llava_foveated.py [--a 40] [--fov 110] [--num-scales 3] [--max-new-tokens 60]
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--num-scales", type=int, default=3)
    ap.add_argument("--a", type=float, default=40.0)
    ap.add_argument("--fov", type=float, default=110.0, help="horizontal field of view of the images, degrees")
    ap.add_argument("--max-new-tokens", type=int, default=60)
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

    images = load_images()
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
            a = pipe.encode_apt(net, tok_off, pv, input_dict=tok_off(pv, [GAZES["centre"]])).float()
            b = pipe.encode_baseline(pv).float()
        cos = F.cosine_similarity(a, b, dim=-1)
        print(f"  {n:<20} tokens {a.shape[0]} vs {b.shape[0]}   token cosine mean {cos.mean():.4f}, min {cos.min():.4f}")
        assert a.shape == b.shape and cos.mean() > 0.95, "foveated path with foveation off does not match LLaVA"

    # Token counts per gaze (geometry only: identical for every image).
    counts = {}
    with torch.inference_mode():
        _, pv = inputs[names[0], "describe"]
        for g, gaze in GAZES.items():
            layout = layout_from_input_dict(tok(pv, [gaze]), cfg)[0]
            counts[g] = [sum(1 for *_, s in layout if s == size) for size in cfg.sizes]
    print("\nImage tokens: baseline 576; foveated " + ", ".join(
        f"{g} {sum(c)} [{'/'.join(map(str, c))}]" for g, c in counts.items()) + f"  (per size {cfg.sizes} px)")

    # Answers.
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n=== Answers (greedy, max {args.max_new_tokens} new tokens) ===")
    for n in names:
        print(f"\n### {n}")
        for p in PROMPTS:
            ids, pv = inputs[n, p]
            print(f"  [{p}]")
            rows = [("baseline (576)", pipe.encode_baseline)]
            rows += [(f"gaze {g} ({sum(counts[g])})", foveated(gaze)) for g, gaze in GAZES.items()]
            for label, encode in rows:
                out = pipe.answer(encode, ids, pv, max_new_tokens=args.max_new_tokens)
                text = processor.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
                print(f"    {label:<20} {text}")

        # What the model was given at each gaze.
        _, pv = inputs[n, "describe"]
        shown = Image.fromarray((pv[0].float().cpu() * torch.tensor(CLIP_STD).view(3, 1, 1)
                                 + torch.tensor(CLIP_MEAN).view(3, 1, 1)).clamp(0, 1).mul(255).byte()
                                .permute(1, 2, 0).numpy())
        with torch.inference_mode():
            panels = [overlay(shown, cfg, gaze, layout_from_input_dict(tok(pv, [gaze]), cfg)[0]) for gaze in GAZES.values()]
        hstack(panels).save(os.path.join(OUT_DIR, f"{n}.png"))
    print(f"\nSaved overlays to {OUT_DIR}/ (panels: {', '.join(GAZES)})")


if __name__ == "__main__":
    main()
