"""Gaze-foveated patching (Steps 1-3) on CLIP ViT-L/14-336 through APT's aggregation + PE interpolation.

For any image and a gaze point (u, v) in [0, 1]:
  Step 1  p from the encoder, Step 2  scale ladder, Step 3  ring radii in degrees (FOVI recipe),
  -> quadtree layout from eccentricity -> APT patch aggregation (Eq. 2) + positional interpolation
  -> ViT forward. No training.

Prints the config, a token-budget table over the fall-off parameter a, tokens per image and gaze,
and saves overlays (patch grid, gaze point, ring contours) to outputs/foveation/.

Usage (Colab):  python scripts/foveation_demo.py [--num-scales 3] [--a 40] [--fov 110]
"""
import argparse
import os
import sys

os.environ.setdefault("APT_ATTN_IMPL", "eager")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from scripts.forward_check import TIMM_NAME, IMG_SIZE, CLIP_MEAN, CLIP_STD, build_apt, load_images, preprocess
from src.models.foveation import (
    FoveaConfig, FoveatedTokenizer, eccentricity_deg, foveated_layout, layout_from_input_dict, ring_radii_px_at_centre,
)

GAZES = {"centre": (0.5, 0.5), "corner": (0.1, 0.1), "right": (0.75, 0.4)}
SIZE_COLOURS = {14: (255, 255, 255), 28: (255, 215, 0), 56: (0, 200, 255), 112: (255, 80, 200)}
OUT_DIR = "outputs/foveation"


def overlay(img: Image.Image, cfg: FoveaConfig, gaze_uv, layout):
    """Patch grid coloured by size, ring contours (iso-eccentricity), gaze point."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im, "RGBA")
    for x, y, s in layout:
        d.rectangle([x, y, x + s - 1, y + s - 1], outline=SIZE_COLOURS.get(s, (255, 0, 0)) + (220,), width=1)
    ys, xs = torch.meshgrid(torch.arange(cfg.img_size) + 0.5, torch.arange(cfg.img_size) + 0.5, indexing="ij")
    ecc = eccentricity_deg(xs, ys, torch.tensor([gaze_uv]) * cfg.img_size, cfg.img_size, cfg.fov_deg)[0]
    arr = np.array(im)
    for rho in cfg.rings_deg:
        inside = (ecc < rho).numpy()
        edge = np.zeros_like(inside)
        edge[1:] |= inside[1:] ^ inside[:-1]
        edge[:, 1:] |= inside[:, 1:] ^ inside[:, :-1]
        arr[edge] = (255, 40, 40)
    im = Image.fromarray(arr)
    gx, gy = gaze_uv[0] * cfg.img_size, gaze_uv[1] * cfg.img_size
    ImageDraw.Draw(im).ellipse([gx - 5, gy - 5, gx + 5, gy + 5], fill=(255, 0, 0), outline=(255, 255, 255))
    return im


def hstack(images, pad=6):
    w = sum(i.width for i in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (w, max(i.height for i in images)), "white")
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + pad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scales", type=int, default=3)
    ap.add_argument("--a", type=float, default=40.0)
    ap.add_argument("--fov", type=float, default=110.0, help="horizontal field of view of the images, degrees")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref = timm.create_model(TIMM_NAME, pretrained=True, num_classes=0).eval()
    net = build_apt(ref, args.num_scales, [0.5] * (args.num_scales - 1), device)
    uniform_net = build_apt(ref, 2, [-1.0], device)  # normal ViT path: every patch 14 px
    del ref

    # Steps 1-3.
    cfg = FoveaConfig.for_encoder(net, num_scales=args.num_scales, img_size=IMG_SIZE, fov_deg=args.fov, a=args.a)
    print(f"Step 1  base patch size p = {cfg.p}  (read from the encoder's patch embedding)")
    print(f"Step 2  scale ladder S = {cfg.num_scales}: {cfg.sizes} px, image {cfg.img_size} px "
          f"= {cfg.img_size // cfg.sizes[-1]}x{cfg.img_size // cfg.sizes[-1]} cells of {cfg.sizes[-1]} px")
    print(f"Step 3  R = {cfg.R_deg:.1f} deg (half of {cfg.fov_deg:.0f} deg FOV), a = {cfg.a:g}: ring radii = "
          + ", ".join(f"{r:.1f} deg" for r in cfg.rings_deg)
          + "  (= " + ", ".join(f"{r:.0f}" for r in ring_radii_px_at_centre(cfg)) + " px for a centred gaze)")
    for i, s in enumerate(cfg.sizes):
        lo = 0 if i == 0 else cfg.rings_deg[i - 1]
        hi = cfg.rings_deg[i] if i < len(cfg.rings_deg) else float("inf")
        print(f"          {s:>3} px patches where eccentricity in [{lo:.1f}, {hi:.1f}) deg")

    # Token budget vs a (like the spec's table).
    uniform = (cfg.img_size // cfg.p) ** 2
    print(f"\nToken budget vs fall-off a  (S = {cfg.num_scales}, {cfg.img_size} px, uniform p = {cfg.p}: {uniform} tokens)")
    print(f"{'a':>6}  {'ring radii (deg)':<20}{'centre gaze':>12}{'corner gaze':>12}")
    for a in (5, 10, 20, 40, 80, 160, 320):
        c = FoveaConfig(p=cfg.p, num_scales=cfg.num_scales, img_size=cfg.img_size, fov_deg=cfg.fov_deg, a=a)
        n_c, n_k = (len(foveated_layout(c, [g])[0]) for g in (GAZES["centre"], GAZES["corner"]))
        print(f"{a:>6}  {', '.join(f'{r:.1f}' for r in c.rings_deg):<20}{n_c:>12}{n_k:>12}")

    # Images x gazes through APT.
    pil = load_images()
    names = list(pil)
    images = preprocess([pil[n] for n in names]).to(device)
    tok = FoveatedTokenizer(cfg, mean=CLIP_MEAN, std=CLIP_STD)
    cfg_u = FoveaConfig(p=cfg.p, num_scales=2, img_size=cfg.img_size, fov_deg=cfg.fov_deg)
    cfg_u.rings_deg = [float("inf")]  # no cell is ever far enough -> all 14 px = normal ViT
    with torch.no_grad():
        d_u = FoveatedTokenizer(cfg_u, mean=CLIP_MEAN, std=CLIP_STD)(images, [GAZES["centre"]] * len(names))
        assert d_u["seqlens"] == [1 + uniform] * len(names)
        x, cu, ms, cls_u, _ = uniform_net.mixed_patch(images, uniform_net.pos_embed, d_u)
        cls_uniform = uniform_net.forward_features(x, cu_seqlens=cu, max_seqlen=ms)[0, cls_u]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nTokens through APT (CLS excluded); per-size counts for {cfg.sizes} px; CLS cosine vs the normal ViT")
    print(f"{'image':<20}" + "".join(f"{g:>34}" for g in GAZES))
    rows = {n: f"{n:<20}" for n in names}
    for gname, gaze in GAZES.items():
        gazes = [gaze] * len(names)
        with torch.no_grad():
            d = tok(images, gazes)
            x, cu, ms, cls_loc, _ = net.mixed_patch(images, net.pos_embed, d)
            feats = net.forward_features(x, cu_seqlens=cu, max_seqlen=ms)
        assert torch.isfinite(feats).all(), "NaN/inf in the foveated forward pass"
        layouts = layout_from_input_dict(d, cfg)
        assert [sorted(l) for l in layouts] == [sorted(l) for l in foveated_layout(cfg, gazes)]
        cos = F.cosine_similarity(feats[0, cls_loc], cls_uniform, dim=-1).tolist()
        for b, n in enumerate(names):
            counts = [sum(1 for *_, s in layouts[b] if s == size) for size in cfg.sizes]
            cell = f"{sum(counts)} [" + "/".join(map(str, counts)) + f"] cos={cos[b]:.3f}"
            rows[n] += f"{cell:>34}"
    for n in names:
        print(rows[n])

    # Overlays: one strip per image, one panel per gaze.
    shown_imgs = [Image.fromarray((images[b].cpu() * torch.tensor(CLIP_STD).view(3, 1, 1)
                                   + torch.tensor(CLIP_MEAN).view(3, 1, 1)).clamp(0, 1).mul(255).byte()
                                  .permute(1, 2, 0).numpy()) for b in range(len(names))]
    for b, n in enumerate(names):
        panels = [overlay(shown_imgs[b], cfg, g, foveated_layout(cfg, [g])[0]) for g in GAZES.values()]
        hstack(panels).save(os.path.join(OUT_DIR, f"{n}.png"))
    print(f"\nSaved overlays to {OUT_DIR}/ (white = {cfg.p} px, yellow = 28, blue = 56, pink = 112; "
          f"red lines = ring boundaries, red dot = gaze; panels: {', '.join(GAZES)})")
    print("All foveated forward passes ran without errors.")


if __name__ == "__main__":
    main()
