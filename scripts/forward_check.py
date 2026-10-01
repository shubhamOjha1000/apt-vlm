"""Forward-pass sanity check for APT on CLIP ViT-L/14-336 (OpenAI weights), eager attention.

No training, no ImageNet: loads pretrained weights, runs a few public images through
  (a) the unmodified timm ViT (baseline) and
  (b) the APT ViT from this repo,
checks the APT forward pass is correct, and reports the number of image tokens
(CLS excluded) each model produces.

Checks:
  1. Weights load: only the new APT modules (mixed_patch.*) are missing.
  2. Parity: with merging disabled, APT gives 576 tokens and matches timm's features.
  3. Packing: a packed batch gives the same features as running each image alone
     (i.e. the block-diagonal attention mask keeps images separate).
  4. APT settings run without NaN/inf; token counts reported per image.

Usage (Colab):  python scripts/forward_check.py
"""
import os
import sys
import io
import urllib.request
from functools import partial

os.environ.setdefault("APT_ATTN_IMPL", "eager")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import vit_components
from src.models.vision_transformer import VisionTransformer
from src.models.patch_embed import TokenizedZeroConvPatchAttn
from src.models.patch_tokenizer import PatchTokenizer

TIMM_NAME = "vit_large_patch14_clip_quickgelu_336.openai"  # OpenAI CLIP ViT-L/14-336, as in LLaVA-1.5
IMG_SIZE, PATCH = 336, 14
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# APT settings to test: name -> (num_scales, thresholds). Thresholds[i] applies to patch size PATCH * 2**(i+1).
APT_SETTINGS = {
    "APT 2-scale (14/28, t=5.5)": (2, [5.5]),
    "APT 3-scale (14/28/56, t=5.75,4.0)": (3, [5.75, 4.0]),
}

IMAGE_URLS = {
    "coco_cats": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "coco_living_room": "http://images.cocodataset.org/val2017/000000000139.jpg",
    "coco_bedroom": "http://images.cocodataset.org/val2017/000000000632.jpg",
    "coco_skier": "http://images.cocodataset.org/val2017/000000000785.jpg",
}

PARITY_TOL = 1e-2  # eager vs fused attention over 24 layers; real bugs give O(1) diffs


def load_images():
    images = {}
    for name, url in IMAGE_URLS.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                images[name] = Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:
            print(f"[warn] could not download {name}: {e}")
    # Synthetic smooth gradient (sky-like): should be merged heavily by APT.
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE] / IMG_SIZE
    grad = np.stack([90 + 60 * y, 150 + 50 * y, 230 - 20 * x], axis=-1).astype(np.uint8)
    images["synthetic_gradient"] = Image.fromarray(grad)
    return images


def preprocess(pil_images):
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return torch.stack([tf(im) for im in pil_images])


def build_apt(ref, num_scales, thresholds, device):
    """APT ViT with the same architecture as `ref`, loaded with ref's weights."""
    net = VisionTransformer(
        img_size=IMG_SIZE, patch_size=PATCH, embed_dim=1024, depth=24, num_heads=16,
        num_classes=0, pre_norm=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=ref.norm.eps),
        act_layer=type(ref.blocks[0].mlp.act),
        mixed_patch_embed=partial(TokenizedZeroConvPatchAttn, patch_size=PATCH),
        num_scales=num_scales, thresholds=thresholds, weight_init="skip",
    )
    missing, unexpected = net.load_state_dict(ref.state_dict(), strict=False)
    bad_missing = [k for k in missing if not k.startswith("mixed_patch.")]
    assert not bad_missing and not unexpected, f"weight mismatch: missing={bad_missing} unexpected={unexpected}"
    # Same order as ViTLitModule: load weights first, then hand patch_embed/cls/pos to mixed_patch.
    net.init_multiscale_patch_embed()
    assert torch.count_nonzero(net.mixed_patch.zero_conv.weight) == 0, "ZeroMLP should start at zero"
    return net.to(device).eval()


def build_tokenizer(num_scales, thresholds):
    return PatchTokenizer(
        num_scales=num_scales, base_patch_size=PATCH, image_size=IMG_SIZE,
        thresholds=thresholds, mean=CLIP_MEAN, std=CLIP_STD, method="entropy",
    )


@torch.no_grad()
def apt_forward(net, tokenizer, images):
    """Returns per-image token features (CLS first) and per-image counts per patch scale."""
    input_dict = tokenizer(images, importance_maps=tokenizer.compute_importance_maps(images))
    x, cu_seqlens, max_seqlen, cls_loc, _ = net.mixed_patch(images, net.pos_embed, input_dict)
    feats = net.forward_features(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)[0]
    seqlens = input_dict["seqlens"]
    assert feats.shape[0] == sum(seqlens)
    assert torch.equal(cls_loc, cu_seqlens[:-1].long()), "CLS token is not first in every sequence"
    per_image = list(torch.split(feats, seqlens))
    masks = torch.split(input_dict["output_mask"], seqlens)
    scale_counts = [[int((m == s + 1).sum()) for s in range(tokenizer.num_scales)] for m in masks]
    return per_image, scale_counts


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  attention={vit_components.ATTN_IMPL}  timm={timm.__version__}  torch={torch.__version__}")

    pil = load_images()
    names = list(pil)
    images = preprocess([pil[n] for n in names]).to(device)
    print(f"images: {names}")

    ref = timm.create_model(TIMM_NAME, pretrained=True, num_classes=0).to(device).eval()
    with torch.no_grad():
        ref_feats = ref.forward_features(images)  # (B, 1 + 576, 1024)
    n_base = ref_feats.shape[1] - ref.num_prefix_tokens
    print(f"\nBaseline ViT ({TIMM_NAME}): {n_base} image tokens per image")

    results = {}
    for setting, (num_scales, thresholds) in APT_SETTINGS.items():
        print(f"\n=== {setting} ===")
        net = build_apt(ref, num_scales, thresholds, device)
        print("[ok] weights loaded (only mixed_patch.* new), ZeroMLP is zero")

        # Check 2: merging disabled (entropy is never < -1) -> must reproduce the plain ViT.
        no_merge = build_tokenizer(num_scales, [-1.0] * (num_scales - 1))
        feats, _ = apt_forward(net, no_merge, images)
        assert all(f.shape[0] - 1 == n_base for f in feats), "no-merge APT should keep every base patch"
        diff = max((f - r).abs().max().item() for f, r in zip(feats, ref_feats))
        status = "ok" if diff < PARITY_TOL else "FAIL"
        print(f"[{status}] parity with merging off: {n_base} tokens, max |APT - timm| = {diff:.2e} (tol {PARITY_TOL})")
        assert diff < PARITY_TOL

        # Check 3/4: real thresholds, packed batch vs one image at a time.
        tok = build_tokenizer(num_scales, thresholds)
        feats, scale_counts = apt_forward(net, tok, images)
        assert all(torch.isfinite(f).all() for f in feats), "NaN/inf in APT output"
        pack_diff = 0.
        for i in range(len(names)):
            single, _ = apt_forward(net, tok, images[i:i + 1])
            pack_diff = max(pack_diff, (single[0] - feats[i]).abs().max().item())
        status = "ok" if pack_diff < PARITY_TOL else "FAIL"
        print(f"[{status}] packed batch == per-image: max diff = {pack_diff:.2e}")
        assert pack_diff < PARITY_TOL

        cls_cos = [F.cosine_similarity(f[0], r[0], dim=0).item() for f, r in zip(feats, ref_feats)]
        results[setting] = (scale_counts, cls_cos)
        del net
        if device == "cuda":
            torch.cuda.empty_cache()

    # Report: image tokens (CLS excluded).
    sizes = lambda n: "/".join(str(PATCH * 2 ** s) for s in range(n))
    print("\n" + "=" * 100)
    print("IMAGE TOKENS (CLS excluded).  APT columns: total  [count per patch size]  CLS cosine vs baseline")
    print("=" * 100)
    header = f"{'image':<20}{'baseline':>10}" + "".join(f"   {s:<36}" for s in APT_SETTINGS)
    print(header)
    for i, name in enumerate(names):
        row = f"{name:<20}{n_base:>10}"
        for setting, (num_scales, _) in APT_SETTINGS.items():
            counts, cos = results[setting][0][i], results[setting][1][i]
            cell = f"{sum(counts)} [{'/'.join(map(str, counts))}] cos={cos:.3f}"
            row += f"   {cell:<36}"
        print(row)
    row = f"{'MEAN':<20}{n_base:>10}"
    for setting in APT_SETTINGS:
        totals = [sum(c) for c in results[setting][0]]
        mean = np.mean(totals)
        row += f"   {f'{mean:.1f} ({100 * (1 - mean / n_base):.1f}% fewer)':<36}"
    print(row)
    for setting, (num_scales, _) in APT_SETTINGS.items():
        print(f"  {setting}: per-size counts are for patch sizes {sizes(num_scales)} px")
    print("\nAll forward-pass checks passed.")


if __name__ == "__main__":
    main()
