"""Paper Sec. 3.1: deciding patch sizes.

Tests:
  9. Full coverage - the chosen patches cover the whole image exactly once: every 14 px cell
                     lies under exactly one chosen patch (no gaps, no overlaps), so their areas
                     add up to the image area (576 cells for a 336 px image).
"""
import pytest
import torch

from src.models.entropy_utils import compute_patch_entropy_batched, select_patches_by_threshold
from src.models.patch_tokenizer import PatchTokenizer

IMG, P = 336, 14
GRID = IMG // P  # 24

THRESHOLDS = {
    2: [[-1.0], [5.5], [float("inf")]],
    3: [[-1.0, -1.0], [5.75, 4.0], [5.5, 5.5], [float("inf"), float("inf")]],
}


def patch_sizes(num_scales):
    return [P * 2 ** i for i in range(num_scales)]


def make_images(B=4, seed=0):
    """Pixel values in [0, 255]: noise, flat colour, half flat / half noise, smooth gradient."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(3, IMG, IMG, generator=g) * 255
    flat = torch.full((3, IMG, IMG), 128.0)
    half = flat.clone()
    half[:, :, IMG // 2:] = noise[:, :, IMG // 2:]
    ramp = torch.linspace(0, 255, IMG).view(1, 1, IMG).expand(3, IMG, IMG)
    return torch.stack([noise, flat, half, ramp])[:B]


def random_entropy_maps(num_scales, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {s: torch.rand(B, IMG // s, IMG // s, generator=g) * 8 for s in patch_sizes(num_scales)}


def check_full_coverage(masks, num_scales):
    """Every 14 px cell is under exactly one chosen patch; areas sum to the image area."""
    B = masks[P].shape[0]
    coverage = torch.zeros(B, GRID, GRID)
    for s in patch_sizes(num_scales):
        k = s // P
        m = masks[s]
        assert set(m.unique().tolist()) <= {0.0, 1.0}, f"{s}px mask is not 0/1"
        coverage += m.repeat_interleave(k, dim=1).repeat_interleave(k, dim=2)
    gaps, overlaps = (coverage == 0).sum().item(), (coverage > 1).sum().item()
    assert gaps == 0 and overlaps == 0, f"{gaps} cells uncovered, {overlaps} cells covered more than once"

    counts = {s: masks[s].flatten(1).sum(1) for s in patch_sizes(num_scales)}
    area = sum(n * (s // P) ** 2 for s, n in counts.items())
    assert (area == GRID * GRID).all(), f"areas add up to {area.tolist()}, expected {GRID * GRID}"
    return counts


# 9. Full coverage ------------------------------------------------------------------------
def entropy_maps(images, num_scales):
    """Repo entropy, one image at a time (its one-hot histogram needs ~0.25 GB per image)."""
    per_image = [compute_patch_entropy_batched(im[None], patch_size=P, num_scales=num_scales) for im in images]
    return {s: torch.cat([m[s] for m in per_image]) for s in patch_sizes(num_scales)}


@pytest.mark.parametrize("num_scales,thresholds", [(n, t) for n, ts in THRESHOLDS.items() for t in ts])
def test_full_coverage_real_entropy(num_scales, thresholds):
    """Entropy of real pixel patterns + paper/edge-case thresholds -> whole image covered exactly once."""
    maps = entropy_maps(make_images(), num_scales)
    masks = select_patches_by_threshold(maps, thresholds=thresholds)
    counts = check_full_coverage(masks, num_scales)
    names = ["noise", "flat", "half", "ramp"]
    for b, name in enumerate(names):
        per_size = " + ".join(f"{int(counts[s][b])}x{(s // P) ** 2}" for s in patch_sizes(num_scales))
        print(f"t={thresholds} {name:>5}: {per_size} = {GRID * GRID} cells")


@pytest.mark.parametrize("num_scales", [2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_full_coverage_random_maps(num_scales, seed):
    """Random entropy maps and random thresholds -> still covered exactly once."""
    g = torch.Generator().manual_seed(100 + seed)
    thresholds = (torch.rand(num_scales - 1, generator=g) * 8).tolist()
    masks = select_patches_by_threshold(random_entropy_maps(num_scales, seed=seed), thresholds=thresholds)
    check_full_coverage(masks, num_scales)


@pytest.mark.parametrize("num_scales", [2, 3])
def test_full_coverage_matches_tokenizer(num_scales):
    """The tokenizer's sequence lengths agree: 1 CLS + one token per chosen patch, same full coverage."""
    maps = random_entropy_maps(num_scales, seed=7)
    thresholds = [4.0] * (num_scales - 1)
    tokenizer = PatchTokenizer(
        num_scales=num_scales, base_patch_size=P, image_size=IMG,
        thresholds=thresholds, mean=[0.5] * 3, std=[0.5] * 3,
    )
    B = maps[P].shape[0]
    d = tokenizer(torch.zeros(B, 3, IMG, IMG), importance_maps=maps)
    masks = {s: d[f"pos_embed_mask_{s}"].float().view(B, IMG // s, IMG // s) for s in patch_sizes(num_scales)}
    counts = check_full_coverage(masks, num_scales)
    n_tokens = sum(counts.values())
    assert d["seqlens"] == (1 + n_tokens).int().tolist()
