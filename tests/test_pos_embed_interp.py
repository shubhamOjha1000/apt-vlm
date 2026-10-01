"""Paper Sec. 3.3: positional-encoding interpolation for different patch sizes.

Setup: a tiny APT ViT (336 px image, 14 px base patch -> 24x24 base grid) whose patch
embedding, CLS token and ZeroMLP are all zero. Every output token of the mixed patch
embedding is then *only* its position embedding, so we can check exactly which
position each token received. Which patches get merged is controlled directly through
hand-made importance maps (0 = merge, 1 = keep), not through image entropy.

Tests:
  1. Grid sizes  - big patches use a 12x12 (28 px) / 6x6 (56 px) grid; CLS position unchanged.
  2. Small patches - 14 px tokens keep exactly the normal ViT position embedding.
  3. Right slot  - each big patch gets its own cell of the coarse grid (no shift / transpose).
  4. Centre      - a big patch's position = the centre of the area it covers.
  5. Same size   - resizing the 24x24 grid to 24x24 returns it unchanged.
"""
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed

from src.models.vision_transformer import VisionTransformer
from src.models.patch_embed import TokenizedZeroConvPatchAttn
from src.models.patch_tokenizer import PatchTokenizer

IMG, P = 336, 14
GRID = IMG // P  # 24
DIM = 8


def patch_sizes(num_scales):
    return [P * 2 ** i for i in range(num_scales)]


def random_pos_embed(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1 + GRID * GRID, DIM, generator=g)


def make_apt(num_scales, pos_embed):
    """Tiny APT ViT whose mixed patch embedding outputs only position embeddings."""
    net = VisionTransformer(
        img_size=IMG, patch_size=P, embed_dim=DIM, depth=1, num_heads=2, num_classes=0,
        mixed_patch_embed=partial(TokenizedZeroConvPatchAttn, patch_size=P),
        num_scales=num_scales, thresholds=[0.5] * (num_scales - 1), weight_init="skip",
    )
    with torch.no_grad():
        net.patch_embed.proj.weight.zero_()
        if net.patch_embed.proj.bias is not None:
            net.patch_embed.proj.bias.zero_()
        net.cls_token.zero_()
        net.pos_embed.copy_(pos_embed)
    net.init_multiscale_patch_embed()
    assert torch.count_nonzero(net.mixed_patch.zero_conv.weight) == 0
    return net.eval()


@torch.no_grad()
def run(net, num_scales, merge):
    """merge: {patch_size: bool (B, h, w)}, True = use this big patch here.

    Returns (cls_vectors per image, {patch_size: {(b, row, col): token vector}}).
    """
    sizes = patch_sizes(num_scales)
    B = next(iter(merge.values())).shape[0]
    importance_maps = {P: torch.ones(B, GRID, GRID)}
    for s in sizes[1:]:
        importance_maps[s] = (~merge[s]).float()  # 0 < threshold 0.5 -> selected
    tokenizer = PatchTokenizer(
        num_scales=num_scales, base_patch_size=P, image_size=IMG,
        thresholds=[0.5] * (num_scales - 1), mean=[0.5] * 3, std=[0.5] * 3,
    )
    images = torch.zeros(B, 3, IMG, IMG)
    d = tokenizer(images, importance_maps=importance_maps)
    out, *_ = net.mixed_patch(images, net.pos_embed, d)

    seqs = torch.split(out[0], d["seqlens"])
    masks = torch.split(d["output_mask"], d["seqlens"])
    cls, tokens = [], {s: {} for s in sizes}
    for b, (seq, m) in enumerate(zip(seqs, masks)):
        assert (m == -1).sum() == 1
        cls.append(seq[m == -1][0])
        for i, s in enumerate(sizes):
            g = IMG // s
            flat_idx = d[f"pos_embed_mask_{s}"][b].nonzero().squeeze(1).tolist()
            vecs = seq[m == i + 1]
            assert len(flat_idx) == len(vecs)
            for f, v in zip(flat_idx, vecs):
                tokens[s][(b, f // g, f % g)] = v
    return cls, tokens


def no_merge(num_scales, B=1):
    return {s: torch.zeros(B, IMG // s, IMG // s, dtype=torch.bool) for s in patch_sizes(num_scales)[1:]}


def random_merge(num_scales, B=2, p=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {s: torch.rand(B, IMG // s, IMG // s, generator=g) < p for s in patch_sizes(num_scales)[1:]}


# 1. Grid sizes ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales,size", [(2, 28), (3, 28), (3, 56)])
def test_grid_sizes(num_scales, size):
    """Big patches need a coarser grid: 24x24 -> 12x12 (28 px) / 6x6 (56 px); CLS unchanged."""
    pos = random_pos_embed()
    net = make_apt(num_scales, pos)
    merge = no_merge(num_scales)
    merge[size][:] = True  # whole image in patches of this size
    cls, tokens = run(net, num_scales, merge)

    g = IMG // size
    assert g == {28: 12, 56: 6}[size]
    expected = resample_abs_pos_embed(pos, new_size=(g, g), old_size=(GRID, GRID), num_prefix_tokens=1)
    assert expected.shape == (1, 1 + g * g, DIM)

    # Exactly g*g big tokens and nothing else; together they use every cell of the g x g grid.
    assert len(tokens[size]) == g * g
    assert all(len(tokens[s]) == 0 for s in tokens if s != size)
    got = torch.stack([tokens[size][(0, r, c)] for r in range(g) for c in range(g)])
    torch.testing.assert_close(got, expected[0, 1:])

    # CLS position is not touched by the resampling, and the CLS token still gets it.
    torch.testing.assert_close(expected[0, 0], pos[0, 0])
    torch.testing.assert_close(cls[0], pos[0, 0])


# 2. Small patches ------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_small_patches_keep_original_pos(num_scales):
    """A 14 px token at grid cell (r, c) gets exactly pos_embed[1 + 24*r + c], as in a normal ViT."""
    pos = random_pos_embed(seed=1)
    net = make_apt(num_scales, pos)
    merge = random_merge(num_scales, B=2, seed=1)
    _, tokens = run(net, num_scales, merge)

    assert len(tokens[P]) > 0
    for (b, r, c), v in tokens[P].items():
        torch.testing.assert_close(v, pos[0, 1 + GRID * r + c], msg=f"image {b}, cell ({r},{c})")

    # With nothing merged, every one of the 576 base positions is used exactly once.
    _, tokens = run(net, num_scales, no_merge(num_scales))
    assert len(tokens[P]) == GRID * GRID
    got = torch.stack([tokens[P][(0, r, c)] for r in range(GRID) for c in range(GRID)])
    torch.testing.assert_close(got, pos[0, 1:])


# 3. Right slot ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_big_patches_get_their_own_cell(num_scales):
    """A big token at coarse cell (r, c) gets resampled[1 + g*r + c]: not a neighbour, not transposed."""
    pos = random_pos_embed(seed=2)
    net = make_apt(num_scales, pos)
    merge = random_merge(num_scales, B=2, seed=2)
    _, tokens = run(net, num_scales, merge)

    for s in patch_sizes(num_scales)[1:]:
        g = IMG // s
        resampled = resample_abs_pos_embed(pos, new_size=(g, g), old_size=(GRID, GRID), num_prefix_tokens=1)[0, 1:]
        for (b, r, c), v in tokens[s].items():
            torch.testing.assert_close(v, resampled[g * r + c], msg=f"{s}px, image {b}, cell ({r},{c})")
    # Make sure the test actually exercised big patches in more than one image.
    assert {b for (b, _, _) in tokens[28]} == {0, 1}


# 4. Centre check -------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales,size", [(2, 28), (3, 56)])
def test_big_patch_position_is_its_centre(num_scales, size):
    """With pos_embed = (row, col) of each 14 px cell, a big patch's position = centre of its area."""
    rows, cols = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
    pos = torch.zeros(1, 1 + GRID * GRID, DIM)
    pos[0, 1:, 0] = rows.flatten().float()
    pos[0, 1:, 1] = cols.flatten().float()

    net = make_apt(num_scales, pos)
    merge = no_merge(num_scales)
    merge[size][:] = True
    _, tokens = run(net, num_scales, merge)

    k = size // P  # 14 px cells per side covered by one big patch (2 or 4)
    g = IMG // size
    interior_err, edge_err = 0., 0.
    for (_, r, c), v in tokens[size].items():
        centre = torch.tensor([k * r + (k - 1) / 2, k * c + (k - 1) / 2])
        err = (v[:2] - centre).abs().max().item()
        # Antialiased bicubic needs neighbours on both sides; the outer 2 cells are near the border.
        if 2 <= r < g - 2 and 2 <= c < g - 2:
            interior_err = max(interior_err, err)
        else:
            edge_err = max(edge_err, err)
    print(f"{size}px: max centre error interior={interior_err:.2e}, edge={edge_err:.2f} (in 14px cells)")
    assert interior_err < 1e-4, "interior big patches should sit exactly at the centre of their area"
    assert edge_err < 0.5, "border big patches should stay within half a 14px cell of their centre"


# 5. Same size ----------------------------------------------------------------------------
def test_resize_to_same_size_is_identity():
    """Resizing the 24x24 grid to 24x24 must return exactly the same position embeddings."""
    pos = random_pos_embed(seed=3)
    same = resample_abs_pos_embed(pos, new_size=(GRID, GRID), old_size=(GRID, GRID), num_prefix_tokens=1)
    torch.testing.assert_close(same, pos)

    # Same check on the underlying interpolation (in case the timm helper short-circuits).
    grid = pos[:, 1:].reshape(1, GRID, GRID, DIM).permute(0, 3, 1, 2)
    resized = F.interpolate(grid, size=(GRID, GRID), mode="bicubic", antialias=True, align_corners=False)
    torch.testing.assert_close(resized, grid)
