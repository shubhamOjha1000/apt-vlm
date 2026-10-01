"""Gaze-foveated patching (Foveated Patching Framework, Steps 1-3) and its hook into APT.

Steps:
  1. Base patch size  - read from the encoder's patch-embedding kernel.
  2. Scale ladder     - p * 2^i; image size must be a multiple of the largest patch.
  3. Ring radii (deg) - FOVI recipe, evenly spaced in log(r + a).
Layout:
  - matches a direct, recursive implementation of the spec's quadtree rule
    ("keep cell at size 2^i p iff eccentricity(centre) >= rho_i, else split into 4"),
  - the cell under the gaze is the smallest, the fovea follows the gaze,
  - every 14 px cell is covered exactly once, centred gaze gives a symmetric layout,
  - extreme rings give all-small / all-big layouts, batches with different gazes work.
APT hook:
  - the layout APT receives is the layout we built,
  - APT's patch aggregation (Eq. 2) and positional-encoding interpolation run on it unchanged.
"""
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed

from src.models.foveation import (
    FoveaConfig, FoveatedTokenizer, base_patch_size, eccentricity_deg, eccentricity_maps, focal_px,
    foveated_layout, foveated_masks, layout_from_input_dict, ring_radii_deg, scale_ladder,
)
from src.models.patch_embed import TokenizedZeroConvPatchAttn
from src.models.vision_transformer import VisionTransformer

IMG, P, DIM = 336, 14, 8
GRID = IMG // P  # 24
GAZES = [(0.5, 0.5), (0.1, 0.1), (0.7, 0.4), (0.93, 0.62), (0.25, 0.8)]


def make_net(p=P, num_scales=3, seed=0):
    torch.manual_seed(seed)
    net = VisionTransformer(
        img_size=IMG, patch_size=p, embed_dim=DIM, depth=1, num_heads=2, num_classes=0,
        mixed_patch_embed=partial(TokenizedZeroConvPatchAttn, patch_size=p),
        num_scales=num_scales, thresholds=[0.5] * (num_scales - 1), weight_init="skip",
    )
    with torch.no_grad():
        net.pos_embed.normal_()
        net.cls_token.normal_()
    net.init_multiscale_patch_embed()
    return net.eval()


def ecc_at(cfg, gaze_uv, x, y):
    gaze = torch.tensor([gaze_uv], dtype=torch.float32) * cfg.img_size
    return eccentricity_deg(torch.tensor(float(x)), torch.tensor(float(y)), gaze, cfg.img_size, cfg.fov_deg)[0].item()


def reference_layout(cfg, gaze_uv):
    """The spec's rule written out directly: recurse from the coarsest cells down."""
    out = []

    def visit(x, y, i):
        s = cfg.sizes[i]
        if i == 0 or ecc_at(cfg, gaze_uv, x + s / 2, y + s / 2) >= cfg.rings_deg[i - 1]:
            out.append((x, y, s))
            return
        h = s // 2
        for dy in (0, h):
            for dx in (0, h):
                visit(x + dx, y + dy, i - 1)

    top = cfg.sizes[-1]
    for y in range(0, cfg.img_size, top):
        for x in range(0, cfg.img_size, top):
            visit(x, y, cfg.num_scales - 1)
    return sorted(out)


# Step 1 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [14, 16])
def test_base_patch_size_comes_from_encoder(p):
    net = VisionTransformer(img_size=224, patch_size=p, embed_dim=DIM, depth=1, num_heads=2, num_classes=0,
                            weight_init="skip")  # the repo's default init path is broken (named_apply not imported)
    assert base_patch_size(net) == p
    assert FoveaConfig.for_encoder(make_net(), img_size=IMG).p == P


# Step 2 --------------------------------------------------------------------------------------
def test_scale_ladder():
    assert scale_ladder(14, 3, 336) == [14, 28, 56]
    assert scale_ladder(14, 4, 448) == [14, 28, 56, 112]
    assert scale_ladder(14, 4, 336) == [14, 28, 56, 112]  # 336 = 3 x 112
    assert scale_ladder(16, 1, 224) == [16]
    with pytest.raises(ValueError, match="multiple of the largest patch 56"):
        scale_ladder(14, 3, 300)
    with pytest.raises(ValueError):
        FoveaConfig(img_size=300, num_scales=3)  # 300 is not a multiple of 56


# Step 3 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("a", [5, 40, 320])
@pytest.mark.parametrize("S", [2, 3, 4])
def test_ring_radii_fovi_recipe(a, S):
    R = 55.0
    rings = ring_radii_deg(R, a, S)
    assert len(rings) == S - 1
    assert all(0 < r < R for r in rings) and rings == sorted(rings)
    # Evenly spaced in u = log(r + a), with the end points log(a) and log(R + a).
    u = torch.log(torch.tensor([0.0] + rings + [R], dtype=torch.float64) + a)
    steps = u[1:] - u[:-1]
    torch.testing.assert_close(steps, torch.full_like(steps, steps.mean().item()))


def test_ring_radii_fall_off_parameter():
    """Small a -> small fovea; large a -> rings approach even spacing R*k/S."""
    assert ring_radii_deg(55, 5, 3)[0] < ring_radii_deg(55, 40, 3)[0] < ring_radii_deg(55, 320, 3)[0]
    big_a = ring_radii_deg(55, 1e6, 3)
    torch.testing.assert_close(torch.tensor(big_a), torch.tensor([55 / 3, 110 / 3]), atol=1e-2, rtol=0)


def test_default_config():
    cfg = FoveaConfig()
    assert cfg.p == 14 and cfg.sizes == [14, 28, 56] and cfg.R_deg == 55.0
    assert cfg.rings_deg == ring_radii_deg(55.0, 40.0, 3)


# Degrees <-> pixels ---------------------------------------------------------------------------
def test_eccentricity_pinhole():
    cfg = FoveaConfig()
    c = IMG / 2
    assert ecc_at(cfg, (0.5, 0.5), c, c) == pytest.approx(0, abs=1e-6)
    # Image edge, centre gaze: half the field of view, in every direction.
    for x, y in [(IMG, c), (0, c), (c, 0), (c, IMG)]:
        assert ecc_at(cfg, (0.5, 0.5), x, y) == pytest.approx(cfg.fov_deg / 2, abs=1e-3)
    # Pinhole: r = f tan(theta).
    f = focal_px(IMG, cfg.fov_deg)
    for deg in (5, 20, 40):
        x = c + f * torch.tan(torch.deg2rad(torch.tensor(float(deg)))).item()
        assert ecc_at(cfg, (0.5, 0.5), x, c) == pytest.approx(deg, abs=1e-3)
    # Zero at the gaze point wherever it is; symmetric in the two points.
    for g in GAZES:
        gx, gy = g[0] * IMG, g[1] * IMG
        assert ecc_at(cfg, g, gx, gy) == pytest.approx(0, abs=1e-4)  # float32 gaze pixel rounding
        assert ecc_at(cfg, g, 30, 200) == pytest.approx(ecc_at(cfg, (30 / IMG, 200 / IMG), gx, gy), abs=1e-4)


# Layout ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("gaze", GAZES)
@pytest.mark.parametrize("S,a", [(2, 40), (3, 5), (3, 40), (3, 320), (4, 40)])
def test_layout_matches_quadtree_rule(gaze, S, a):
    cfg = FoveaConfig(num_scales=S, a=a)
    assert sorted(foveated_layout(cfg, [gaze])[0]) == reference_layout(cfg, gaze)


@pytest.mark.parametrize("gaze", GAZES)
@pytest.mark.parametrize("S", [2, 3, 4])
def test_full_coverage(gaze, S):
    cfg = FoveaConfig(num_scales=S)
    cover = torch.zeros(GRID, GRID, dtype=torch.int)
    for x, y, s in foveated_layout(cfg, [gaze])[0]:
        cover[y // P:(y + s) // P, x // P:(x + s) // P] += 1
    assert (cover == 1).all(), f"{(cover == 0).sum()} cells uncovered, {(cover > 1).sum()} covered twice"


@pytest.mark.parametrize("gaze", GAZES)
def test_gaze_cell_is_smallest_and_fovea_follows_gaze(gaze):
    cfg = FoveaConfig()
    layout = foveated_layout(cfg, [gaze])[0]
    gx, gy = gaze[0] * IMG, gaze[1] * IMG
    under_gaze = [s for x, y, s in layout if x <= gx < x + s and y <= gy < y + s]
    assert under_gaze == [P]
    # The 14 px patches are gathered around the gaze: their mean is nearer this gaze than any other.
    small = torch.tensor([(x + s / 2, y + s / 2) for x, y, s in layout if s == P])
    dists = [((small.mean(0) - torch.tensor([g[0] * IMG, g[1] * IMG])) ** 2).sum().item() for g in GAZES]
    assert min(range(len(GAZES)), key=dists.__getitem__) == GAZES.index(gaze)


@pytest.mark.parametrize("S", [2, 3, 4])
def test_centre_gaze_is_symmetric(S):
    masks = foveated_masks(FoveaConfig(num_scales=S), [(0.5, 0.5)])
    for s, m in masks.items():
        m = m[0]
        assert torch.equal(m, m.flip(0)) and torch.equal(m, m.flip(1)) and torch.equal(m, m.T), f"{s}px mask"


def test_extreme_rings():
    cfg = FoveaConfig()
    cfg.rings_deg = [float("inf")] * 2   # nothing is far enough -> all 14 px, like a normal ViT
    assert len(foveated_layout(cfg, [(0.3, 0.6)])[0]) == GRID * GRID
    cfg.rings_deg = [0.0, 0.0]           # everything is far enough -> all 56 px
    assert len(foveated_layout(cfg, [(0.3, 0.6)])[0]) == (IMG // 56) ** 2


def test_more_foveation_fewer_tokens():
    """Larger a pushes the rings out -> more small patches -> more tokens; uniform is the upper bound."""
    counts = [len(foveated_layout(FoveaConfig(a=a), [(0.5, 0.5)])[0]) for a in (5, 40, 320)]
    assert counts == sorted(counts) and counts[-1] < GRID * GRID
    print(f"tokens at centre gaze for a = 5 / 40 / 320: {counts} (uniform: {GRID * GRID})")


def test_batch_with_different_gazes():
    cfg = FoveaConfig()
    batch = foveated_layout(cfg, GAZES)
    assert [sorted(l) for l in batch] == [sorted(foveated_layout(cfg, [g])[0]) for g in GAZES]
    ecc = eccentricity_maps(cfg, GAZES)
    assert all(m.shape == (len(GAZES), IMG // s, IMG // s) for s, m in ecc.items())


# Hook into APT ---------------------------------------------------------------------------------
@pytest.mark.parametrize("S", [2, 3])
def test_apt_receives_the_foveated_layout(S):
    cfg = FoveaConfig(num_scales=S)
    tok = FoveatedTokenizer(cfg, mean=[0.5] * 3, std=[0.5] * 3)
    images = torch.randn(len(GAZES), 3, IMG, IMG)
    d = tok(images, GAZES)
    expected = foveated_layout(cfg, GAZES)
    assert [sorted(l) for l in layout_from_input_dict(d, cfg)] == [sorted(l) for l in expected]
    assert d["seqlens"] == [1 + len(l) for l in expected]


@pytest.mark.parametrize("S", [2, 3])
def test_apt_aggregation_and_pos_interp_on_foveated_layout(S):
    """Through APT's mixed patch embedding: every token is Eq. 2 at its layout cell with the
    interpolated position (ZeroMLP = 0 at init -> patch_embed(resized patch) + resampled pos)."""
    cfg = FoveaConfig(num_scales=S)
    net = make_net(num_scales=S)
    tok = FoveatedTokenizer(cfg, mean=[0.5] * 3, std=[0.5] * 3)
    gazes = GAZES[:3]
    images = torch.randn(len(gazes), 3, IMG, IMG, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        d = tok(images, gazes)
        out, cu_seqlens, max_seqlen, _, _ = net.mixed_patch(images, net.pos_embed, d)
        feats = net.forward_features(out, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    assert torch.isfinite(feats).all()

    proj = net.mixed_patch.patch_embed.proj
    seqs = torch.split(out[0], d["seqlens"])
    masks = torch.split(d["output_mask"], d["seqlens"])
    for b, (seq, m) in enumerate(zip(seqs, masks)):
        torch.testing.assert_close(seq[m == -1][0], net.mixed_patch.cls_token[0, 0] + net.pos_embed[0, 0])
        for i, s in enumerate(cfg.sizes):
            g = IMG // s
            pos = net.pos_embed[0, 1:] if s == P else resample_abs_pos_embed(
                net.pos_embed, new_size=(g, g), old_size=(GRID, GRID), num_prefix_tokens=1)[0, 1:]
            cells = d[f"pos_embed_mask_{s}"][b].nonzero().squeeze(1).tolist()
            vecs = seq[m == i + 1]
            assert len(cells) == len(vecs)
            for f, v in zip(cells, vecs):
                r, c = divmod(f, g)
                patch = images[b:b + 1, :, r * s:(r + 1) * s, c * s:(c + 1) * s]
                if s != P:
                    patch = F.interpolate(patch, size=(P, P), mode="bilinear")
                expected = F.conv2d(patch, proj.weight, proj.bias, stride=P).flatten() + pos[f]
                torch.testing.assert_close(v, expected, atol=1e-5, rtol=1e-4, msg=f"image {b}, {s}px cell ({r},{c})")
