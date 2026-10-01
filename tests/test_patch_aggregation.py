"""Paper Sec. 3.2: patch aggregation (Eq. 2).

    E(P) = ZeroMLP( Conv2d^(i)( {E(P_j) | P_j in P} ) ) + E( Resize_p(P) )   (+ position embedding)

Setup: a tiny APT ViT (112 px image, 14 px base patch -> 8x8 base grid, 4x4 grid of 28 px
patches, 2x2 grid of 56 px patches) with random weights and random images. Which patches get
merged is set by hand through importance maps (0 = merge, 1 = keep), not through entropy.
Expected values are recomputed independently from the raw image crops with F.conv2d.

Tests (numbers follow the plan):
  13. Resize      - a big patch shrunk to 14x14 uses only its own pixels.
  14. Split       - the small pieces of a big patch put back together give the original patch.
  15. Conv count  - 28 px goes through the combining conv once, 56 px twice, ending at 1 token.
  16. Start       - with ZeroMLP = 0, big token = patch_embed(shrunken patch) + its position.
  17. Small       - 14 px tokens equal a standard ViT patch embedding + position.
  18. ZeroMLP on  - a non-zero ZeroMLP changes only big tokens, exactly as in Eq. 2.
  19. Learning    - ZeroMLP gets a gradient at the start; after one step the conv does too.
  20. CLS         - CLS = cls_token + pos_embed[0], first in every image's sequence.
  21. No big ones - a patch size with nothing selected does not crash; counts stay right.
"""
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed

from src.models.vision_transformer import VisionTransformer
from src.models.patch_embed import TokenizedZeroConvPatchAttn
from src.models.patch_tokenizer import PatchTokenizer

IMG, P = 112, 14
GRID = IMG // P  # 8
DIM = 16
TOL = dict(atol=1e-5, rtol=1e-4)


def patch_sizes(num_scales):
    return [P * 2 ** i for i in range(num_scales)]


def make_apt(num_scales, seed=0):
    """Tiny APT ViT with random patch embedding, position embedding, CLS token and conv."""
    torch.manual_seed(seed)
    net = VisionTransformer(
        img_size=IMG, patch_size=P, embed_dim=DIM, depth=1, num_heads=2, num_classes=0,
        mixed_patch_embed=partial(TokenizedZeroConvPatchAttn, patch_size=P),
        num_scales=num_scales, thresholds=[0.5] * (num_scales - 1), weight_init="skip",
    )
    with torch.no_grad():
        net.pos_embed.normal_()
        net.cls_token.normal_()
    net.init_multiscale_patch_embed()
    assert torch.count_nonzero(net.mixed_patch.zero_conv.weight) == 0
    assert torch.count_nonzero(net.mixed_patch.zero_conv.bias) == 0
    return net.eval()


def make_images(B, seed=0):
    return torch.randn(B, 3, IMG, IMG, generator=torch.Generator().manual_seed(seed))


def hand_merge(num_scales, B=2):
    """Fixed merge pattern with big patches of every size in image 0 and different ones in image 1."""
    merge = {s: torch.zeros(B, IMG // s, IMG // s, dtype=torch.bool) for s in patch_sizes(num_scales)[1:]}
    merge[28][0, 0, 2] = merge[28][0, 3, 3] = merge[28][0, 2, 0] = True
    if B > 1:
        merge[28][1, 1, 1] = merge[28][1, 3, 0] = True
    if num_scales > 2:
        merge[56][0, 0, 0] = True
        if B > 1:
            merge[56][1, 1, 1] = True
    return merge


def tokenize(num_scales, images, merge):
    importance_maps = {P: torch.ones(images.shape[0], GRID, GRID)}
    for s in patch_sizes(num_scales)[1:]:
        importance_maps[s] = (~merge[s]).float()  # 0 < threshold 0.5 -> selected
    tokenizer = PatchTokenizer(
        num_scales=num_scales, base_patch_size=P, image_size=IMG,
        thresholds=[0.5] * (num_scales - 1), mean=[0.5] * 3, std=[0.5] * 3,
    )
    return tokenizer(images, importance_maps=importance_maps)


def cells(d, s):
    """(b, row, col) of every selected patch of size s, in the order the tokenizer stores them."""
    g = IMG // s
    mask = d[f"pos_embed_mask_{s}"]
    return [(b, f // g, f % g) for b in range(mask.shape[0]) for f in mask[b].nonzero().squeeze(1).tolist()]


def run(net, num_scales, images, merge):
    """Returns (tokenizer dict, mixed-patch output (N, D), {size: {(b, r, c): token}}, [cls per image])."""
    d = tokenize(num_scales, images, merge)
    out, cu_seqlens, _, _, _ = net.mixed_patch(images, net.pos_embed, d)
    out = out[0]
    seqs = torch.split(out, d["seqlens"])
    masks = torch.split(d["output_mask"], d["seqlens"])
    tokens, cls = {s: {} for s in patch_sizes(num_scales)}, []
    for b, (seq, m) in enumerate(zip(seqs, masks)):
        cls.append(seq[m == -1])
        for i, s in enumerate(patch_sizes(num_scales)):
            mine = [c for c in cells(d, s) if c[0] == b]
            vecs = seq[m == i + 1]
            assert len(mine) == len(vecs)
            tokens[s].update(zip(mine, vecs))
    return d, out, tokens, cls


def crop(images, b, r, c, s):
    return images[b:b + 1, :, r * s:(r + 1) * s, c * s:(c + 1) * s]


def embed(net, x):
    """Reference patch embedding: plain strided conv over an image/crop -> (D, H/14, W/14)."""
    proj = net.mixed_patch.patch_embed.proj
    return F.conv2d(x, proj.weight, proj.bias, stride=P)[0]


def resampled_pos(net, s):
    g = IMG // s
    return resample_abs_pos_embed(net.pos_embed, new_size=(g, g), old_size=(GRID, GRID), num_prefix_tokens=1)[0, 1:]


def eq2(net, images, b, r, c, s):
    """Big-patch token computed from scratch, straight from paper Eq. 2."""
    sub = embed(net, crop(images, b, r, c, s)).unsqueeze(0)  # (1, D, k, k) sub-patch embeddings, spatial layout kept
    conv = net.mixed_patch.patch_attn
    for _ in range((s // P).bit_length() - 1):  # i = log2(s / 14) times
        sub = F.conv2d(sub, conv.weight, conv.bias, stride=2)
    agg = sub.flatten()
    zero_mlp = net.mixed_patch.zero_conv
    resized = F.interpolate(crop(images, b, r, c, s), size=(P, P), mode="bilinear")
    return F.linear(agg, zero_mlp.weight, zero_mlp.bias) + embed(net, resized).flatten() + resampled_pos(net, s)[r * (IMG // s) + c]


# 13. Resize ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_resize_uses_only_its_own_pixels(num_scales):
    """Each shrunken big patch equals shrinking that patch alone (no pixels from neighbours)."""
    images = make_images(2)
    d = tokenize(num_scales, images, hand_merge(num_scales))
    for s in patch_sizes(num_scales)[1:]:
        resized = d[f"resized_patches_{s}"]
        assert resized.shape[1:] == (3, P, P)
        for (b, r, c), got in zip(cells(d, s), resized):
            alone = F.interpolate(crop(images, b, r, c, s), size=(P, P), mode="bilinear")[0]
            torch.testing.assert_close(got, alone, **TOL, msg=f"{s}px patch at image {b}, cell ({r},{c})")

            # Changing every pixel outside this patch must not change its shrunken version.
            noisy = images + 100.0
            noisy[b, :, r * s:(r + 1) * s, c * s:(c + 1) * s] = images[b, :, r * s:(r + 1) * s, c * s:(c + 1) * s]
            d2 = tokenize(num_scales, noisy, hand_merge(num_scales))
            idx = cells(d2, s).index((b, r, c))
            torch.testing.assert_close(d2[f"resized_patches_{s}"][idx], got, **TOL)

        # Info: is the shrink a true average of all pixels (2x2 / 4x4 average pooling)?
        is_avg = all(
            torch.allclose(got, F.avg_pool2d(crop(images, b, r, c, s), s // P)[0], atol=1e-5)
            for (b, r, c), got in zip(cells(d, s), resized)
        )
        print(f"{s}px resize == average of all {(s // P) ** 2} pixels per output pixel: {is_avg}")


# 14. Split -------------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_split_reassembles_patch(num_scales):
    """The 14 px pieces of a big patch, put back in row-major order, rebuild it exactly."""
    images = make_images(2, seed=1)
    d = tokenize(num_scales, images, hand_merge(num_scales))
    for s in patch_sizes(num_scales)[1:]:
        k = s // P
        pieces = d[f"full_patches_{s}"]
        assert pieces.shape[1:] == (k * k, 3, P, P)
        for (b, r, c), p in zip(cells(d, s), pieces):
            rebuilt = p.view(k, k, 3, P, P).permute(2, 0, 3, 1, 4).reshape(3, s, s)
            assert torch.equal(rebuilt, crop(images, b, r, c, s)[0]), f"{s}px patch at image {b}, cell ({r},{c})"


# 15. Conv count --------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_conv_applied_i_times(num_scales):
    """28 px: one 2x2 conv (2x2 -> 1x1). 56 px: two (4x4 -> 2x2 -> 1x1). One token per big patch."""
    net = make_apt(num_scales)
    images = make_images(2)
    merge = hand_merge(num_scales)
    calls = []
    net.mixed_patch.patch_attn.register_forward_hook(
        lambda m, inp, out: calls.append((tuple(inp[0].shape), tuple(out.shape))))
    d, *_ = run(net, num_scales, images, merge)

    n28 = len(cells(d, 28))
    expected = [((n28, DIM, 2, 2), (n28, DIM, 1, 1))]
    if num_scales > 2:
        n56 = len(cells(d, 56))
        expected += [((n56, DIM, 4, 4), (n56, DIM, 2, 2)), ((n56, DIM, 2, 2), (n56, DIM, 1, 1))]
    print(f"conv calls (input -> output): {calls}")
    assert calls == expected


# 16. Start of training -------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_zero_mlp_start_is_resize_only(num_scales):
    """With ZeroMLP = 0, a big token is just patch_embed(shrunken patch) + its resampled position."""
    net = make_apt(num_scales)
    images = make_images(2, seed=2)
    _, _, tokens, _ = run(net, num_scales, images, hand_merge(num_scales))
    for s in patch_sizes(num_scales)[1:]:
        assert len(tokens[s]) > 0
        pos = resampled_pos(net, s)
        for (b, r, c), v in tokens[s].items():
            resized = F.interpolate(crop(images, b, r, c, s), size=(P, P), mode="bilinear")
            expected = embed(net, resized).flatten() + pos[r * (IMG // s) + c]
            torch.testing.assert_close(v, expected, **TOL, msg=f"{s}px patch at image {b}, cell ({r},{c})")


# 17. Small patches -----------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_small_tokens_match_standard_vit(num_scales):
    """A 14 px token equals the standard ViT patch embedding of that cell + pos_embed[1 + 8*r + c]."""
    net = make_apt(num_scales)
    images = make_images(2, seed=3)
    _, _, tokens, _ = run(net, num_scales, images, hand_merge(num_scales))
    proj = net.mixed_patch.patch_embed.proj
    standard = F.conv2d(images, proj.weight, proj.bias, stride=P)  # (B, D, 8, 8): standard ViT patchify
    assert len(tokens[P]) > 0
    for (b, r, c), v in tokens[P].items():
        expected = standard[b, :, r, c] + net.pos_embed[0, 1 + GRID * r + c]
        torch.testing.assert_close(v, expected, **TOL, msg=f"image {b}, cell ({r},{c})")


# 18. ZeroMLP on --------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_nonzero_zero_mlp_changes_only_big_tokens(num_scales):
    """A non-zero ZeroMLP changes only big tokens, and each big token matches Eq. 2 computed by hand."""
    net = make_apt(num_scales)
    images = make_images(2, seed=4)
    merge = hand_merge(num_scales)
    d, out_before, _, _ = run(net, num_scales, images, merge)
    with torch.no_grad():
        net.mixed_patch.zero_conv.weight.normal_()
        net.mixed_patch.zero_conv.bias.normal_()
        _, out_after, tokens, _ = run(net, num_scales, images, merge)

    small_or_cls = d["output_mask"] <= 1
    assert torch.equal(out_after[small_or_cls], out_before[small_or_cls]), "14 px / CLS tokens changed"
    big = ~small_or_cls
    assert big.any()
    assert ((out_after[big] - out_before[big]).abs().amax(dim=1) > 1e-3).all(), "some big token did not change"

    for s in patch_sizes(num_scales)[1:]:
        for (b, r, c), v in tokens[s].items():
            torch.testing.assert_close(v, eq2(net, images, b, r, c, s), **TOL, msg=f"{s}px, image {b}, cell ({r},{c})")


# 19. Can it learn ------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_zero_mlp_gets_gradient(num_scales):
    """At init the ZeroMLP gets a non-zero gradient (the conv behind it gets none, as in ControlNet);
    after one update of the ZeroMLP the conv starts receiving gradient too."""
    net = make_apt(num_scales)
    images = make_images(2, seed=5)
    merge = hand_merge(num_scales)
    target = torch.randn(1, DIM, generator=torch.Generator().manual_seed(0))
    zero_mlp, conv = net.mixed_patch.zero_conv, net.mixed_patch.patch_attn

    def step():
        net.zero_grad()
        _, out, _, _ = run(net, num_scales, images, merge)
        (out * target).sum().backward()

    step()
    assert zero_mlp.weight.grad.abs().sum() > 0, "ZeroMLP weight gets no gradient at init"
    assert zero_mlp.bias.grad.abs().sum() > 0, "ZeroMLP bias gets no gradient at init"
    assert conv.weight.grad is None or torch.count_nonzero(conv.weight.grad) == 0

    with torch.no_grad():
        zero_mlp.weight -= 0.1 * zero_mlp.weight.grad
    step()
    assert conv.weight.grad.abs().sum() > 0, "conv still gets no gradient after the ZeroMLP moved"


# 20. CLS token ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_scales", [2, 3])
def test_cls_token(num_scales):
    """Every image's sequence starts with exactly one CLS token equal to cls_token + pos_embed[0]."""
    net = make_apt(num_scales)
    images = make_images(3, seed=6)
    merge = hand_merge(num_scales, B=3)  # image 2 has nothing merged -> sequences of different lengths
    d, out, _, cls = run(net, num_scales, images, merge)
    assert len(set(d["seqlens"])) > 1

    expected = net.mixed_patch.cls_token[0, 0] + net.pos_embed[0, 0]
    starts = d["cu_seqlens"][:-1].long()
    assert (d["output_mask"][starts] == -1).all(), "CLS is not first in some sequence"
    assert (d["output_mask"] == -1).sum() == images.shape[0]
    for b, start in enumerate(starts):
        assert cls[b].shape == (1, DIM)
        torch.testing.assert_close(out[start], expected, msg=f"image {b}")


# 21. No big patches at some size ---------------------------------------------------------
NO_BIG_CASES = {
    "2-scale, nothing merged": (2, lambda m: m[28].zero_()),
    "3-scale, no 56px": (3, lambda m: m[56].zero_()),
    "3-scale, no 28px": (3, lambda m: m[28].zero_()),
    "3-scale, nothing merged": (3, lambda m: (m[28].zero_(), m[56].zero_())),
}


@pytest.mark.parametrize("case", list(NO_BIG_CASES))
def test_size_with_no_big_patches(case):
    """A patch size with no selected patches must not crash and must add no tokens."""
    num_scales, clear = NO_BIG_CASES[case]
    net = make_apt(num_scales)
    images = make_images(2, seed=7)
    merge = hand_merge(num_scales)
    clear(merge)
    with torch.no_grad():
        d, out, tokens, _ = run(net, num_scales, images, merge)

    assert torch.isfinite(out).all()
    assert out.shape == (sum(d["seqlens"]), DIM)
    for b in range(images.shape[0]):
        counts = {s: sum(1 for c in tokens[s] if c[0] == b) for s in patch_sizes(num_scales)}
        area = sum(n * (s // P) ** 2 for s, n in counts.items())
        assert area == GRID * GRID, f"image {b}: patches cover {area} of {GRID * GRID} cells"
        assert d["seqlens"][b] == 1 + sum(counts.values())
        for s in patch_sizes(num_scales)[1:]:
            if not merge[s][b].any():
                assert counts[s] == 0
