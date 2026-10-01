"""Gaze-foveated patch layout (Foveated Patching Framework, Steps 1-3).

APT decides patch sizes from image entropy. Here they come from gaze: a quadtree cell of size
2^i * p is kept whole if the eccentricity of its centre (angle from the gaze direction) is at least
rho_i, otherwise it is split into four. Everything downstream - patch aggregation (APT Eq. 2) and
positional-encoding interpolation (APT Sec. 3.3) - is APT's own code, unchanged: the layout is handed
to APT as importance maps (1 = split, 0 = keep the big patch, threshold 0.5).

  Step 1  base patch size p      = the encoder's patch-embedding kernel  (base_patch_size)
  Step 2  scale ladder S         = p * 2^i, image size a multiple of the largest  (scale_ladder)
  Step 3  ring radii in degrees  = FOVI recipe, uniform in log(r + a)  (ring_radii_deg)

Degrees <-> pixels uses a pinhole camera with a given horizontal field of view. (The calibrated
fisheye model is Step 4 and not part of this module; swap `eccentricity_deg` to use it.)

Gaze is given per image as (u, v) in [0, 1] image coordinates (u to the right, v down).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .entropy_utils import select_patches_by_threshold
from .patch_tokenizer import PatchTokenizer


# Step 1 ------------------------------------------------------------------------------------
def base_patch_size(net) -> int:
    """p is not a choice: it is the kernel of the encoder's patch-embedding conv (weights [d, 3, p, p])."""
    k = net.patch_embed.proj.kernel_size
    assert k[0] == k[1], f"non-square patch embedding {k}"
    return int(k[0])


# Step 2 ------------------------------------------------------------------------------------
def scale_ladder(p: int, num_scales: int, img_size: int) -> List[int]:
    """{p, 2p, ..., 2^(S-1) p}. The coarsest level tiles the image, so img_size must be a multiple of it."""
    sizes = [p * 2 ** i for i in range(num_scales)]
    largest = sizes[-1]
    if img_size % largest:
        lo = img_size // largest * largest
        raise ValueError(f"image size {img_size} is not a multiple of the largest patch {largest}; "
                         f"use e.g. {lo or largest} or {lo + largest}")
    return sizes


# Step 3 ------------------------------------------------------------------------------------
def ring_radii_deg(R_deg: float, a: float, num_scales: int) -> List[float]:
    """FOVI recipe: S-1 interior ring boundaries, evenly spaced in u = log(r + a) on [log a, log(R + a)].
    Ring i (0-based) separates patch size p*2^i (inside) from p*2^(i+1) (outside).
    Small a -> small fovea, rings bunched near the centre; large a -> rings close to evenly spaced."""
    assert R_deg > 0 and a > 0 and num_scales >= 1
    lo, hi = math.log(a), math.log(R_deg + a)
    return [math.exp(lo + (hi - lo) * k / num_scales) - a for k in range(1, num_scales)]


# Degrees <-> pixels (pinhole) ----------------------------------------------------------------
def focal_px(img_size: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels for a square image whose horizontal field of view is fov_deg."""
    return (img_size / 2) / math.tan(math.radians(fov_deg) / 2)


def eccentricity_deg(x: torch.Tensor, y: torch.Tensor, gaze_xy: torch.Tensor, img_size: int, fov_deg: float):
    """Angle (degrees) between the camera rays through pixel(s) (x, y) and through the gaze pixel.
    x, y: tensors of pixel coordinates (any shape); gaze_xy: (B, 2) pixel coordinates.
    Returns (B, *x.shape). Uses atan2(|a x b|, a . b), which stays exact near 0 degrees."""
    f, c = focal_px(img_size, fov_deg), img_size / 2
    ray = torch.stack([x - c, y - c, torch.full_like(x, f)], dim=-1).double().unsqueeze(0)  # (1, *, 3)
    g = torch.cat([gaze_xy - c, torch.full_like(gaze_xy[:, :1], f)], dim=-1).double()      # (B, 3)
    g = g.view(g.shape[0], *([1] * x.dim()), 3)                                            # (B, 1.., 3)
    cross = torch.linalg.cross(ray, g, dim=-1)  # broadcasts to (B, *, 3)
    dot = (ray * g).sum(-1)
    return torch.rad2deg(torch.atan2(cross.norm(dim=-1), dot)).float()


def ring_radii_px_at_centre(cfg: "FoveaConfig") -> List[float]:
    """Ring radii in pixels for a gaze at the image centre (pinhole: r = f tan(theta)). For reporting."""
    f = focal_px(cfg.img_size, cfg.fov_deg)
    return [f * math.tan(math.radians(r)) for r in cfg.rings_deg]


# Config ------------------------------------------------------------------------------------
@dataclass
class FoveaConfig:
    p: int = 14                    # Step 1: from the encoder (14 for CLIP ViT-L/14, SigLIP-so400m/14)
    num_scales: int = 3            # Step 2: S = 3 -> {14, 28, 56}
    img_size: int = 336            # multiple of the largest patch (336 = LLaVA-1.5 / CLIP-L-336 input)
    fov_deg: float = 110.0         # horizontal field of view of the image (Aria RGB ~110 deg)
    a: float = 40.0                # Step 3: FOVI fall-off parameter, tuned to the token budget
    R_deg: Optional[float] = None  # max eccentricity of interest; default = half the field of view
    sizes: List[int] = field(init=False)
    rings_deg: List[float] = field(init=False)

    def __post_init__(self):
        if self.R_deg is None:
            self.R_deg = self.fov_deg / 2
        self.sizes = scale_ladder(self.p, self.num_scales, self.img_size)
        self.rings_deg = ring_radii_deg(self.R_deg, self.a, self.num_scales)

    @classmethod
    def for_encoder(cls, net, **kwargs) -> "FoveaConfig":
        return cls(p=base_patch_size(net), **kwargs)


# Layout --------------------------------------------------------------------------------------
def gaze_to_px(gaze_uv, img_size: int) -> torch.Tensor:
    """(B, 2) gaze in [0, 1] -> pixel coordinates."""
    return torch.as_tensor(gaze_uv, dtype=torch.float32).view(-1, 2) * img_size


def eccentricity_maps(cfg: FoveaConfig, gaze_uv) -> Dict[int, torch.Tensor]:
    """{patch size s: (B, img/s, img/s) eccentricity in degrees of every s-cell centre}."""
    gaze = gaze_to_px(gaze_uv, cfg.img_size)
    maps = {}
    for s in cfg.sizes:
        centres = (torch.arange(cfg.img_size // s, dtype=torch.float32) + 0.5) * s
        y, x = torch.meshgrid(centres, centres, indexing="ij")
        maps[s] = eccentricity_deg(x, y, gaze, cfg.img_size, cfg.fov_deg)
    return maps


def foveated_importance_maps(cfg: FoveaConfig, gaze_uv, device=None) -> Dict[int, torch.Tensor]:
    """The eccentricity predicate as APT importance maps: 1 = split (ecc < rho), 0 = keep big (ecc >= rho).
    With APT thresholds of 0.5 this reproduces 'keep cell at size 2^i p iff ecc(centre) >= rho_i'."""
    ecc = eccentricity_maps(cfg, gaze_uv)
    maps = {cfg.p: torch.ones_like(ecc[cfg.p])}
    for i, s in enumerate(cfg.sizes[1:]):
        maps[s] = (ecc[s] < cfg.rings_deg[i]).float()
    return {s: m.to(device) for s, m in maps.items()} if device is not None else maps


def foveated_masks(cfg: FoveaConfig, gaze_uv) -> Dict[int, torch.Tensor]:
    """{patch size: (B, g, g) 0/1 mask of the chosen patches}, via APT's own quadtree selection."""
    return select_patches_by_threshold(foveated_importance_maps(cfg, gaze_uv), thresholds=[0.5] * (cfg.num_scales - 1))


def masks_to_layout(masks: Dict[int, torch.Tensor]) -> List[List[Tuple[int, int, int]]]:
    """Per image: list of (x, y, size) of the chosen patches, top-left corner in pixels."""
    B = next(iter(masks.values())).shape[0]
    layout = [[] for _ in range(B)]
    for s, m in sorted(masks.items()):
        for b, r, c in m.nonzero().tolist():
            layout[b].append((c * s, r * s, s))
    return layout


def foveated_layout(cfg: FoveaConfig, gaze_uv) -> List[List[Tuple[int, int, int]]]:
    return masks_to_layout(foveated_masks(cfg, gaze_uv))


# Hook into APT ---------------------------------------------------------------------------------
class FoveatedTokenizer(PatchTokenizer):
    """APT PatchTokenizer whose patch sizes come from gaze instead of entropy.

    Output dict is exactly APT's, so it feeds `net.mixed_patch(images, net.pos_embed, d)` - APT's patch
    aggregation (Eq. 2) and positional-encoding interpolation - with no other change.
    """

    def __init__(self, cfg: FoveaConfig, mean, std):
        super().__init__(
            num_scales=cfg.num_scales, base_patch_size=cfg.p, image_size=cfg.img_size,
            thresholds=[0.5] * (cfg.num_scales - 1), mean=mean, std=std, method="entropy",
        )
        self.cfg = cfg

    def forward(self, images: torch.Tensor, gaze_uv) -> Dict:
        assert images.shape[-1] == images.shape[-2] == self.cfg.img_size, "resize images to cfg.img_size first"
        maps = foveated_importance_maps(self.cfg, gaze_uv, device=images.device)
        assert maps[self.cfg.p].shape[0] == images.shape[0], "one gaze per image"
        return super().forward(images, importance_maps=maps)


def layout_from_input_dict(d: Dict, cfg: FoveaConfig) -> List[List[Tuple[int, int, int]]]:
    """(x, y, size) per image, read back from the tokenizer output (to check what APT actually received)."""
    masks = {}
    for s in cfg.sizes:
        g = cfg.img_size // s
        masks[s] = d[f"pos_embed_mask_{s}"].view(-1, g, g)
    return masks_to_layout(masks)
