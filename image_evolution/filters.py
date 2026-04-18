"""
Filter catalogue for the Image Evolution Engine.

Each entry in FILTERS exposes:
    apply(img: np.ndarray[H,W,3] uint8, params: dict) -> np.ndarray[H,W,3] uint8
    params: dict[str, (min, max, sigma)]  declaring the search space and
            mutation step size of every evolvable parameter.

Param tuples (min, max, sigma):
    - min / max define the clamp range.
    - sigma is the standard deviation of Gaussian mutation.
    - Params whose name appears in INT_PARAMS are rounded after mutation.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


INT_PARAMS = {"block", "key", "reverse", "levels"}


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img, mode="RGB")


def _from_pil(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def fn_blur(img: np.ndarray, p: dict) -> np.ndarray:
    return _from_pil(_to_pil(img).filter(ImageFilter.GaussianBlur(radius=float(p["radius"]))))


def fn_sharpen(img: np.ndarray, p: dict) -> np.ndarray:
    return _from_pil(ImageEnhance.Sharpness(_to_pil(img)).enhance(1.0 + float(p["strength"])))


def fn_edge(img: np.ndarray, p: dict) -> np.ndarray:
    pil = _to_pil(img).filter(ImageFilter.FIND_EDGES)
    arr = _from_pil(pil).astype(np.float32) / 255.0
    thr = float(p["threshold"])
    mask = (arr.max(axis=-1) > thr).astype(np.float32)[..., None]
    return (arr * mask * 255.0).clip(0, 255).astype(np.uint8)


def fn_hue(img: np.ndarray, p: dict) -> np.ndarray:
    shift = float(p["degrees"]) / 360.0
    hsv = np.asarray(_to_pil(img).convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(shift * 255)) % 256
    return _from_pil(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"))


def fn_sat(img: np.ndarray, p: dict) -> np.ndarray:
    return _from_pil(ImageEnhance.Color(_to_pil(img)).enhance(float(p["factor"])))


def fn_pixelate(img: np.ndarray, p: dict) -> np.ndarray:
    block = max(1, int(p["block"]))
    h, w = img.shape[:2]
    small = _to_pil(img).resize((max(1, w // block), max(1, h // block)), Image.BILINEAR)
    return _from_pil(small.resize((w, h), Image.NEAREST))


def fn_posterize(img: np.ndarray, p: dict) -> np.ndarray:
    bits = max(1, min(8, int(round(math.log2(max(2, int(p["levels"])))))))
    return _from_pil(ImageOps.posterize(_to_pil(img), bits))


def fn_threshold(img: np.ndarray, p: dict) -> np.ndarray:
    luma = img.astype(np.float32) @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    mask = (luma / 255.0 > float(p["level"]))[..., None].astype(np.uint8) * 255
    return np.broadcast_to(mask, img.shape).astype(np.uint8)


def fn_emboss(img: np.ndarray, p: dict) -> np.ndarray:
    pil = _to_pil(img).filter(ImageFilter.EMBOSS)
    arr = _from_pil(pil).astype(np.float32)
    mid = 128.0
    s = float(p["strength"])
    out = mid + (arr - mid) * (0.5 + s)
    return out.clip(0, 255).astype(np.uint8)


def fn_noise(img: np.ndarray, p: dict) -> np.ndarray:
    amount = float(p["amount"])
    noise = np.random.normal(0.0, amount * 255.0, img.shape)
    return (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def fn_invert(img: np.ndarray, p: dict) -> np.ndarray:
    return 255 - img


def fn_vignette(img: np.ndarray, p: dict) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / r.max()
    strength = float(p["strength"])
    mask = (1.0 - strength * (r ** 2)).clip(0.0, 1.0)[..., None]
    return (img.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)


def fn_channel_shift(img: np.ndarray, p: dict) -> np.ndarray:
    dx = int(round(float(p["dx"])))
    dy = int(round(float(p["dy"])))
    out = img.copy()
    out[..., 0] = np.roll(out[..., 0], shift=(dy, dx), axis=(0, 1))
    out[..., 2] = np.roll(out[..., 2], shift=(-dy, -dx), axis=(0, 1))
    return out


def fn_contrast(img: np.ndarray, p: dict) -> np.ndarray:
    return _from_pil(ImageEnhance.Contrast(_to_pil(img)).enhance(float(p["factor"])))


def _luma(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def _pixel_sort_horizontal(img: np.ndarray, threshold: float, key_idx: int, reverse: bool) -> np.ndarray:
    """Sort contiguous runs of pixels whose luma is between threshold and 1-threshold."""
    luma = _luma(img) / 255.0
    lo, hi = min(threshold, 1 - threshold), max(threshold, 1 - threshold)
    in_range = (luma >= lo) & (luma <= hi)
    out = img.copy()

    if key_idx == 0:
        key_arr = luma
    elif key_idx in (1, 2):
        hsv = np.asarray(_to_pil(img).convert("HSV"), dtype=np.float32)
        key_arr = hsv[..., 0] if key_idx == 1 else hsv[..., 1]
    else:
        key_arr = img[..., 0].astype(np.float32)

    for y in range(img.shape[0]):
        row_mask = in_range[y]
        if not row_mask.any():
            continue
        start = None
        for x in range(img.shape[1] + 1):
            inside = x < img.shape[1] and row_mask[x]
            if inside and start is None:
                start = x
            elif not inside and start is not None:
                if x - start > 1:
                    segment = out[y, start:x].copy()
                    keys = key_arr[y, start:x]
                    order = np.argsort(keys, kind="stable")
                    if reverse:
                        order = order[::-1]
                    out[y, start:x] = segment[order]
                start = None
    return out


def fn_pixel_sort(img: np.ndarray, p: dict) -> np.ndarray:
    threshold = float(p["threshold"])
    angle = float(p["angle"]) % 180.0
    key_idx = int(p["key"]) % 4
    reverse = bool(int(p["reverse"]) % 2)

    if abs(angle - 90.0) < 0.5:
        rotated = np.rot90(img, k=1)
        sorted_img = _pixel_sort_horizontal(rotated, threshold, key_idx, reverse)
        return np.rot90(sorted_img, k=-1)
    if angle < 0.5:
        return _pixel_sort_horizontal(img, threshold, key_idx, reverse)

    pil = _to_pil(img)
    rotated = pil.rotate(angle, expand=True, resample=Image.BILINEAR)
    rot_arr = _from_pil(rotated)
    sorted_arr = _pixel_sort_horizontal(rot_arr, threshold, key_idx, reverse)
    back = _to_pil(sorted_arr).rotate(-angle, expand=True, resample=Image.BILINEAR)
    bw, bh = back.size
    h, w = img.shape[:2]
    left = (bw - w) // 2
    top = (bh - h) // 2
    return _from_pil(back.crop((left, top, left + w, top + h)))


def fn_neon_pixels(img: np.ndarray, p: dict) -> np.ndarray:
    density = float(p["density"])
    brightness = float(p["brightness"])
    saturation = float(p["saturation"])
    glow = float(p["glow"])

    h, w = img.shape[:2]
    n = int(density * h * w)
    out = img.copy()
    if n <= 0:
        return out

    ys = np.random.randint(0, h, size=n)
    xs = np.random.randint(0, w, size=n)
    hues = np.random.random(size=n)

    hsv = np.stack(
        [
            (hues * 255).astype(np.uint8),
            np.full(n, int(saturation * 255), dtype=np.uint8),
            np.full(n, int(brightness * 255), dtype=np.uint8),
        ],
        axis=-1,
    )
    hsv_img = Image.fromarray(hsv.reshape(1, n, 3), mode="HSV").convert("RGB")
    rgb = np.asarray(hsv_img, dtype=np.uint8).reshape(n, 3)

    out[ys, xs] = rgb

    if glow > 0.05:
        bloom_mask = np.zeros((h, w, 3), dtype=np.uint8)
        bloom_mask[ys, xs] = rgb
        bloom = _from_pil(_to_pil(bloom_mask).filter(ImageFilter.GaussianBlur(radius=glow)))
        out = np.clip(out.astype(np.int32) + bloom.astype(np.int32), 0, 255).astype(np.uint8)

    return out


FilterFn = Callable[[np.ndarray, dict], np.ndarray]

FILTERS: dict[str, dict] = {
    "blur":          {"apply": fn_blur,          "params": {"radius":    (0.5, 8.0, 1.5)}},
    "sharpen":       {"apply": fn_sharpen,       "params": {"strength":  (0.0, 3.0, 0.5)}},
    "edge":          {"apply": fn_edge,          "params": {"threshold": (0.1, 0.9, 0.3)}},
    "hue_shift":     {"apply": fn_hue,           "params": {"degrees":   (-180.0, 180.0, 30.0)}},
    "saturation":    {"apply": fn_sat,           "params": {"factor":    (0.0, 3.0, 0.8)}},
    "pixelate":      {"apply": fn_pixelate,      "params": {"block":     (2, 32, 4)}},
    "posterize":     {"apply": fn_posterize,     "params": {"levels":    (2, 16, 4)}},
    "threshold":     {"apply": fn_threshold,     "params": {"level":     (0.1, 0.9, 0.5)}},
    "emboss":        {"apply": fn_emboss,        "params": {"strength":  (0.0, 2.0, 0.5)}},
    "noise":         {"apply": fn_noise,         "params": {"amount":    (0.0, 0.4, 0.1)}},
    "invert":        {"apply": fn_invert,        "params": {}},
    "vignette":      {"apply": fn_vignette,      "params": {"strength":  (0.0, 1.0, 0.3)}},
    "channel_shift": {"apply": fn_channel_shift, "params": {"dx": (-20, 20, 4), "dy": (-20, 20, 4)}},
    "contrast":      {"apply": fn_contrast,      "params": {"factor":    (0.2, 3.0, 0.5)}},
    "pixel_sort":    {"apply": fn_pixel_sort,    "params": {
        "threshold": (0.05, 0.95, 0.12),
        "angle":     (0.0, 180.0, 30.0),
        "key":       (0, 3, 1),
        "reverse":   (0, 1, 1),
    }},
    "neon_pixels":   {"apply": fn_neon_pixels,   "params": {
        "density":    (0.001, 0.15, 0.02),
        "brightness": (0.5, 1.0, 0.15),
        "saturation": (0.5, 1.0, 0.15),
        "glow":       (0.0, 2.5, 0.5),
    }},
}


def filter_names() -> list[str]:
    return list(FILTERS.keys())


def random_params(filter_type: str) -> dict:
    spec = FILTERS[filter_type]["params"]
    out = {}
    for name, (lo, hi, _sigma) in spec.items():
        val = random.uniform(lo, hi)
        if name in INT_PARAMS:
            val = int(round(val))
        out[name] = val
    return out


def mutate_one_param(filter_type: str, params: dict) -> dict:
    spec = FILTERS[filter_type]["params"]
    if not spec:
        return params
    name = random.choice(list(spec.keys()))
    lo, hi, sigma = spec[name]
    new_val = params.get(name, (lo + hi) / 2) + random.gauss(0.0, sigma)
    new_val = max(lo, min(hi, new_val))
    if name in INT_PARAMS:
        new_val = int(round(new_val))
    out = dict(params)
    out[name] = new_val
    return out


def apply_op(filter_type: str, params: dict, img: np.ndarray) -> np.ndarray:
    return FILTERS[filter_type]["apply"](img, params)
