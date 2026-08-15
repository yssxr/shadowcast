"""Codecs, quantisation, and the mixture fit.

Three transforms, and each has to be exactly invertible in TypeScript by a reader that
is a few dozen lines long. That constraint did most of the design work here: anything
requiring a lookup table, a variable-length code, or a branch per element was rejected,
not because it would not compress better but because a reader nobody can check is a
reader nobody should trust.

**Modular deltas.** A delta between two `u8` values ranges over [-255, 255] and does not
fit in a byte, so the encoder writes `(new - old) mod 256` and the decoder adds it back
modulo 256. Exact for every input, including the one that breaks any clamped scheme — a
champion flashing across the map, which is exactly the moment an analyst is looking at.

**XOR for bitmaps.** Vision changes at the edges of a region: a tick flips a few hundred
of 16,384 bits, so the XOR against the previous frame is almost entirely zeros. It is
also its own inverse, which removes a whole class of encoder/decoder mismatch.

**Keyframes, which are not a compression knob.** A scrubber dropped at 11:42 must draw a
frame without decoding from tick zero. Every keyframe row is stored absolute, so a reader
seeks to the nearest one and walks forward at most a keyframe's worth of rows.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from shadowcast import constants as C

__all__ = [
    "apply_codec",
    "dequantise_positions",
    "fit_mixture",
    "fit_mixture_ref",
    "invert_codec",
    "quantise_positions",
    "rasterise_mixture",
]

#: 12-bit positions: 4096 steps over the world span, 3.6 units each.
POSITION_STEPS = 1 << C.POSITION_QUANT_BITS


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------
def _modulus(arr: np.ndarray) -> int:
    return 1 << (8 * arr.dtype.itemsize)


def apply_codec(arr: np.ndarray, codec: str, keyframe: int) -> np.ndarray:
    """Encode along axis 0. Returns a new array of the same shape and dtype."""
    if codec == "raw" or arr.shape[0] <= 1:
        return arr.copy()

    out = arr.copy()
    n = arr.shape[0]
    absolute = _keyframe_rows(n, keyframe)
    prev_index = np.arange(n) - 1
    body = ~absolute
    body[0] = False

    if codec == "delta":
        mod = _modulus(arr)
        diff = (arr[body].astype(np.int64) - arr[prev_index[body]].astype(np.int64)) % mod
        out[body] = diff.astype(arr.dtype)
    elif codec == "xor":
        out[body] = arr[body] ^ arr[prev_index[body]]
    else:
        raise ValueError(f"unknown codec {codec!r}")
    return out


def invert_codec(arr: np.ndarray, codec: str, keyframe: int) -> np.ndarray:
    """Decode along axis 0. The Python reference the TypeScript reader is checked against.

    Sequential by necessity — each row depends on the one before it — but only within a
    keyframe interval, which is what makes seeking cheap.
    """
    if codec == "raw" or arr.shape[0] <= 1:
        return arr.copy()

    out = arr.copy()
    n = arr.shape[0]
    absolute = _keyframe_rows(n, keyframe)
    mod = _modulus(arr)
    for t in range(1, n):
        if absolute[t]:
            continue
        if codec == "delta":
            out[t] = ((out[t].astype(np.int64) + out[t - 1].astype(np.int64)) % mod).astype(
                arr.dtype
            )
        elif codec == "xor":
            out[t] = out[t] ^ out[t - 1]
        else:
            raise ValueError(f"unknown codec {codec!r}")
    return out


def _keyframe_rows(n: int, keyframe: int) -> np.ndarray:
    absolute = np.zeros(n, dtype=bool)
    absolute[0] = True
    if keyframe > 0:
        absolute[::keyframe] = True
    return absolute


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
def quantise_positions(pos: np.ndarray) -> np.ndarray:
    """World (x, z) to 12-bit lattice coordinates, clamped to the map.

    Clamping rather than wrapping: a position outside the world span is a bug upstream,
    and folding it to the far side of the map would place a champion in the enemy base
    rather than at the edge where the error is visible.
    """
    out = np.empty(pos.shape, dtype=np.uint16)
    for axis, origin in enumerate((C.WORLD_MIN_X, C.WORLD_MIN_Z)):
        scaled = (pos[..., axis] - origin) / C.WORLD_SPAN * (POSITION_STEPS - 1)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        out[..., axis] = np.clip(np.rint(scaled), 0, POSITION_STEPS - 1).astype(np.uint16)
    return out


def dequantise_positions(q: np.ndarray) -> np.ndarray:
    out = np.empty(q.shape, dtype=np.float64)
    for axis, origin in enumerate((C.WORLD_MIN_X, C.WORLD_MIN_Z)):
        out[..., axis] = origin + q[..., axis] / (POSITION_STEPS - 1) * C.WORLD_SPAN
    return out


# ---------------------------------------------------------------------------
# The belief mixture
# ---------------------------------------------------------------------------
@njit(cache=True)
def fit_mixture(
    pts: np.ndarray,
    w: np.ndarray,
    centres: np.ndarray,
    warm: bool,
    iterations: int,
    out: np.ndarray,
) -> None:
    """Weighted k-means over particle positions, into `out` as `(k, 4)` = (x, z, w, sigma).

    Compiled because it runs about seventy thousand times per match — once per observer,
    enemy and tick — and a NumPy implementation spends more time in call overhead than in
    arithmetic at 1,024 points and 16 clusters. `fit_mixture_ref` below is the readable
    twin, and a differential test holds them together.

    **`warm` seeds from the previous tick's centres, and that is the point rather than an
    optimisation.** k-means from a fresh initialisation returns the same clusters in a
    different order each tick, and a delta against a permuted set of centres is noise —
    it encodes *larger* than the raw values it replaced. Warm-starting keeps component
    *i* meaning the same blob from tick to tick, which is what leaves the deltas small
    enough for the mixture to fit in the byte budget at all.

    `sigma` is the weighted RMS spread per cluster, so the frontend rasterises a smooth
    field instead of sixteen dots.
    """
    n = pts.shape[0]
    k = centres.shape[0]

    if not warm:
        # Weighted farthest-point seeding: deterministic, and it spreads the initial
        # centres across the support rather than clumping them in the densest blob,
        # which a fixed-seed k-means++ would still do a good fraction of the time.
        best = 0
        for i in range(1, n):
            if w[i] > w[best]:
                best = i
        centres[0, 0] = pts[best, 0]
        centres[0, 1] = pts[best, 1]
        d2 = np.empty(n)
        for i in range(n):
            dx = pts[i, 0] - centres[0, 0]
            dz = pts[i, 1] - centres[0, 1]
            d2[i] = dx * dx + dz * dz
        for c in range(1, k):
            best = 0
            bestv = -1.0
            for i in range(n):
                v = d2[i] * w[i]
                if v > bestv:
                    bestv = v
                    best = i
            centres[c, 0] = pts[best, 0]
            centres[c, 1] = pts[best, 1]
            for i in range(n):
                dx = pts[i, 0] - centres[c, 0]
                dz = pts[i, 1] - centres[c, 1]
                v = dx * dx + dz * dz
                if v < d2[i]:
                    d2[i] = v

    assign = np.zeros(n, dtype=np.int64)
    sx = np.empty(k)
    sz = np.empty(k)
    sw = np.empty(k)
    for _ in range(iterations):
        for c in range(k):
            sx[c] = 0.0
            sz[c] = 0.0
            sw[c] = 0.0
        for i in range(n):
            best = 0
            bestv = 1e30
            for c in range(k):
                dx = pts[i, 0] - centres[c, 0]
                dz = pts[i, 1] - centres[c, 1]
                v = dx * dx + dz * dz
                if v < bestv:
                    bestv = v
                    best = c
            assign[i] = best
            sx[best] += pts[i, 0] * w[i]
            sz[best] += pts[i, 1] * w[i]
            sw[best] += w[i]
        for c in range(k):
            if sw[c] > 0.0:
                centres[c, 0] = sx[c] / sw[c]
                centres[c, 1] = sz[c] / sw[c]
            # An empty cluster KEEPS its centre rather than being re-seeded. Re-seeding
            # would break exactly the identity the warm start exists to preserve, and a
            # zero-weight component costs four bytes that delta to nothing.

    for c in range(k):
        out[c, 0] = centres[c, 0]
        out[c, 1] = centres[c, 1]
        out[c, 2] = sw[c]
        out[c, 3] = 0.0
    for i in range(n):
        c = assign[i]
        dx = pts[i, 0] - centres[c, 0]
        dz = pts[i, 1] - centres[c, 1]
        out[c, 3] += (dx * dx + dz * dz) * w[i]
    for c in range(k):
        if sw[c] > 0.0:
            out[c, 3] = np.sqrt(out[c, 3] / sw[c] / 2.0)


def fit_mixture_ref(
    pts: np.ndarray,
    w: np.ndarray,
    centres: np.ndarray,
    warm: bool = True,
    iterations: int = 5,
) -> np.ndarray:
    """Readable NumPy twin of `fit_mixture`, for the differential test."""
    k = centres.shape[0]
    centres = centres.astype(np.float64).copy()
    if not warm:
        centres[0] = pts[np.argmax(w)]
        d2 = ((pts - centres[0]) ** 2).sum(axis=1)
        for c in range(1, k):
            centres[c] = pts[np.argmax(d2 * w)]
            d2 = np.minimum(d2, ((pts - centres[c]) ** 2).sum(axis=1))

    assign = np.zeros(pts.shape[0], dtype=np.int64)
    mass = np.zeros(k)
    for _ in range(iterations):
        d2 = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        for c in range(k):
            hit = assign == c
            mass[c] = w[hit].sum()
            if mass[c] > 0:
                centres[c] = (pts[hit] * w[hit, None]).sum(axis=0) / mass[c]

    out = np.zeros((k, 4))
    out[:, 0:2] = centres
    out[:, 2] = mass
    for c in range(k):
        hit = assign == c
        if mass[c] > 0:
            var = (((pts[hit] - centres[c]) ** 2).sum(axis=1) * w[hit]).sum() / mass[c]
            out[c, 3] = np.sqrt(max(var, 0.0) / 2.0)
    return out


def rasterise_mixture(
    components: np.ndarray,
    grid: int = C.DISPLAY_BELIEF_GRID,
    min_sigma: float | None = None,
) -> np.ndarray:
    """Render a mixture back to a display grid. The Python twin of what the frontend draws.

    Its job here is validation rather than display: the fit is lossy, and the only
    defensible way to bound that loss is to rasterise both the mixture and the original
    particle cloud onto the same grid and report the divergence between them. A lossy
    encoding whose loss has never been measured is a claim, not a format.
    """
    cell = C.WORLD_SPAN / grid
    if min_sigma is None:
        min_sigma = cell * 0.5
    axis = C.WORLD_MIN_X + (np.arange(grid) + 0.5) * cell
    zaxis = C.WORLD_MIN_Z + (np.arange(grid) + 0.5) * cell
    gx, gz = np.meshgrid(axis, zaxis, indexing="xy")

    out = np.zeros((grid, grid))
    for cx, cz, w, sigma in components:
        if w <= 0:
            continue
        s = max(float(sigma), min_sigma)
        d2 = (gx - cx) ** 2 + (gz - cz) ** 2
        out += w * np.exp(-0.5 * d2 / (s * s)) / (2.0 * np.pi * s * s)
    total = out.sum()
    return out / total if total > 0 else out
