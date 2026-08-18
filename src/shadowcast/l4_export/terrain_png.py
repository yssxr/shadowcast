"""The terrain, as one PNG shipped once for the whole corpus.

Terrain is the only thing every match shares, so it does not belong in a per-match
artifact, 25 kB times a thousand matches is 25 MB of identical bytes and a thousand
cache misses. It goes out once, and every match's artifact is smaller for it.

Three channels, matching the three the engine carries:

    R  walkable        255 where a champion can stand
    G  brush_id        0 outside brush, otherwise the patch id (1..40), so the frontend
                       can tint a single brush on hover without a second lookup
    B  blocks_vision   255 where vision stops

**`blocks_vision` is a separate channel from `walkable` on purpose**, and this is the one
thing about the file that would look redundant to someone reading it cold. Riot stamps
see-through cells along wall diagonals, 1,819 of them on Summoner's Rift, which block
movement but transmit vision, added after S5 Worlds specifically to fix line-of-sight
artefacts. A frontend that derived one channel from the other would draw shadows Riot
patched out eight years ago.

PNG rather than a raw section because the browser decodes it on the GPU thread for free,
and because a terrain file that can be opened and looked at is worth more than a few
kilobytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shadowcast.terrain.terrain import Terrain

__all__ = ["TERRAIN_PNG_NAME", "write_terrain_png"]

TERRAIN_PNG_NAME = "terrain.png"


def write_terrain_png(terrain: Terrain, path: Path | str) -> tuple[Path, dict[str, object]]:
    """Write the three-channel terrain PNG. Returns the path and a size report."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = terrain.grid
    rgb = np.zeros((grid, grid, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(terrain.walkable, 255, 0)
    # Brush ids run to 40, so they fit a byte directly and stay legible in a pixel
    # inspector, which matters more here than the two bits it would save to pack them.
    rgb[..., 1] = np.clip(terrain.brush_id, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.where(terrain.blocks_vision, 255, 0)

    # Row 0 is z-minimum, and the frontend draws z upward, so the image is flipped here
    # rather than in the renderer. Doing it once at write time means no consumer has to
    # remember, and a consumer that forgets would render Summoner's Rift upside down,
    # visible, but only if you already know which base is which.
    Image.fromarray(rgb[::-1], mode="RGB").save(path, optimize=True)

    return path, {
        "grid": grid,
        "bytes": path.stat().st_size,
        "walkable": int(terrain.walkable.sum()),
        "brush_groups": int(terrain.n_brush_groups),
        "vision_blocking": int(terrain.blocks_vision.sum()),
        "terrain_hash": terrain.spec.content_hash,
    }
