---
name: numba-kernel
description: Write, debug, or optimise a Numba @njit kernel in this repo. Use when adding a kernel under geom/ fov/ or terrain/, when a numba TypingError or lowering error appears, when a JIT'd function returns wrong values, or when deciding whether to reach for parallel=True/prange. Also use before changing a signature that a cached kernel depends on.
---

# Numba kernels in Shadowcast

The numeric core is `@njit` throughout. Numba's failure modes are unlike normal Python's — a
typing error is a wall of text pointing at the wrong line, and a *silently wrong* kernel is more
likely than a crashing one. This is the order of operations that works here.

## House style

```python
@njit(cache=True, inline="always")   # small helpers: index math, bit twiddling
@njit(cache=True)                    # the kernel itself
@njit(cache=True, parallel=True)     # only where a prange actually pays for itself
```

No `fastmath`, ever. Shadowcasting compares shadow-interval slopes; reassociation changes which
cells are lit, which silently violates radius monotonicity. If you think a kernel needs
`fastmath`, it needs a better algorithm.

`cache=True` on everything. The suite is ~4s warm precisely because kernels are not recompiled
per run.

## Writing a new kernel

1. **Write the plain-NumPy or plain-Python version first**, in the same module or in the test.
   It is the oracle. `fov/reference.py` is exactly this pattern — a brute-force implementation
   whose only job is to disagree with the fast one.
2. **Pin types at the boundary.** Callers pass `np.ascontiguousarray(x, dtype=np.bool_)` (or
   `np.int64`, `np.uint64` for bitsets) before entering. Do not let a kernel accept whatever
   dtype arrives; numba will happily compile a second specialisation and halve your cache hits.
3. **Return arrays, do not allocate inside a prange body.** Allocate in the caller, write into
   slices.
4. **Add the oracle test in the same commit.** Exhaustive over a small grid, plus a random-trial
   comparison over the real terrain, marked `slow` if it takes seconds.

## Debugging

**Always start here:**

```bash
NUMBA_DISABLE_JIT=1 uv run pytest tests/test_fov.py -x -q
```

That runs every kernel as ordinary Python: real tracebacks, real line numbers, `print` and
`breakpoint()` work. Most "numba bugs" are ordinary logic bugs and this finds them in one step.

If it passes with the JIT disabled and fails with it enabled, the bug is genuinely in the
compilation, and it is nearly always one of:

- **An integer that became a float.** Numba propagates `int64 / int64 -> float64`. Use `//`.
- **A numpy scalar leaking in as a Python `int` parameter.** `np.int16(3)` and `3` are different
  types to numba and produce a second specialisation, or a typing error at the call site.
- **A `max`/`min`/`+=` across `prange` iterations.** Numba only recognises a fixed set of
  reduction patterns; anything else races. `fov/table.py:91` documents this exact trap — read
  that comment before writing a reduction over `prange`.
- **A stale on-disk cache** after a signature change. `find . -name __pycache__ -path "*/shadowcast/*" -exec rm -rf {} +`
  then re-run.

For a typing error, read the **last** "During: resolving callee type" line, not the first — that
is the actual expression numba choked on.

## Before reaching for parallel=True

Measure first. `prange` costs thread setup and forces you into numba's reduction rules. The FOV
table build is the one place it earns its keep because each source cell is independent and there
are 512² of them. A kernel called once per frame almost certainly should not be parallel.

```bash
uv run pytest --durations=10 -q     # what is actually slow
```

## Never do these

- Do not add a wall-lighting post-pass or flood-reveal the source's brush in any FOV kernel.
  Both break `FOV(r) == FOV_max AND disc(r)`, and there is a test that will fail. See CLAUDE.md.
- Do not make correctness depend on the FOV table being populated. A miss must fall back to a
  live computation.
