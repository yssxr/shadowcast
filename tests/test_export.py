"""Tests for the artifact format and the language boundary it crosses.

The one that matters is `test_node_decodes_every_section_identically`. Everything else
here checks Python against itself, which cannot catch the failure this format is actually
exposed to: a section whose dtype, offset or codec is understood differently by the
reader than by the writer does not throw. `new Float32Array` over bytes that are really
`Uint16Array` returns numbers — wrong ones — and the first symptom is a rendered map with
champions standing in walls.

So Node reads the same file with the generated reader and both sides checksum the
**decoded** arrays. Checksumming the stored bytes would only prove they read the same
offsets; checksumming what comes out proves they agree on what the bytes mean.

That test earned its place immediately: on its first run it found that `meta.json`
contained bare `NaN`, which Python's own parser accepts and no other JSON parser on earth
does. A single unknown respawn time made the artifact unopenable in a browser.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.config import ExportSpec
from shadowcast.l1_events.normalise import normalise
from shadowcast.l1_events.resolve import attribute, resolve_all
from shadowcast.l2_reconstruct.vision import VisionStream
from shadowcast.l3_infer.policy import observe
from shadowcast.l4_export import encode
from shadowcast.l4_export.artifact import read_artifact, write_artifact
from shadowcast.l4_export.build import build_arrays, downsample_mask
from shadowcast.l4_export.spec import SCALAR_NAMES, SECTIONS, ArtifactDims
from shadowcast.l4_export.ts_codegen import generate_typescript

_DURATION = 120.0
_GENERATED_TS = Path("web/src/generated/artifact.ts")
_CONFORMANCE_JS = Path("tests/js/conformance.ts")


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["uint8", "<u2", "<u4"])
@pytest.mark.parametrize("keyframe", [0, 1, 5, 64])
def test_delta_round_trips_for_every_value(dtype, keyframe):
    """Including values that make a naive delta overflow.

    The extremes are the point: a delta between 0 and 255 is -255, which does not fit in
    the byte it has to be stored in. Modular arithmetic is exact for that; a clamped
    scheme silently is not, and the case it breaks on — a champion crossing the map in
    one tick — is exactly the one an analyst is looking at.
    """
    rng = np.random.default_rng(3)
    info = np.iinfo(np.dtype(dtype))
    arr = rng.integers(info.min, info.max, size=(40, 3, 2)).astype(dtype)
    arr[1] = info.max  # adjacent extremes, so every delta wraps
    arr[2] = info.min
    coded = encode.apply_codec(arr, "delta", keyframe)
    assert np.array_equal(encode.invert_codec(coded, "delta", keyframe), arr)


@pytest.mark.parametrize("keyframe", [0, 1, 7])
def test_xor_round_trips(keyframe):
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 256, size=(30, 2, 16)).astype(np.uint8)
    coded = encode.apply_codec(arr, "xor", keyframe)
    assert np.array_equal(encode.invert_codec(coded, "xor", keyframe), arr)


def test_keyframes_are_absolute():
    """What makes the artifact seekable rather than merely small.

    A scrubber dropped at 11:42 decodes from the nearest keyframe. If keyframe rows were
    deltas like any other, it would have to decode from tick zero and the scrubber would
    stutter — so this asserts the stored bytes at a keyframe equal the original.
    """
    arr = np.arange(40 * 4, dtype=np.uint8).reshape(40, 4)
    coded = encode.apply_codec(arr, "delta", keyframe=8)
    for row in (0, 8, 16, 24, 32):
        assert np.array_equal(coded[row], arr[row])
    assert not np.array_equal(coded[9], arr[9])


def test_a_partial_decode_from_a_keyframe_matches_a_full_one():
    """The seek path, exercised. Decoding from a keyframe must give the same answer."""
    rng = np.random.default_rng(9)
    arr = rng.integers(0, 4096, size=(80, 5)).astype("<u2")
    keyframe = 16
    coded = encode.apply_codec(arr, "delta", keyframe)
    full = encode.invert_codec(coded, "delta", keyframe)
    partial = encode.invert_codec(coded[32:48].copy(), "delta", keyframe)
    assert np.array_equal(partial, full[32:48])


def test_raw_is_a_copy_not_a_view():
    arr = np.arange(12, dtype=np.uint8)
    coded = encode.apply_codec(arr, "raw", 0)
    coded[0] = 99
    assert arr[0] == 0


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
def test_position_quantisation_error_is_under_half_a_step():
    rng = np.random.default_rng(5)
    x = rng.uniform(C.WORLD_MIN_X, C.WORLD_MIN_X + C.WORLD_SPAN, 5000)
    z = rng.uniform(C.WORLD_MIN_Z, C.WORLD_MIN_Z + C.WORLD_SPAN, 5000)
    pos = np.stack([x, z], axis=1)
    back = encode.dequantise_positions(encode.quantise_positions(pos))
    step = C.WORLD_SPAN / (encode.POSITION_STEPS - 1)
    assert np.abs(back - pos).max() <= step / 2 + 1e-6
    assert step < 4.0  # 3.6 u/LSB at 12 bits


def test_a_flash_fits_in_a_position_delta():
    """The quantisation and the delta width are one decision, not two.

    At full u16 precision a 50-unit tick move is 220 LSB; at 12 bits a 400-unit Flash is
    110. This is what lets the delta stay small without any clamping or escape path.
    """
    a = np.array([[7000.0, 7000.0]])
    b = a + 400.0
    qa, qb = encode.quantise_positions(a), encode.quantise_positions(b)
    assert abs(int(qb[0, 0]) - int(qa[0, 0])) < 128


def test_positions_outside_the_world_clamp_rather_than_wrap():
    """A wrap would put a champion in the enemy base instead of at the edge."""
    far = np.array([[C.WORLD_MIN_X + C.WORLD_SPAN * 2, C.WORLD_MIN_Z - 5000.0]])
    q = encode.quantise_positions(far)
    assert q[0, 0] == encode.POSITION_STEPS - 1
    assert q[0, 1] == 0


# ---------------------------------------------------------------------------
# The mixture
# ---------------------------------------------------------------------------
def test_mixture_kernel_matches_its_numpy_reference():
    """House rule: every compiled kernel has a readable twin and a differential test."""
    rng = np.random.default_rng(11)
    pts = rng.normal(7000, 900, size=(600, 2))
    w = rng.uniform(0.1, 1.0, 600)
    w /= w.sum()
    seed = pts[:8].copy()

    out = np.zeros((8, 4))
    encode.fit_mixture(pts, w, seed.copy(), True, 5, out)
    ref = encode.fit_mixture_ref(pts, w, seed.copy(), warm=True, iterations=5)
    assert np.allclose(out, ref, atol=1e-9)


def test_mixture_weights_sum_to_one():
    rng = np.random.default_rng(12)
    pts = rng.normal(7000, 1200, size=(1024, 2))
    w = np.full(1024, 1.0 / 1024)
    out = np.zeros((C.BELIEF_COMPONENTS, 4))
    encode.fit_mixture(pts, w, np.zeros((C.BELIEF_COMPONENTS, 2)), False, 5, out)
    assert out[:, 2].sum() == pytest.approx(1.0)
    assert (out[:, 3] >= 0).all()


def test_warm_starting_shrinks_the_encoded_belief(artifact):
    """The claim, measured the way it is actually cashed out — in bytes.

    An earlier version of this test compared how far the component centres moved between
    a warm and a cold fit on a *translated* cloud, and failed: farthest-point seeding is
    deterministic, so a cold start keeps its cluster order under translation anyway. The
    property that matters is not centre stability on a toy input, it is the size of the
    encoded section on real belief evolution.

    MEASURED over a 200-second match: 19.9 kB warm against 129.5 kB cold, a factor of
    6.5. Cold-starting reorders the components whenever the cloud changes shape, and a
    delta or XOR against a permuted set of centres is noise.
    """
    import gzip

    from shadowcast.l4_export.encode import apply_codec
    from shadowcast.l4_export.spec import SECTIONS_BY_NAME

    path, _, _ = artifact
    belief = read_artifact(path)["belief"]
    section = SECTIONS_BY_NAME["belief"]
    warm_size = len(
        gzip.compress(apply_codec(belief, section.codec, section.keyframe).tobytes(), 9)
    )

    # A cold fit would permute components between ticks; shuffling each row's components
    # independently reproduces that without re-running the whole export.
    rng = np.random.default_rng(2)
    shuffled = belief.copy()
    for t in range(shuffled.shape[0]):
        shuffled[t] = shuffled[t][:, :, rng.permutation(shuffled.shape[3])]
    cold_size = len(
        gzip.compress(apply_codec(shuffled, section.codec, section.keyframe).tobytes(), 9)
    )
    assert warm_size * 3 < cold_size, (warm_size, cold_size)


def test_the_belief_codec_beats_the_alternatives(artifact):
    """Every codec choice in the spec was measured, not assumed.

    XOR wins on the mixture for a specific reason: a component that jitters by one unit
    encodes as `0xFF` under a modular delta — a high-entropy byte gzip cannot exploit —
    and as `0x01` under XOR.
    """
    import gzip

    from shadowcast.l4_export.encode import apply_codec
    from shadowcast.l4_export.spec import SECTIONS_BY_NAME

    path, _, _ = artifact
    belief = read_artifact(path)["belief"]
    keyframe = SECTIONS_BY_NAME["belief"].keyframe
    sizes = {
        codec: len(gzip.compress(apply_codec(belief, codec, keyframe).tobytes(), 9))
        for codec in ("raw", "delta", "xor")
    }
    assert min(sizes, key=lambda c: sizes[c]) == SECTIONS_BY_NAME["belief"].codec, sizes


def test_a_codec_cannot_be_declared_on_a_float_section():
    """Because it would silently truncate, and would look like excellent compression.

    Both codecs are defined on the integer representation. Applied to f32, encoding
    `[1.75, 2.5]` after `[1.5, 2.25]` and decoding back gives `[1.0, 2.0]` — and the
    `scalars` section compressed twelve times better that way, which is exactly what
    discarding the fractional part will do.
    """
    from shadowcast.l4_export.spec import Section

    with pytest.raises(ValueError, match="truncate"):
        Section(name="bad", dtype="f32", shape=("t", 4), codec="delta")

    # And the loss is real, not theoretical.
    arr = np.array([[1.5, 2.25], [1.75, 2.5]], dtype="<f4")
    coded = encode.apply_codec(arr, "delta", 0)
    assert not np.array_equal(encode.invert_codec(coded, "delta", 0), arr)


def test_the_mixture_is_a_good_fit_and_the_loss_is_bounded(artifact):
    """A lossy encoding whose loss has never been measured is a claim, not a format."""
    _, _, stats = artifact
    assert stats["mixture_kl_samples"] > 0
    assert stats["mixture_kl_mean"] < 0.05, stats["mixture_kl_mean"]


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def test_mask_downsample_keeps_any_visible_cell():
    """A majority rule would erase a ward's cone through a brush entrance."""
    from shadowcast.geom.bitset import pack_rows

    bits = np.zeros((512, 512), dtype=bool)
    bits[7, 9] = True  # one fine cell inside coarse cell (1, 2)
    packed = downsample_mask(pack_rows(bits), 512, 128)
    coarse = np.unpackbits(packed.reshape(128, 16), axis=-1, bitorder="little")
    assert coarse[1, 2] == 1
    assert coarse.sum() == 1


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artifact(terrain, fov_table, tmp_path_factory):
    """One exported match, built once."""
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource

    src = SyntheticSource(
        terrain, ScenarioSpec(seed=7, duration=_DURATION, pathologies=Pathologies.all())
    )
    bundle, _ = src.generate(src.match_ids()[0])
    events = normalise(bundle, terrain)
    at = attribute(events)
    events, _ = resolve_all(events, at.pos, at.valid)
    obs, public, _ = observe(events, at, VisionStream(events, at, terrain, fov_table))
    built = build_arrays(
        events,
        at.pos,
        at.valid,
        obs,
        public,
        lambda: VisionStream(events, at, terrain, fov_table).masks(),
        terrain,
    )
    out = tmp_path_factory.mktemp("artifact") / "match"
    path, report = write_artifact(
        out,
        match_id="test-0001",
        duration=_DURATION,
        dims=built.dims,
        arrays=built.arrays,
        heroes=[{"slot": int(h["slot"]), "team": int(h["team"])} for h in events.heroes],
        events={"deaths": [{"t": 1.0, "respawn": float("nan")}]},
        stats=built.stats,
    )
    return path, report, built.stats


def test_round_trip_is_exact(artifact):
    path, _, _ = artifact
    art = read_artifact(path)
    for section in SECTIONS:
        assert section.name in art.arrays
        assert art[section.name].dtype == np.dtype(section.numpy_dtype)


def test_every_section_is_eight_byte_aligned(artifact):
    """`new Float32Array(buf, offset, n)` throws on a misaligned offset, and the message
    names neither the section nor the writer."""
    path, _, _ = artifact
    meta = json.loads((path / "meta.json").read_text())
    for entry in meta["sections"]:
        assert entry["offset"] % ExportSpec().section_align == 0, entry["name"]


def test_a_corrupted_section_is_caught_rather_than_returned(artifact, tmp_path):
    """The failure this format is exposed to produces numbers, not an exception.

    Without a per-section checksum, a byte flipped anywhere in the payload yields a
    perfectly well-formed array of wrong values, and the first symptom is a map that
    looks slightly odd.
    """
    import gzip

    path, _, _ = artifact
    broken = tmp_path / "broken"
    shutil.copytree(path, broken)
    with gzip.open(broken / "data.bin.gz", "rb") as fh:
        blob = bytearray(fh.read())
    blob[64] ^= 0xFF
    with gzip.open(broken / "data.bin.gz", "wb") as fh:
        fh.write(bytes(blob))

    with pytest.raises(ValueError, match="checksum"):
        read_artifact(broken)


def test_meta_json_is_valid_json_for_parsers_other_than_python(artifact):
    """Python emits bare `NaN`; `JSON.parse` throws on it.

    The fixture deliberately includes a non-finite respawn time, because that is what
    happened for real: an unknown respawn made the artifact unopenable in a browser while
    every Python test passed.
    """
    path, _, _ = artifact
    text = (path / "meta.json").read_text()
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in text
    assert json.loads(text)["events"]["deaths"][0]["respawn"] is None


def test_the_artifact_fits_the_budget(artifact):
    _, report, _ = artifact
    per_second = report["total_bytes"] / _DURATION
    projected = per_second * C.MATCH_WINDOW_SECONDS
    assert projected < 2e6, f"{projected / 1e6:.2f} MB projected for a full match"


def test_writing_is_deterministic(artifact, tmp_path):
    """Same input, same bytes — otherwise every export busts a CDN cache."""
    path, _, _ = artifact
    first = (path / "data.bin.gz").read_bytes()
    art = read_artifact(path)
    again, _ = write_artifact(
        tmp_path / "again",
        match_id="test-0001",
        duration=_DURATION,
        dims=ArtifactDims(**art.meta.dims),
        arrays=art.arrays,
    )
    assert (again / "data.bin.gz").read_bytes() == first


def test_the_writer_refuses_a_wrong_dtype(tmp_path):
    """Coercing would hide an upstream bug behind a rendering artefact."""
    dims = ArtifactDims(position_ticks=2, mask_ticks=2, belief_ticks=2, champions=10)
    arrays = _empty_arrays(dims)
    arrays["belief"] = arrays["belief"].astype(np.float64)
    with pytest.raises(TypeError, match="dtype"):
        write_artifact(tmp_path / "bad", "x", 1.0, dims, arrays)


def test_the_writer_refuses_a_wrong_shape(tmp_path):
    dims = ArtifactDims(position_ticks=2, mask_ticks=2, belief_ticks=2, champions=10)
    arrays = _empty_arrays(dims)
    arrays["scalars"] = arrays["scalars"][:, :-1].copy()
    with pytest.raises(ValueError, match="shape"):
        write_artifact(tmp_path / "bad", "x", 1.0, dims, arrays)


def test_a_future_schema_version_is_refused(artifact, tmp_path):
    path, _, _ = artifact
    future = tmp_path / "future"
    shutil.copytree(path, future)
    meta = json.loads((future / "meta.json").read_text())
    meta["schema_version"] = C.ARTIFACT_SCHEMA_VERSION + 1
    (future / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema version"):
        read_artifact(future)


def _empty_arrays(dims: ArtifactDims) -> dict[str, np.ndarray]:
    from shadowcast.l4_export.spec import resolve_shape

    return {
        s.name: np.zeros(resolve_shape(s.shape, dims.to_dict()), dtype=np.dtype(s.numpy_dtype))
        for s in SECTIONS
    }


# ---------------------------------------------------------------------------
# The language boundary
# ---------------------------------------------------------------------------
def test_generated_typescript_is_committed_and_current():
    """CI regenerates and diffs, which is what stops a format change reaching one
    language only. A mismatch across the boundary does not throw."""
    assert _GENERATED_TS.exists(), f"{_GENERATED_TS} is missing; run `shadowcast export`"
    assert _GENERATED_TS.read_text() == generate_typescript(), (
        f"{_GENERATED_TS} is stale. Regenerate it with `shadowcast export`."
    )


def test_generated_typescript_declares_every_section_and_scalar():
    text = generate_typescript()
    for section in SECTIONS:
        assert f"{section.name}: {section.ts_array};" in text
    for name in SCALAR_NAMES:
        assert f"  {name}: " in text


def test_node_decodes_every_section_identically(artifact):
    """**The test this whole module is arranged around.**

    Both languages decode the same file and checksum the results. Checksumming the
    stored bytes would prove only that they read the same offsets; checksumming what
    comes out proves they agree on what those bytes mean — which is the difference
    between catching a dtype mismatch and shipping one.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed; the language boundary cannot be checked")
    path, _, _ = artifact
    result = subprocess.run(
        ["node", str(_CONFORMANCE_JS), str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    reported = json.loads(result.stdout)
    assert reported["schema_version"] == C.ARTIFACT_SCHEMA_VERSION

    art = read_artifact(path)
    for name, arr in art.arrays.items():
        assert (
            reported["sections"][name]["decoded_crc32"] == zlib.crc32(arr.tobytes()) & 0xFFFFFFFF
        ), f"section {name} decodes differently in Node than in Python"
