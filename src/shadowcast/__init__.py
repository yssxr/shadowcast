"""Shadowcast — reconstructing League of Legends information state from replays.

Layers, each behind an interface so any one can be rebuilt without the others:

    L0   acquisition        packets/            HuggingFace shards -> local
    L1   normalisation      l1_events/          packets -> typed event tables
    L1.5 resolution         l1_events/resolve/  entity <-> team <-> role, order attribution
    L2   reconstruction     l2_reconstruct/     trajectories + vision -> mask stream
    L3   inference          l3_infer/           masks -> belief -> metrics
    L4   presentation       l4_export/          precomputed artifacts -> static site

Two seams carry the design. `packets/source.py` is the only thing that knows what
a replay looks like, so swapping synthetic data for real is one new file.
`l4_export/spec.py` is the only description of the artifact format, and both the
Python codec and the TypeScript reader are generated from it.
"""

__version__ = "0.1.0"
