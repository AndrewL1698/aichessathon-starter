"""Warm-start a king-bucketed net from a trained 768-input one. Offline only.

    uv run python -m tools.nnue.bucketize --weights weights/nnue-768.npz \
        --out weights/nnue.npz
    uv run python -m tools.nnue.bucketize --checkpoint tools/nnue/checkpoints/epoch_060.pt \
        --out tools/nnue/checkpoints-buckets/epoch_000.pt

Both modes do the same thing: the first layer's 768 rows are copied into all four bucket blocks,
and everything else -- layer 2, layer 3, every bias, the hidden width, the quantisation scales and
the target marker -- is carried across untouched.

**Why this reproduces the old net exactly.** Every feature active in one perspective carries that
perspective's bucket, so a position reads 32 rows from one block and never mixes blocks. Four
identical blocks therefore means the same 32 rows are summed whichever bucket the king selects,
and the accumulator, and so the evaluation, is the number the 768 net returned.
`tests/test_king_buckets.py` proves that on 10,000 positions.

**What it is not.** A warm-started file has learned nothing about kings: its four blocks are
identical, which is precisely the statement that the bucket carries no information yet. It is a
starting point for fine-tuning on bucketed features, and until that fine-tuning has happened and
been benched, no strength claim can be made. Do not describe a bucketised file as a king-relative
model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.nnue.features import BASE_FEATURES, NUM_BUCKETS, NUM_FEATURES, tile_rows
from tools.nnue.nnue_ref import SCHEME_VERSION

# The scheme version a 768-input file carries. Anything else is not a warm start candidate.
FLAT_SCHEME_VERSION = 1


def bucketize_weights(source: Path, out: Path) -> Path:
    """Rewrite a quantised scheme-1 npz as a scheme-2 one with four identical bucket blocks."""
    with np.load(source) as data:
        version = int(data["version"])
        if version != FLAT_SCHEME_VERSION:
            raise SystemExit(
                f"{source} is scheme version {version}; only a version {FLAT_SCHEME_VERSION} "
                f"(768-input) file can be warm started into version {SCHEME_VERSION}"
            )
        contents = {name: data[name] for name in data.files}

    contents["version"] = np.int32(SCHEME_VERSION)
    contents["l1_weight"] = tile_rows(contents["l1_weight"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **contents)
    size = out.stat().st_size
    print(
        f"wrote {out} ({size:,} bytes, {size / 1e6:.2f} MB) "
        f"l1_weight {contents['l1_weight'].shape} = {NUM_BUCKETS} identical blocks of "
        f"{BASE_FEATURES}, scheme_version={SCHEME_VERSION}"
    )
    return out


def bucketize_checkpoint(source: Path, out: Path) -> Path:
    """Rewrite a torch checkpoint so its first layer has one identical block per bucket."""
    checkpoint: dict[str, Any] = torch.load(source, map_location="cpu", weights_only=True)
    state = dict(checkpoint["model"])
    weight = state["l1.weight"]
    if weight.shape[1] == NUM_FEATURES:
        raise SystemExit(f"{source} is already king-bucketed ({NUM_FEATURES} inputs)")
    if weight.shape[1] != BASE_FEATURES:
        raise SystemExit(
            f"{source} has {weight.shape[1]} inputs, not the {BASE_FEATURES} a 768-input net has"
        )
    # nn.Linear stores [out, in], so the feature axis is 1 and the blocks are laid end to end
    # in the same order the feature index uses: bucket * 768 + base.
    state["l1.weight"] = weight.repeat(1, NUM_BUCKETS).contiguous()
    checkpoint["model"] = state
    checkpoint["warm_started_from"] = str(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    print(
        f"wrote {out}: l1.weight {tuple(state['l1.weight'].shape)} = {NUM_BUCKETS} identical "
        f"blocks of {BASE_FEATURES}, warm started from {source}"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Warm-start a king-bucketed net from a flat one.")
    parser.add_argument("--weights", type=Path, help="a quantised scheme-1 npz to bucketise")
    parser.add_argument("--checkpoint", type=Path, help="a torch checkpoint to bucketise")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if bool(arguments.weights) == bool(arguments.checkpoint):
        raise SystemExit("pass exactly one of --weights and --checkpoint")
    if arguments.weights:
        bucketize_weights(arguments.weights, arguments.out)
    else:
        bucketize_checkpoint(arguments.checkpoint, arguments.out)


if __name__ == "__main__":
    main()
