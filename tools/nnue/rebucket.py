"""Convert scheme-1 (768-input) training shards to scheme 2 (king-bucketed). Offline only.

    uv run python -m tools.nnue.rebucket --data tools/nnue/data/lichess-100m \
        --out tools/nnue/data/lichess-100m-k4

**No relabelling and no re-ingestion.** A scheme-1 shard already holds, for every row, the exact
768-input feature set of the position it came from, and a scheme-2 index is that same number plus
`bucket * 768`. So the bucket can be recovered from the row itself: the friendly king is a
feature like any other, on plane 5, and its square is what selects the block. Nothing is decoded
back to a board, nothing is re-evaluated, and `cp`, `wdl`, row order and shard boundaries come
out exactly as they went in. That matters because relabelling 290M rows means another pass over
the source database and a different label distribution, and the point of a warm start is that
the only thing that changes is the feature indexing.

Every row must have exactly one plane-5 feature. A row with none, or with two, or with an index
outside `[0, 768)`, is not a scheme-1 row this can convert, and the whole run stops rather than
writing a shard that is quietly wrong in a way training would never surface: a mis-bucketed row
teaches one block a position that belongs to another, and every downstream check would pass.

`tests/test_king_buckets.py::check_rebucket` is the proof that this agrees with building the
features from the board directly, on real positions, in all four buckets and on mirrors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.nnue.features import BASE_FEATURES, KING_BUCKET, NUM_BUCKETS, PAD
from tools.nnue.nnue_ref import FLAT_SCHEME_VERSION, SCHEME_VERSION

# Planes are our pawn, knight, bishop, rook, queen, king, then theirs, 64 squares each, so the
# friendly king's feature is `5 * 64 + square` and its square is the index less that base.
OUR_KING_PLANE = 5
KING_LOW = OUR_KING_PLANE * 64
KING_HIGH = KING_LOW + 64


class Malformed(Exception):
    """A shard this converter refuses, naming the row that made it refuse."""


def bucket_offsets(indices: np.ndarray, where: str) -> np.ndarray:
    """The `bucket * 768` offset for every row of a padded ``[N, 32]`` scheme-1 index matrix.

    Vectorised because the real shards are 500k rows each, but every refusal names the first
    offending row, because "some row somewhere is wrong" is not a message anyone can act on.
    """
    if indices.ndim != 2:
        raise Malformed(f"{where}: indices has shape {indices.shape}, expected [rows, 32]")
    active = indices != PAD
    if (indices[~active] != PAD).any():
        raise Malformed(f"{where}: padding is not all {PAD}")
    # Scheme-1 indices live in [0, 768). Anything else is a scheme-2 shard, a corrupt one, or a
    # padding value this code does not know about.
    bad = active & ((indices < 0) | (indices >= BASE_FEATURES))
    if bad.any():
        row = int(np.argmax(bad.any(axis=1)))
        value = int(indices[row][bad[row]][0])
        raise Malformed(
            f"{where}: row {row} has feature index {value}, outside the scheme-1 range "
            f"[0, {BASE_FEATURES})"
        )

    kings = active & (indices >= KING_LOW) & (indices < KING_HIGH)
    counts = kings.sum(axis=1)
    if (counts != 1).any():
        row = int(np.argmax(counts != 1))
        raise Malformed(
            f"{where}: row {row} has {int(counts[row])} friendly-king features on plane "
            f"{OUR_KING_PLANE}, expected exactly 1"
        )
    squares = indices[kings].astype(np.int64) - KING_LOW
    offsets: np.ndarray = (KING_BUCKET[squares].astype(np.int64) * BASE_FEATURES).astype(np.int32)
    return offsets


def convert_indices(indices: np.ndarray, where: str) -> np.ndarray:
    """Return the scheme-2 index matrix for a scheme-1 one, same shape, same dtype, same order."""
    offsets = bucket_offsets(indices, where)
    out = indices.astype(np.int32).copy()
    active = indices != PAD
    out[active] += np.repeat(offsets, active.sum(axis=1))
    if (out[active] >= BASE_FEATURES * NUM_BUCKETS).any() or (out[active] < 0).any():
        raise Malformed(f"{where}: a converted index landed outside the scheme-2 range")
    out[~active] = PAD
    return out.astype(indices.dtype)


def convert_shard(source: Path, destination: Path) -> tuple[int, dict[int, int]]:
    """Convert one shard, preserving every array but `indices` and the scheme marker."""
    with np.load(source) as shard:
        contents = {name: shard[name] for name in shard.files}
    scheme = int(contents["scheme"]) if "scheme" in contents else FLAT_SCHEME_VERSION
    if scheme == SCHEME_VERSION:
        raise Malformed(f"{source.name}: already scheme {SCHEME_VERSION}")
    if scheme != FLAT_SCHEME_VERSION:
        raise Malformed(f"{source.name}: scheme {scheme} is not one this converts")
    if "indices" not in contents:
        raise Malformed(f"{source.name}: no indices array")

    indices = contents["indices"]
    offsets = bucket_offsets(indices, source.name)
    spread: dict[int, int] = {}
    for bucket in range(NUM_BUCKETS):
        spread[bucket] = int((offsets == bucket * BASE_FEATURES).sum())
    contents["indices"] = convert_indices(indices, source.name)
    contents["scheme"] = np.int32(SCHEME_VERSION)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **contents)
    return indices.shape[0], spread


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory of scheme-1 shards")
    parser.add_argument("--out", type=Path, required=True, help="where the scheme-2 shards go")
    arguments = parser.parse_args(argv)

    shards = sorted(arguments.data.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no .npz shards under {arguments.data}")
    rows = 0
    spread = dict.fromkeys(range(NUM_BUCKETS), 0)
    for source in shards:
        try:
            written, shard_spread = convert_shard(source, arguments.out / source.name)
        except Malformed as failure:
            raise SystemExit(f"refused: {failure}") from failure
        rows += written
        for bucket, count in shard_spread.items():
            spread[bucket] += count
        print(f"  {source.name}: {written:,} rows")
    share = ", ".join(f"bucket {b} {spread[b] / max(rows, 1):.1%}" for b in sorted(spread))
    print(
        f"converted {len(shards)} shard(s), {rows:,} rows, to scheme {SCHEME_VERSION} in "
        f"{arguments.out}\n  {share}"
    )


if __name__ == "__main__":
    main()
