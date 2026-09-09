"""Integer reference inference. This is the spec the numba runtime must reproduce exactly.

Plain numpy, no torch. Every arithmetic step is the step the runtime will take, in the same
order and the same width, so `test_export.py` comparing this to the torch model is a real check
on the quantisation rather than a check on numpy.

The pipeline, with `qa`, `qb`, `qc` and `cp_scale` read from the weight file:

    acc = l1_bias + sum(l1_weight[i] for each active feature i)   int16   scale qa
    a1  = clip(acc, 0, qa)                                        int16   scale qa
    z2  = l2_bias + a1 . l2_weight                                int32   scale qa*qb
    a2  = clip(z2 // qb, 0, qa)                                   int16   scale qa
    z3  = l3_bias + a2 . l3_weight                                int32   scale qa*qc
    cp  = (z3 * cp_scale) // (qa * qc)                            int32   centipawns

Two details the runtime must copy rather than reinvent:

  * `z2 // qb` is floor division. C-style truncation would differ for negative z2, but every
    negative value is clipped to 0 immediately afterwards, so the two agree. The final
    `// (qa * qc)` is floor division with no clip after it, and there truncation would differ
    by one centipawn, so the runtime must floor.
  * The accumulator is int16 and export.py proves it cannot overflow: for every hidden neuron,
    |l1_bias| plus the 32 largest |l1_weight| in that neuron's column fits in int16. A position
    holds at most 32 men, so no reachable position can push it out of range.
  * Everything after the accumulator is int32 and stays there. At the shipped scales the widest
    intermediates measured are z2 ~ 1.4e6 and z3 * cp_scale ~ 1.9e8, both far inside int32, so
    the runtime needs no int64 anywhere.

The evaluation returned is side-to-move relative, in centipawns, like `agent.evaluate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bumped whenever the feature scheme or the quantisation layout changes, so a stale weight file
# fails loudly instead of being read with the wrong shapes.
SCHEME_VERSION = 1


@dataclass(frozen=True)
class Weights:
    """Quantised weights, exactly as stored in ``weights/nnue.npz``."""

    version: int
    hidden: int
    qa: int
    qb: int
    qc: int
    cp_scale: int
    l1_weight: np.ndarray  # int16 [768, hidden], indexed by feature
    l1_bias: np.ndarray  # int16 [hidden]
    l2_weight: np.ndarray  # int16 [hidden, 32]
    l2_bias: np.ndarray  # int32 [32]
    l3_weight: np.ndarray  # int16 [32]
    l3_bias: int  # int32 scalar


def load(path: str | Path) -> Weights:
    with np.load(Path(path)) as data:
        version = int(data["version"])
        if version != SCHEME_VERSION:
            raise SystemExit(
                f"weight file is scheme version {version}, this code speaks {SCHEME_VERSION}"
            )
        return Weights(
            version=version,
            hidden=int(data["hidden"]),
            qa=int(data["qa"]),
            qb=int(data["qb"]),
            qc=int(data["qc"]),
            cp_scale=int(data["cp_scale"]),
            l1_weight=data["l1_weight"],
            l1_bias=data["l1_bias"],
            l2_weight=data["l2_weight"],
            l2_bias=data["l2_bias"],
            l3_weight=data["l3_weight"],
            l3_bias=int(data["l3_bias"]),
        )


def evaluate(weights: Weights, active: np.ndarray) -> int:
    """Evaluate one position from its active feature indices. Returns centipawns."""
    accumulator = weights.l1_bias.astype(np.int16).copy()
    for index in active:
        if index < 0:
            continue
        accumulator += weights.l1_weight[index]

    first = np.clip(accumulator, 0, weights.qa).astype(np.int32)
    second_raw = weights.l2_bias + first @ weights.l2_weight.astype(np.int32)
    second = np.clip(second_raw // weights.qb, 0, weights.qa).astype(np.int32)
    output = weights.l3_bias + int(second @ weights.l3_weight.astype(np.int32))
    return int(output * weights.cp_scale // (weights.qa * weights.qc))


def evaluate_batch(weights: Weights, indices: np.ndarray) -> np.ndarray:
    """Evaluate a padded ``[N, 32]`` index matrix. Same arithmetic, one row at a time."""
    return np.array([evaluate(weights, row) for row in indices], dtype=np.int32)
