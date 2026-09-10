"""Rebuild a float torch checkpoint from a shipped quantised weight file. Offline only.

    uv run python -m tools.nnue.dequantize --weights weights/nnue.npz \
        --out tools/nnue/checkpoints-dequantised/epoch_000.pt

**Why this exists.** Fine-tuning needs a float checkpoint to start from, and the float
checkpoint of the shipped net lives on the machine that trained it. What ships is the quantised
export. Rather than start a king-bucket fine-tune from random weights -- which would not be a
fine-tune at all, and whose result could not be compared with the net it replaced -- this
reconstructs the float model the export was made from, to within the quantisation step.

**What it can and cannot recover.** Quantisation is `round(w * scale)`, so dequantising is
`q / scale` and the most it can be wrong by is half a step: 1/(2*qa) on layer 1, and
correspondingly on the others. It cannot recover what rounding threw away, and it cannot recover
anything the export did not store. What it does recover is a model whose forward pass reproduces
the shipped net's integer output to within the export's own error budget, which is what a warm
start needs -- the point of a warm start is the position in weight space, not the last bit of it.

`--check` proves that rather than asserting it: it runs the rebuilt float model and
`tools/nnue/nnue_ref.py`'s integer pipeline over the same positions and reports the gap, against
the same 5 cp mean / 25 cp max gate `tools/nnue/test_export.py` uses in the other direction.

The output is a checkpoint `tools.nnue.train` accepts through `--init-checkpoint`, and for a
scheme-1 file it is a 768-input one, so the order is: dequantise, then
`tools.nnue.bucketize --checkpoint` to tile it into four buckets, then train.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from tools.nnue import nnue_ref
from tools.nnue.features import BASE_FEATURES, MAX_ACTIVE, NUM_FEATURES, PAD, features
from tools.nnue.train import Nnue

# The gate `test_export` holds the other direction to, and the same one applies here: the two
# pipelines are the same arithmetic at different precision, so the gap is the quantisation step.
MEAN_TOLERANCE_CP = 5.0
MAX_TOLERANCE_CP = 25.0


def dequantise(weights: nnue_ref.Weights) -> dict[str, torch.Tensor]:
    """The float state dict the quantised weights were rounded from.

    Each layer is divided by the scale `tools/nnue/export.py` multiplied it by, and the biases
    by the scale of the sum they join -- l1_bias on qa, l2_bias on qa*qb, l3_bias on qa*qc --
    because that is how they were stored. `nn.Linear` keeps `[out, in]`, and the export
    transposed layers 1 and 2 on the way out, so both are transposed back here.
    """
    qa, qb, qc = float(weights.qa), float(weights.qb), float(weights.qc)
    return {
        "l1.weight": torch.from_numpy(weights.l1_weight.astype(np.float32).T / qa).contiguous(),
        "l1.bias": torch.from_numpy(weights.l1_bias.astype(np.float32) / qa).contiguous(),
        "l2.weight": torch.from_numpy(weights.l2_weight.astype(np.float32).T / qb).contiguous(),
        "l2.bias": torch.from_numpy(weights.l2_bias.astype(np.float32) / (qa * qb)).contiguous(),
        "l3.weight": torch.from_numpy(
            weights.l3_weight.astype(np.float32).reshape(1, -1) / qc
        ).contiguous(),
        "l3.bias": torch.tensor(
            [float(weights.l3_bias) / (qa * qc)], dtype=torch.float32
        ).contiguous(),
    }


def sample_positions(count: int, seed: int = 20260910) -> list[chess.Board]:
    """Random legal positions from short playouts: what the net will actually be asked about."""
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(2, 60)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
            if rng.random() < 0.25:
                boards.append(board.copy())
                if len(boards) >= count:
                    break
    return boards


def check(model: Nnue, weights: nnue_ref.Weights, boards: list[chess.Board]) -> tuple[float, float]:
    """Mean and worst absolute centipawn gap between the float model and the integer pipeline."""
    width = model.l1.weight.shape[1]
    padded = np.full((len(boards), MAX_ACTIVE), PAD, dtype=np.int16)
    for row, board in enumerate(boards):
        active = features(board)
        if width == BASE_FEATURES:
            active = active % BASE_FEATURES
        padded[row, : active.size] = active
    model.eval()
    with torch.no_grad():
        dense = torch.zeros(len(boards), width + 1)
        dense.scatter_(1, torch.from_numpy(padded).long() + 1, 1.0)
        raw = model(dense[:, 1:]).numpy() * float(weights.cp_scale)
    integer = np.array(
        [nnue_ref.evaluate(weights, row[row >= 0]) for row in padded], dtype=np.float64
    )
    gap = np.abs(raw - integer)
    return float(gap.mean()), float(gap.max())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild a float checkpoint from a weight file.")
    parser.add_argument("--weights", type=Path, default=Path("weights/nnue.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--target", default="cp", choices=("cp", "residual"))
    arguments = parser.parse_args(argv)

    weights = nnue_ref.load(arguments.weights)
    rows = weights.l1_weight.shape[0]
    if rows not in (BASE_FEATURES, NUM_FEATURES):
        raise SystemExit(f"{arguments.weights} has {rows} first-layer rows, not 768 or 3072")
    state = dequantise(weights)
    model = Nnue(weights.hidden)
    if rows == BASE_FEATURES:
        # A scheme-1 file rebuilds a 768-input model; `Nnue` is 3,072 wide, so the layer is
        # resized here and `tools.nnue.bucketize` is what tiles it into four buckets.
        model.l1 = torch.nn.Linear(BASE_FEATURES, weights.hidden)
    model.load_state_dict(state, strict=True)

    mean_cp, worst_cp = check(model, weights, sample_positions(arguments.positions))
    print(
        f"rebuilt {rows} x {weights.hidden} from {arguments.weights} "
        f"(qa {weights.qa}, qb {weights.qb}, qc {weights.qc}, cp_scale {weights.cp_scale})"
    )
    print(
        f"float model against the integer pipeline over {arguments.positions} positions: "
        f"{mean_cp:.2f} cp mean, {worst_cp:.2f} cp worst "
        f"(tolerance {MEAN_TOLERANCE_CP:.0f} / {MAX_TOLERANCE_CP:.0f})"
    )
    if mean_cp > MEAN_TOLERANCE_CP or worst_cp > MAX_TOLERANCE_CP:
        raise SystemExit(
            "the rebuilt model does not reproduce the shipped network inside the quantisation "
            "tolerance; it is not a safe warm start"
        )

    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
        "hidden": weights.hidden,
        "cp_scale": float(weights.cp_scale),
        "target": arguments.target,
        "epoch": 0,
        "val_loss": None,
        "dequantised_from": {
            "path": str(arguments.weights.resolve()),
            "scheme": weights.version,
            "qa": weights.qa,
            "qb": weights.qb,
            "qc": weights.qc,
            "mean_cp": round(mean_cp, 3),
            "max_cp": round(worst_cp, 3),
        },
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, arguments.out)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
