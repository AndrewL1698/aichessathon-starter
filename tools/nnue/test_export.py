"""Check that the exported int16 net is the net that was trained. Offline only.

Three checks, all on positions drawn from random games so they cover real material imbalances
rather than the start position over and over:

  1. Agreement. The integer inference in ``nnue_ref`` must match the torch float model, in
     centipawns, within the tolerances below. This is the check that would catch a wrong scale,
     a transposed layer, a bias stored on the wrong scale, or an accumulator that wraps.
  2. Exact match through the encoder. Feeding the same position twice must give bit-identical
     integer output; the encoder must be deterministic and order-independent.
  3. The perspective invariant. ``chess.Board.mirror()`` flips the board vertically, swaps
     colours and swaps the side to move, which is exactly the transform the feature scheme
     already applies for Black. So the features are identical and the evaluation must be
     *identical*, not negated: both numbers are "how good this is for the player to move".

     A note, because the negated form is the one people expect: flipping only the side to move
     (a null move) is not a symmetry here and eval(pos) == -eval(pos with turn flipped) is not
     true for this or any NNUE. Those are two different positions with two different feature
     sets. The mirror identity below is the real, exactly checkable statement.

Run from the repo root:

    uv run python -m tools.nnue.test_export --weights weights/nnue.npz \\
        --checkpoint tools/nnue/checkpoints/epoch_002.pt
"""

from __future__ import annotations

import argparse
import random
import sys

import chess
import numpy as np
import torch

from tools.nnue import nnue_ref
from tools.nnue.features import MAX_ACTIVE, PAD, features
from tools.nnue.train import Nnue, densify

# Quantisation is lossy by design: rounding error at each of the three layers accumulates, so a
# few centipawns of disagreement is the scheme working rather than failing. The bounds below are
# set from measurement with room for a net whose weights are larger than the smoke net's. On the
# smoke net the scheme measures 1.9cp mean / 6.5cp worst; the scales that were tried and
# rejected measured 26cp mean / 85cp worst, so these bounds do separate a good export from a
# bad one. A few centipawns is in any case far below the evaluation's own error, and the search
# only ever compares evaluations, so a uniform offset of this size costs nothing.
MEAN_TOLERANCE_CP = 5.0
MAX_TOLERANCE_CP = 25.0
POSITIONS = 1000


def sample_positions(count: int, seed: int = 7) -> list[chess.Board]:
    """Play random games and collect ``count`` distinct, not-in-check positions."""
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(8, 120)):
            if board.is_game_over(claim_draw=False):
                break
            board.push(rng.choice(list(board.legal_moves)))
            if board.is_check() or board.is_game_over(claim_draw=False):
                continue
            boards.append(board.copy(stack=False))
            if len(boards) == count:
                break
    return boards


def encode(boards: list[chess.Board]) -> np.ndarray:
    padded = np.full((len(boards), MAX_ACTIVE), PAD, dtype=np.int16)
    for row, board in enumerate(boards):
        active = features(board)
        padded[row, : active.size] = active
    return padded


def torch_centipawns(model: Nnue, cp_scale: float, indices: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        raw = model(densify(torch.from_numpy(indices), torch.device("cpu"))).numpy()
    return raw * cp_scale


def check_agreement(weights: nnue_ref.Weights, model: Nnue, boards: list[chess.Board]) -> None:
    indices = encode(boards)
    reference = nnue_ref.evaluate_batch(weights, indices).astype(np.float64)
    floating = torch_centipawns(model, float(weights.cp_scale), indices)
    error = np.abs(reference - floating)
    print(
        f"int vs float over {len(boards)} positions: "
        f"mean {error.mean():.2f}cp  p99 {np.percentile(error, 99):.2f}cp  max {error.max():.2f}cp"
    )
    assert error.mean() <= MEAN_TOLERANCE_CP, (
        f"mean quantisation error {error.mean():.2f}cp exceeds {MEAN_TOLERANCE_CP}cp"
    )
    assert error.max() <= MAX_TOLERANCE_CP, (
        f"worst quantisation error {error.max():.2f}cp exceeds {MAX_TOLERANCE_CP}cp"
    )


def check_determinism(weights: nnue_ref.Weights, boards: list[chess.Board]) -> None:
    indices = encode(boards)
    first = nnue_ref.evaluate_batch(weights, indices)
    second = nnue_ref.evaluate_batch(weights, encode(boards))
    assert np.array_equal(first, second), "integer inference is not deterministic"
    print(f"exact match on re-encode: {len(boards)}/{len(boards)} positions identical")


def check_mirror(weights: nnue_ref.Weights, boards: list[chess.Board]) -> None:
    mirrored = [board.mirror() for board in boards]
    plain = encode(boards)
    flipped = encode(mirrored)
    same_features = sum(
        1 for row in range(len(boards)) if set(plain[row].tolist()) == set(flipped[row].tolist())
    )
    assert same_features == len(boards), (
        f"only {same_features}/{len(boards)} positions encode to the same features as their "
        "mirror; the perspective flip in features.py is wrong"
    )
    direct = nnue_ref.evaluate_batch(weights, plain)
    through_mirror = nnue_ref.evaluate_batch(weights, flipped)
    assert np.array_equal(direct, through_mirror), "mirror invariance broken"
    print(f"mirror invariance: {len(boards)}/{len(boards)} positions evaluate identically")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the exported int16 net.")
    parser.add_argument("--weights", default="weights/nnue.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positions", type=int, default=POSITIONS)
    arguments = parser.parse_args(argv)

    weights = nnue_ref.load(arguments.weights)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    model = Nnue(int(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    boards = sample_positions(arguments.positions)
    check_agreement(weights, model, boards)
    check_determinism(weights, boards)
    check_mirror(weights, boards)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
