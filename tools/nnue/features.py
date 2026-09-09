"""The feature scheme for the learned evaluation: 768 inputs, side-to-move relative.

Offline only. Nothing here is imported by a root module, and nothing here ships.

The runtime (numba) has to reproduce the index formula exactly, so it is stated once, here:

    INDEX = (0 if piece_colour == side_to_move else 6) + (piece_type - 1)) * 64
            + (square ^ (56 if side_to_move == BLACK else 0))

which is 12 planes of 64 squares:

  * plane 0..5   our  pawn, knight, bishop, rook, queen, king   (piece_type 1..6)
  * plane 6..11  their pawn, knight, bishop, rook, queen, king
  * ``square ^ 56`` flips the board vertically (a1<->a8), so when Black is to move the
    net sees the position from Black's side with Black's men on the low ranks.

Colour swap plus vertical flip means the net only ever learns "us versus them". The
consequence a test can check: ``features(board) == features(board.mirror())`` exactly,
because ``chess.Board.mirror()`` is that same flip-and-swap. See test_export.py.

A position holds at most 32 men, so at most 32 features are active. Index arrays are
padded to 32 with -1, which the accumulator loop skips.
"""

from __future__ import annotations

import chess
import numpy as np

# 12 planes x 64 squares.
NUM_FEATURES = 768
# Upper bound on men on the board, and so on active features per position.
MAX_ACTIVE = 32
# Padding value in the fixed-width index arrays. The accumulator skips anything negative.
PAD = -1


def features(board: chess.Board) -> np.ndarray:
    """Return the active feature indices for ``board``, side-to-move relative.

    The result is a 1-D int16 array of length <= 32, unpadded and unsorted.
    """
    flip = 56 if board.turn == chess.BLACK else 0
    us = board.turn
    out = np.empty(MAX_ACTIVE, dtype=np.int16)
    count = 0
    for square, piece in board.piece_map().items():
        plane = (0 if piece.color == us else 6) + piece.piece_type - 1
        out[count] = plane * 64 + (square ^ flip)
        count += 1
    return out[:count]


def encode(boards: list[chess.Board]) -> np.ndarray:
    """Encode a batch of boards into a padded ``[N, 32]`` int16 index matrix.

    Rows are padded with ``PAD`` (-1). This is the on-disk layout of every shard.
    """
    out = np.full((len(boards), MAX_ACTIVE), PAD, dtype=np.int16)
    for row, board in enumerate(boards):
        active = features(board)
        out[row, : active.size] = active
    return out


def to_dense(indices: np.ndarray) -> np.ndarray:
    """Expand a padded ``[N, 32]`` index matrix into a dense ``[N, 768]`` float32 matrix.

    Only used for checking the sparse path; training never materialises this.
    """
    dense = np.zeros((indices.shape[0], NUM_FEATURES), dtype=np.float32)
    rows, cols = np.nonzero(indices >= 0)
    dense[rows, indices[rows, cols]] = 1.0
    return dense
