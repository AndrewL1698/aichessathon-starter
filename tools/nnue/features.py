"""The feature scheme for the learned evaluation: 3,072 inputs, king-bucketed, side-to-move
relative.

Offline only. Nothing here is imported by a root module, and nothing here ships.

This is the 768-input scheme of v4.1 with one thing added: the index is offset by a bucket read
off the *perspective-oriented friendly king square*. `docs/NNUE_KING_BUCKETS.md` is the spec and
the reason; this file is the definition of record that the other three implementations copy.

    INDEX = KING_BUCKET[our_king_square ^ flip] * 768
            + ((0 if piece_colour == side_to_move else 6) + (piece_type - 1)) * 64
            + (square ^ flip)

    flip = 56 if side_to_move == BLACK else 0

The 768 part is unchanged, so it is still 12 planes of 64 squares:

  * plane 0..5   our  pawn, knight, bishop, rook, queen, king   (piece_type 1..6)
  * plane 6..11  their pawn, knight, bishop, rook, queen, king
  * ``square ^ 56`` flips the board vertically (a1<->a8), so when Black is to move the
    net sees the position from Black's side with Black's men on the low ranks.

And the bucket is:

    KING_BUCKET[square] = 2 * (rank >= 4) + (file >= 4)

on that same flipped square, so bucket 0 is "own half, queenside" for whichever side is looking.
Colour swap plus vertical flip means the net only ever learns "us versus them", and applying the
flip before the bucket lookup keeps that true of the bucket as well. The consequence a test can
check is unchanged: ``features(board) == features(board.mirror())`` exactly, because
``chess.Board.mirror()`` is that same flip-and-swap. See test_export.py and
tests/test_king_buckets.py.

A position holds at most 32 men, so at most 32 features are active, and every one of them carries
the same bucket -- a perspective's features all live in one 768-row block. Index arrays are padded
to 32 with -1, which the accumulator loop skips.
"""

from __future__ import annotations

import chess
import numpy as np

# 12 planes x 64 squares: one bucket's worth of features.
BASE_FEATURES = 768
# King buckets, indexed by the perspective-oriented friendly king square.
NUM_BUCKETS = 4
# The whole first layer: one 768-row block per bucket.
NUM_FEATURES = BASE_FEATURES * NUM_BUCKETS
# Upper bound on men on the board, and so on active features per position.
MAX_ACTIVE = 32
# Padding value in the fixed-width index arrays. The accumulator skips anything negative.
PAD = -1

# KING_BUCKET[oriented square] -> bucket. Built from the formula rather than written out, so the
# table and the spec cannot drift apart. a1 = 0, h8 = 63, already flipped for the perspective.
KING_BUCKET = np.array(
    [2 * (square // 8 >= 4) + (square % 8 >= 4) for square in range(64)], dtype=np.int16
)


def king_bucket(oriented_king_square: int) -> int:
    """The bucket for a friendly king already flipped into its own perspective."""
    return int(KING_BUCKET[oriented_king_square])


def tile_rows(l1_weight: np.ndarray) -> np.ndarray:
    """Copy a ``[768, hidden]`` first layer into the ``[3072, hidden]`` four-block layout.

    The warm start, stated once here because three places need it to agree: the offline
    bucketiser, the reference inference, and the runtime's own loader (which cannot import this
    module, because this module does not ship, and so repeats the two lines under a test that
    checks the two agree).

    Four identical blocks evaluate every position to the integer the 768 net returned, because
    a position's features all carry one bucket and so read one block. It is a starting point
    for fine-tuning and not a king-relative model.
    """
    if l1_weight.shape[0] != BASE_FEATURES:
        raise ValueError(
            f"the first layer has {l1_weight.shape[0]} rows, not the {BASE_FEATURES} a "
            f"768-input net has; this is not a warm-start candidate"
        )
    return np.ascontiguousarray(np.tile(l1_weight, (NUM_BUCKETS, 1)))


def features(board: chess.Board) -> np.ndarray:
    """Return the active feature indices for ``board``, side-to-move relative.

    The result is a 1-D int16 array of length <= 32, unpadded and unsorted. Every index in it
    lies in the same 768-row block, the one this side's king square selects.
    """
    flip = 56 if board.turn == chess.BLACK else 0
    us = board.turn
    king = board.king(us)
    if king is None:
        raise ValueError("a position with no king for the side to move has no king bucket")
    offset = int(KING_BUCKET[king ^ flip]) * BASE_FEATURES
    out = np.empty(MAX_ACTIVE, dtype=np.int16)
    count = 0
    for square, piece in board.piece_map().items():
        plane = (0 if piece.color == us else 6) + piece.piece_type - 1
        out[count] = offset + plane * 64 + (square ^ flip)
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
    """Expand a padded ``[N, 32]`` index matrix into a dense ``[N, 3072]`` float32 matrix.

    Only used for checking the sparse path; training never materialises this.
    """
    dense = np.zeros((indices.shape[0], NUM_FEATURES), dtype=np.float32)
    rows, cols = np.nonzero(indices >= 0)
    dense[rows, indices[rows, cols]] = 1.0
    return dense
