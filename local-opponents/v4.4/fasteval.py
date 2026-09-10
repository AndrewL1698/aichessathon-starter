"""The tapered evaluation of `agent.py`, rewritten as `@njit` code over the `fastboard` state.

This is a port, not a redesign. Every number below is copied from the `evaluate` in `agent.py`
that PR #5 introduces, and the two are held to being *exactly* equal, not merely close:
`tests/test_fasteval.py` imports that `evaluate` and compares it against this one on two
thousand random positions, and any disagreement at all is a bug here. Read the reasoning for a
weight there; this file deliberately keeps the comments that say what a number is for and drops
the ones that only describe python-chess.

The shape is different because the data is. `agent.py` walks six bitboards; here there is one
10x12 mailbox, so everything is one pass over squares 21..98 that accumulates, at the same
time, the tapered piece-square totals and the handful of summaries the positional terms need:

- `wp` / `bp`  a pawn's rank set per file, packed eight bits per file into one int64, so
               `(wp >> (8 * file)) & 255` is the set of ranks White has a pawn on in that file.
               Every pawn term (passed, isolated, doubled) and both rook-file terms are a mask
               test against those, which is what the bitboard code was doing by another route.
- `w_rooks` / `b_rooks`  a rook count per file, packed four bits per file, so the open-file
               scan afterwards is eight iterations rather than a second walk of the board.
- material, man counts, and per-piece-type counts for the phase, the bishop pair, the bare
  king tests and the drawish scaling.

Nothing allocates: the packed summaries are plain integers in registers, so `evaluate` runs
without touching the heap, which is what lets a search call it a million times a second.

Squares are mailbox indices (a8 is 21, h1 is 98). `FILE_OF` and `RANK_OF` map one to a file
0..7 and a rank 0..7 counted from White's home rank, which is the orientation every table and
every mask below is written in.
"""

import time

import numpy as np
from numba import njit
from numba import types as nbt

# Every `@njit` below compiles as it is decorated, so what `COMPILE_SECONDS` at the foot of
# this file spans is the compilation of the whole module, not only the `warm()` call at the
# end of it. `agent.py` prints it, so the platform's log says what the init budget went on.
_STARTED = time.perf_counter()

# --------------------------------------------------------------------------------------
# The weights. Copied verbatim from `evaluate` and its constants in `agent.py`; that file is
# where they are defined and where the reasoning for each one lives. They are duplicated
# rather than imported because `agent.py` imports python-chess and this must not.
# `tests/test_fasteval.py` asserts every table and constant here is equal to the one there,
# so the two cannot drift apart without a test going red.
# --------------------------------------------------------------------------------------

# Indexed by piece type minus one: pawn, knight, bishop, rook, queen, king.
PIECE_VALUES: tuple[int, ...] = (100, 320, 330, 500, 900, 0)

# The evaluation is written twice, once for a board full of pieces and once for a board that is
# nearly bare, and the two are mixed by how much material is left. The same square is worth
# different things at the two ends: a king belongs behind its pawns in one and in the middle of
# the board in the other, and a pawn on the sixth rank is a nuisance in one and a queen in the
# other. A single table has to compromise between them and gets both wrong.
#
# Phase counts non-pawn material: a minor is 1, a rook 2, a queen 4, so a full board is 24 and
# a pawn endgame is 0. Promotions can push it past 24, so it is clamped.
PHASE_ROOK = 2
PHASE_QUEEN = 4
PHASE_MAX = 24

type PieceSquareTable = tuple[tuple[int, ...], ...]

# Each row is one rank, starting at White's home rank. A Black piece looks up the vertically
# mirrored square. The bonuses refine material without being large enough to overpower it.

# Middlegame pawns: take the centre, and leave the pawns in front of a castled king alone.
# The -20 on d2/e2 is what makes the engine push them; the +10 on the wing pawns beside them
# is what stops it from opening its own king for nothing.
PAWN_MG: PieceSquareTable = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (5, 10, 10, -20, -20, 10, 10, 5),
    (5, -5, -10, 0, 0, -10, -5, 5),
    (0, 0, 0, 20, 20, 0, 0, 0),
    (5, 5, 10, 25, 25, 10, 5, 5),
    (10, 10, 20, 30, 30, 20, 10, 10),
    (50, 50, 50, 50, 50, 50, 50, 50),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# Endgame pawns: a pawn's only ambition is the eighth rank, and no file is better than another
# for it, so the table is flat across each rank and grows steeply up the board. Passed-pawn
# bonuses are added on top of this; these numbers are what an ordinary pawn is worth.
PAWN_EG: PieceSquareTable = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (5, 5, 5, 5, 5, 5, 5, 5),
    (15, 15, 15, 15, 15, 15, 15, 15),
    (30, 30, 30, 30, 30, 30, 30, 30),
    (55, 55, 55, 55, 55, 55, 55, 55),
    (90, 90, 90, 90, 90, 90, 90, 90),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# Middlegame knights: a knight on the rim reaches four squares and a knight in the centre
# reaches eight, so the table is a bowl. The small plus on the second rank is development.
KNIGHT_MG: PieceSquareTable = (
    (-50, -40, -30, -30, -30, -30, -40, -50),
    (-40, -20, 0, 5, 5, 0, -20, -40),
    (-30, 5, 10, 15, 15, 10, 5, -30),
    (-30, 0, 15, 20, 20, 15, 0, -30),
    (-30, 5, 15, 20, 20, 15, 5, -30),
    (-30, 0, 10, 15, 15, 10, 0, -30),
    (-40, -20, 0, 0, 0, 0, -20, -40),
    (-50, -40, -30, -30, -30, -30, -40, -50),
)

# Endgame knights: the same bowl, symmetric about the middle rank because there is no home
# rank to develop from any more, and a little shallower because with few targets left the
# difference between a good square and a bad one matters less than it did.
KNIGHT_EG: PieceSquareTable = (
    (-40, -30, -20, -20, -20, -20, -30, -40),
    (-30, -10, 0, 0, 0, 0, -10, -30),
    (-20, 0, 10, 15, 15, 10, 0, -20),
    (-20, 5, 15, 20, 20, 15, 5, -20),
    (-20, 5, 15, 20, 20, 15, 5, -20),
    (-20, 0, 10, 15, 15, 10, 0, -20),
    (-30, -10, 0, 0, 0, 0, -10, -30),
    (-40, -30, -20, -20, -20, -20, -30, -40),
)

# Middlegame bishops: long diagonals and the two fianchetto squares, corners punished.
BISHOP_MG: PieceSquareTable = (
    (-20, -10, -10, -10, -10, -10, -10, -20),
    (-10, 5, 0, 0, 0, 0, 5, -10),
    (-10, 10, 10, 10, 10, 10, 10, -10),
    (-10, 0, 10, 10, 10, 10, 0, -10),
    (-10, 5, 5, 10, 10, 5, 5, -10),
    (-10, 0, 5, 10, 10, 5, 0, -10),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-20, -10, -10, -10, -10, -10, -10, -20),
)

# Endgame bishops: centralisation only. The fianchetto squares mean nothing once the king is
# not sitting next to them, and the bishop wants to see both wings at once.
BISHOP_EG: PieceSquareTable = (
    (-15, -10, -10, -5, -5, -10, -10, -15),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-10, 0, 5, 10, 10, 5, 0, -10),
    (-5, 5, 10, 15, 15, 10, 5, -5),
    (-5, 5, 10, 15, 15, 10, 5, -5),
    (-10, 0, 5, 10, 10, 5, 0, -10),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-15, -10, -10, -5, -5, -10, -10, -15),
)

# Middlegame rooks: the seventh rank, and the centre files where the pawns come off first.
# Rook on an open file is worth much more than this and is scored separately.
ROOK_MG: PieceSquareTable = (
    (0, 0, 0, 5, 5, 0, 0, 0),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (5, 10, 10, 10, 10, 10, 10, 5),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# Endgame rooks: almost flat. A rook is a rook wherever it stands once the board is empty, so
# the only thing left to say is that an active rook up the board cuts the enemy king off.
ROOK_EG: PieceSquareTable = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 5, 5, 5, 5, 0, 0),
    (0, 0, 5, 5, 5, 5, 0, 0),
    (5, 5, 5, 5, 5, 5, 5, 5),
    (10, 10, 10, 10, 10, 10, 10, 10),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# Middlegame queen: keep her off the rim and out of the game before the minors are out.
QUEEN_MG: PieceSquareTable = (
    (-20, -10, -10, -5, -5, -10, -10, -20),
    (-10, 0, 5, 0, 0, 0, 0, -10),
    (-10, 5, 5, 5, 5, 5, 0, -10),
    (0, 0, 5, 5, 5, 5, 0, -5),
    (-5, 0, 5, 5, 5, 5, 0, -5),
    (-10, 0, 5, 5, 5, 5, 0, -10),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-20, -10, -10, -5, -5, -10, -10, -20),
)

# Endgame queen: centralised, where she covers both wings. Symmetric: there is no development
# left to encourage, only activity.
QUEEN_EG: PieceSquareTable = (
    (-20, -15, -10, -5, -5, -10, -15, -20),
    (-15, -5, 0, 0, 0, 0, -5, -15),
    (-10, 0, 5, 5, 5, 5, 0, -10),
    (-5, 0, 5, 10, 10, 5, 0, -5),
    (-5, 0, 5, 10, 10, 5, 0, -5),
    (-10, 0, 5, 5, 5, 5, 0, -10),
    (-15, -5, 0, 0, 0, 0, -5, -15),
    (-20, -15, -10, -5, -5, -10, -15, -20),
)

# Middlegame king: behind its own pawns, and the castled corners are the two peaks. Everything
# from the third rank up is punished hard enough that the king only walks there to escape.
KING_MG: PieceSquareTable = (
    (20, 30, 10, 0, 0, 10, 30, 20),
    (20, 20, 0, 0, 0, 0, 20, 20),
    (-10, -20, -20, -20, -20, -20, -20, -10),
    (-20, -30, -30, -40, -40, -30, -30, -20),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
)

# Endgame king: the exact opposite, a bowl pulling the king to the middle. This is the sign
# flip that makes the king walk out and support its pawns once the queens are gone, and it is
# also the first half of driving a bare enemy king to the edge: their king in a corner is
# -50 for them. The mop-up term below is the other half.
KING_EG: PieceSquareTable = (
    (-50, -30, -30, -30, -30, -30, -30, -50),
    (-30, -20, -10, -10, -10, -10, -20, -30),
    (-30, -10, 10, 20, 20, 10, -10, -30),
    (-30, -10, 20, 30, 30, 20, -10, -30),
    (-30, -10, 20, 30, 30, 20, -10, -30),
    (-30, -10, 10, 20, 20, 10, -10, -30),
    (-30, -20, -10, -10, -10, -10, -20, -30),
    (-50, -30, -30, -30, -30, -30, -30, -50),
)

MIDDLEGAME_TABLES: tuple[PieceSquareTable, ...] = (
    PAWN_MG,
    KNIGHT_MG,
    BISHOP_MG,
    ROOK_MG,
    QUEEN_MG,
    KING_MG,
)
ENDGAME_TABLES: tuple[PieceSquareTable, ...] = (
    PAWN_EG,
    KNIGHT_EG,
    BISHOP_EG,
    ROOK_EG,
    QUEEN_EG,
    KING_EG,
)

# Passed pawns by the rank they have reached, counted from their own side: index 1 is a pawn
# still at home and index 6 is one step from queening. Nothing else on the board grows this
# steeply, and in the endgame a passer on the seventh is most of a piece on its own.
PASSED_MG: tuple[int, ...] = (0, 5, 10, 20, 35, 60, 90, 0)
PASSED_EG: tuple[int, ...] = (0, 10, 20, 35, 60, 95, 140, 0)
# A pawn with no friend on either neighbouring file can never be defended by a pawn again.
ISOLATED_MG = 14
ISOLATED_EG = 18
# Two pawns on one file: one of them is not going anywhere, and they cover five files between
# them instead of six. It costs more in the endgame, where the extra file is a passed pawn.
DOUBLED_MG = 10
DOUBLED_EG = 22

# A rook sees the whole board down a file with no pawns on it, and half of one down a file
# with only enemy pawns. Worth less once there are few pawns left to block anything.
ROOK_OPEN_MG = 22
ROOK_OPEN_EG = 12
ROOK_SEMI_OPEN_MG = 10
ROOK_SEMI_OPEN_EG = 6
# Two bishops cover both square colours, which matters more as the board empties.
BISHOP_PAIR_MG = 25
BISHOP_PAIR_EG = 45

# Charged per missing pawn of the three the king would like in front of it. Middlegame only:
# in the endgame there is nothing left to attack with and the king table wants the king out.
# The mask is six squares, three files by two ranks, and the count is capped at three, so this
# is "how many of the three files are covered at all" rather than "how close the cover is":
# pawns on f3, g3 and h3 read as a full shield even though f2, g2 and h2 are empty. A king on
# its own last rank has an empty mask and is charged the full amount, which double-counts the
# middlegame king table's own dislike of that square; both push the same way, so it stands.
SHIELD_PENALTY = 14
SHIELD_MAX_COVER = 3

# Mop-up. When one side cannot lose material fast enough to matter, the only thing left to
# score is geometry: push their king to a corner and walk ours up next to it. Without this
# every move in KRvK ties on material and piece-square score and the game is drawn by the
# fifty move rule. The weights are bounded by MOP_UP_CMD * 6 + MOP_UP_CLOSE * 12, which is
# 120 centipawns: well under a minor piece, so geometry never outweighs a piece, though it
# can outweigh a pawn, which in a position with a bare king is the trade we want anyway.
MOP_UP_CMD = 10
MOP_UP_CLOSE = 5
# The same geometry, a third as loud, for a won position that has not been reduced to a bare
# king yet: enough to stop a +15 position wandering, not enough to distort a real endgame.
MOP_UP_LOOSE_CMD = 4
MOP_UP_LOOSE_CLOSE = 2
# Below this much material advantage there is nothing to mop up. The bound is inclusive so
# that a queen against a rook, or a rook against a pawn, which land exactly on it, are in.
MOP_UP_MIN_ADVANTAGE = 400
# Mop-up is only looked at when the weaker side has a king and at most two other men. The
# cheap version of that test, that neither side is bigger than this, is what keeps the whole
# term off the middlegame path; which side is actually the weaker one is decided after.
MOP_UP_MAX_WEAK_PIECES = 3
# The full weights need the weaker side down to a king and at most one minor.
MOP_UP_BARE_PIECES = 2

# With no pawns anywhere, an advantage smaller than a minor piece is usually not a win at all:
# rook against a bishop, or two rooks against a queen, are book draws with correct defence.
# Halving keeps the sign, so the search still prefers the ending, but stops it paying material
# to reach one and stops contempt reading it as a won game. Those two are the whole target;
# nothing here claims to know which rook endings are drawn.
DRAWISH_MARGIN = 200

# The side to move is stalemated far more often than it looks when it has almost nothing left,
# and quiescence never asks the rules for it. Same limit as `agent.py`: a king and at most two
# other men. `fastsearch` reads this; it lives here because it is a property of the evaluation
# being blind to stalemate, not of the search.
STALEMATE_PIECE_LIMIT = 3


# --------------------------------------------------------------------------------------
# Everything above is the source of truth. Everything below turns it into flat arrays a
# jitted function can index, and none of it introduces a number of its own.
# --------------------------------------------------------------------------------------

# Mailbox 21..98 to a 0..63 square, and its file and rank counted from White's home rank,
# which is the orientation the tables and every mask are written in. Off-board squares get
# -1 so a stray index is a loud failure rather than a plausible wrong answer.
FILE_OF = np.full(120, -1, dtype=np.int32)
RANK_OF = np.full(120, -1, dtype=np.int32)
for _mb in range(21, 99):
    _offset = _mb - 21
    if _offset % 10 <= 7:
        FILE_OF[_mb] = _offset % 10
        RANK_OF[_mb] = 7 - _offset // 10


def _tapered_tables() -> tuple[np.ndarray, np.ndarray]:
    """One signed number per (piece code, mailbox square), for each end of the taper.

    Material is folded into the table exactly as `_flat_table` does, and the sign is folded in
    too: a Black piece's entry is already negative, so the scan in `evaluate` is one lookup and
    one add per piece with no branch on colour. Black reads the vertically mirrored row, which
    is the mirroring `_flat_table` does with `table[::-1]`.
    """
    middlegame = np.zeros((13, 120), dtype=np.int32)
    endgame = np.zeros((13, 120), dtype=np.int32)
    for index in range(6):
        value = PIECE_VALUES[index]
        for mailbox in range(21, 99):
            file_index, rank_index = int(FILE_OF[mailbox]), int(RANK_OF[mailbox])
            if file_index < 0:
                continue
            middlegame[index + 1, mailbox] = value + MIDDLEGAME_TABLES[index][rank_index][
                file_index
            ]
            endgame[index + 1, mailbox] = value + ENDGAME_TABLES[index][rank_index][file_index]
            middlegame[index + 7, mailbox] = -(
                value + MIDDLEGAME_TABLES[index][7 - rank_index][file_index]
            )
            endgame[index + 7, mailbox] = -(
                value + ENDGAME_TABLES[index][7 - rank_index][file_index]
            )
    return middlegame, endgame


MG_TABLE, EG_TABLE = _tapered_tables()

# Rank sets, as the eight-bit-per-file masks the packed pawn summaries are compared against.
# `AHEAD_WHITE[r]` is every rank above r and `AHEAD_BLACK[r]` every rank below it, which is
# what `_front_spans` builds a bitboard of; combined with the file the mask is taken from,
# the pair reproduces the doubled test (own file) and the passed test (three files).
AHEAD_WHITE = np.zeros(8, dtype=np.int32)
AHEAD_BLACK = np.zeros(8, dtype=np.int32)
# The two ranks in front of a king, which is the vertical half of `_shields`; the horizontal
# half is the three files the shield loop reads.
SHIELD_WHITE = np.zeros(8, dtype=np.int32)
SHIELD_BLACK = np.zeros(8, dtype=np.int32)
for _rank in range(8):
    AHEAD_WHITE[_rank] = (0xFF << (_rank + 1)) & 0xFF
    AHEAD_BLACK[_rank] = (1 << _rank) - 1
    for _step in (1, 2):
        if _rank + _step <= 7:
            SHIELD_WHITE[_rank] |= 1 << (_rank + _step)
        if _rank - _step >= 0:
            SHIELD_BLACK[_rank] |= 1 << (_rank - _step)

# Population count of a one-file rank set, for the shield cover.
POPCOUNT8 = np.array([bin(_i).count("1") for _i in range(256)], dtype=np.int32)

# Manhattan distance to the nearest centre square, as `_centre_distances` computes it, but
# indexed by mailbox square.
CENTRE_DISTANCE = np.zeros(120, dtype=np.int32)
for _mb in range(21, 99):
    if FILE_OF[_mb] >= 0:
        CENTRE_DISTANCE[_mb] = abs(2 * int(FILE_OF[_mb]) - 7) // 2 + abs(
            2 * int(RANK_OF[_mb]) - 7
        ) // 2

PASSED_MG_BY_RANK = np.array(PASSED_MG, dtype=np.int32)
PASSED_EG_BY_RANK = np.array(PASSED_EG, dtype=np.int32)

_BOARD_T = nbt.int8[::1]
_ST_T = nbt.int64[::1]

PIECE_VALUE_BY_KIND = np.array(PIECE_VALUES, dtype=np.int32)
# Index of the lowest set bit of a one-file rank set, so the pawn loops walk pawns rather
# than the eight ranks each file might hold one on.
CTZ8 = np.array([0] + [(_i & -_i).bit_length() - 1 for _i in range(1, 256)], dtype=np.int32)


@njit(nbt.int64(_BOARD_T, _ST_T), cache=False)
def evaluate(board: np.ndarray, st: np.ndarray) -> int:
    """The tapered score in centipawns, from the point of view of the side to move.

    Equal to `agent.py`'s `evaluate` on every position, by construction and by test.
    """
    middlegame = 0
    endgame = 0
    # Rank sets per file, eight bits per file; rook counts per file, four bits per file.
    white_pawn_files = 0
    black_pawn_files = 0
    white_rook_files = 0
    black_rook_files = 0
    white_men = 0
    black_men = 0
    white_material = 0
    black_material = 0
    white_pawns = 0
    black_pawns = 0
    # The square of the last pawn seen; only read when it is the only pawn on the board.
    lone_pawn_square = -1
    white_bishops = 0
    black_bishops = 0
    # A side's pawns, rooks and queens together: zero is the "cannot make progress" test the
    # mop-up weights are chosen for.
    white_heavy = 0
    black_heavy = 0
    minors = 0
    rooks = 0
    queens = 0

    for square in range(21, 99):
        piece = board[square]
        if piece == 0 or piece == 13:
            continue
        middlegame += MG_TABLE[piece, square]
        endgame += EG_TABLE[piece, square]
        white = piece <= 6
        kind = piece if white else piece - 6
        value = PIECE_VALUE_BY_KIND[kind - 1]
        file_index = FILE_OF[square]
        rank_index = RANK_OF[square]
        if white:
            white_men += 1
            white_material += value
        else:
            black_men += 1
            black_material += value
        if kind == 1:
            lone_pawn_square = square
            if white:
                white_pawn_files |= 1 << (8 * file_index + rank_index)
                white_pawns += 1
                white_heavy += 1
            else:
                black_pawn_files |= 1 << (8 * file_index + rank_index)
                black_pawns += 1
                black_heavy += 1
        elif kind == 2:
            minors += 1
        elif kind == 3:
            minors += 1
            if white:
                white_bishops += 1
            else:
                black_bishops += 1
        elif kind == 4:
            rooks += 1
            if white:
                white_rook_files += 1 << (4 * file_index)
                white_heavy += 1
            else:
                black_rook_files += 1 << (4 * file_index)
                black_heavy += 1
        elif kind == 5:
            queens += 1
            if white:
                white_heavy += 1
            else:
                black_heavy += 1

    # Passed, isolated and doubled pawns, and the rook files, one pass over the eight files.
    for file_index in range(8):
        shift = 8 * file_index
        white_file = (white_pawn_files >> shift) & 255
        black_file = (black_pawn_files >> shift) & 255
        left = shift - 8
        right = shift + 8
        if white_file != 0:
            neighbours = 0
            enemy_span = black_file
            if file_index > 0:
                neighbours |= (white_pawn_files >> left) & 255
                enemy_span |= (black_pawn_files >> left) & 255
            if file_index < 7:
                neighbours |= (white_pawn_files >> right) & 255
                enemy_span |= (black_pawn_files >> right) & 255
            ranks = white_file
            while ranks != 0:
                rank_index = CTZ8[ranks]
                ranks &= ranks - 1
                ahead = AHEAD_WHITE[rank_index]
                # The rear pawn of a doubled pair is behind one of its own and can never queen
                # ahead of it, so it is not a passed pawn however empty the enemy files are.
                blocked = white_file & ahead
                if blocked == 0 and (enemy_span & ahead) == 0:
                    middlegame += PASSED_MG_BY_RANK[rank_index]
                    endgame += PASSED_EG_BY_RANK[rank_index]
                if neighbours == 0:
                    middlegame -= ISOLATED_MG
                    endgame -= ISOLATED_EG
                if blocked != 0:
                    middlegame -= DOUBLED_MG
                    endgame -= DOUBLED_EG
        if black_file != 0:
            neighbours = 0
            enemy_span = white_file
            if file_index > 0:
                neighbours |= (black_pawn_files >> left) & 255
                enemy_span |= (white_pawn_files >> left) & 255
            if file_index < 7:
                neighbours |= (black_pawn_files >> right) & 255
                enemy_span |= (white_pawn_files >> right) & 255
            ranks = black_file
            while ranks != 0:
                rank_index = CTZ8[ranks]
                ranks &= ranks - 1
                ahead = AHEAD_BLACK[rank_index]
                blocked = black_file & ahead
                if blocked == 0 and (enemy_span & ahead) == 0:
                    middlegame -= PASSED_MG_BY_RANK[7 - rank_index]
                    endgame -= PASSED_EG_BY_RANK[7 - rank_index]
                if neighbours == 0:
                    middlegame += ISOLATED_MG
                    endgame += ISOLATED_EG
                if blocked != 0:
                    middlegame += DOUBLED_MG
                    endgame += DOUBLED_EG
        white_rooks = (white_rook_files >> (4 * file_index)) & 15
        if white_rooks != 0:
            if white_file == 0 and black_file == 0:
                middlegame += white_rooks * ROOK_OPEN_MG
                endgame += white_rooks * ROOK_OPEN_EG
            elif white_file == 0:
                middlegame += white_rooks * ROOK_SEMI_OPEN_MG
                endgame += white_rooks * ROOK_SEMI_OPEN_EG
        black_rooks = (black_rook_files >> (4 * file_index)) & 15
        if black_rooks != 0:
            if white_file == 0 and black_file == 0:
                middlegame -= black_rooks * ROOK_OPEN_MG
                endgame -= black_rooks * ROOK_OPEN_EG
            elif black_file == 0:
                middlegame -= black_rooks * ROOK_SEMI_OPEN_MG
                endgame -= black_rooks * ROOK_SEMI_OPEN_EG

    if white_bishops >= 2:
        middlegame += BISHOP_PAIR_MG
        endgame += BISHOP_PAIR_EG
    if black_bishops >= 2:
        middlegame -= BISHOP_PAIR_MG
        endgame -= BISHOP_PAIR_EG

    white_king = st[5]
    black_king = st[6]
    white_king_file = FILE_OF[white_king]
    white_king_rank = RANK_OF[white_king]
    black_king_file = FILE_OF[black_king]
    black_king_rank = RANK_OF[black_king]

    # King shield, middlegame only, positive when White is the better covered of the two.
    mask = SHIELD_WHITE[white_king_rank]
    white_cover = 0
    for file_index in range(max(0, white_king_file - 1), min(7, white_king_file + 1) + 1):
        white_cover += POPCOUNT8[((white_pawn_files >> (8 * file_index)) & 255) & mask]
    if white_cover > SHIELD_MAX_COVER:
        white_cover = SHIELD_MAX_COVER
    mask = SHIELD_BLACK[black_king_rank]
    black_cover = 0
    for file_index in range(max(0, black_king_file - 1), min(7, black_king_file + 1) + 1):
        black_cover += POPCOUNT8[((black_pawn_files >> (8 * file_index)) & 255) & mask]
    if black_cover > SHIELD_MAX_COVER:
        black_cover = SHIELD_MAX_COVER
    middlegame += (white_cover - black_cover) * SHIELD_PENALTY

    # Mop-up, endgame only. The two man counts are the gate that keeps this off the
    # middlegame path; which side is the weaker one is decided after.
    if white_men <= MOP_UP_MAX_WEAK_PIECES or black_men <= MOP_UP_MAX_WEAK_PIECES:
        advantage = white_material - black_material
        sign = 0
        weak_king = white_king
        weak_count = white_men
        weak_heavy = white_heavy
        if advantage >= MOP_UP_MIN_ADVANTAGE:
            sign = 1
            weak_king = black_king
            weak_count = black_men
            weak_heavy = black_heavy
        elif advantage <= -MOP_UP_MIN_ADVANTAGE:
            sign = -1
        # The gate above only proved that one of the two sides is small. If the small one is
        # the side that is ahead, there is nothing to mop up: the other side still has an army.
        if sign != 0 and weak_count <= MOP_UP_MAX_WEAK_PIECES:
            bare = weak_count <= MOP_UP_BARE_PIECES and weak_heavy == 0
            if bare:
                centre_weight = MOP_UP_CMD
                close_weight = MOP_UP_CLOSE
            else:
                centre_weight = MOP_UP_LOOSE_CMD
                close_weight = MOP_UP_LOOSE_CLOSE
            separation = abs(white_king_file - black_king_file) + abs(
                white_king_rank - black_king_rank
            )
            endgame += sign * (
                centre_weight * CENTRE_DISTANCE[weak_king] + close_weight * (14 - separation)
            )

    phase = minors + PHASE_ROOK * rooks + PHASE_QUEEN * queens
    if phase > PHASE_MAX:
        phase = PHASE_MAX
    # Truncated toward zero rather than floored, so that mirroring the board negates the score
    # exactly instead of leaving White a centipawn ahead of Black in the same position.
    total = middlegame * phase + endgame * (PHASE_MAX - phase)
    score = total // PHASE_MAX if total >= 0 else -((-total) // PHASE_MAX)
    # King and rook pawn against a bare king is a draw when the defending king stands in front
    # of the pawn on its file or the one beside it: the corner square cannot be taken from it,
    # and the pawn ends in a stalemate or is captured. The tables call it a pawn up plus a
    # passer on the sixth, around +250, so the search would trade into it believing it wins,
    # and contempt then refuses the draw it is. `agent.py` has the same rule, word for word.
    if (
        white_men + black_men == 3
        and white_pawns + black_pawns == 1
        and minors == 0
        and rooks == 0
        and queens == 0
    ):
        pawn_file = FILE_OF[lone_pawn_square]
        if pawn_file == 0 or pawn_file == 7:
            pawn_rank = RANK_OF[lone_pawn_square]
            if white_pawns == 1:
                ahead = black_king_rank > pawn_rank
                defender_file = black_king_file
            else:
                ahead = white_king_rank < pawn_rank
                defender_file = white_king_file
            if ahead and abs(defender_file - pawn_file) <= 1:
                return 0
    if white_pawns == 0 and black_pawns == 0:
        # A single minor and two kings is a dead draw, and the tables would otherwise call it
        # a third of a piece. The search asks the rules for this, but quiescence never does.
        if rooks == 0 and queens == 0 and minors <= 1:
            return 0
        advantage = white_material - black_material
        if -DRAWISH_MARGIN <= advantage <= DRAWISH_MARGIN:
            score = score // 2 if score >= 0 else -((-score) // 2)
    return score if st[0] == 0 else -score


@njit(nbt.int64(_BOARD_T, _ST_T), cache=False)
def mop_up(board: np.ndarray, st: np.ndarray) -> int:
    """`evaluate`'s mop-up term on its own, tapered, from the side to move's point of view.

    `fastsearch.leaf` scores a bare endgame as the blend of the two evaluations plus a whole
    mop-up term, so it needs the term separately; `evaluate` folds it in with everything else
    and has to stay byte-identical for the hand policy, so this recounts rather than returning
    it from there. The gate, the weights and the taper are `evaluate`'s, line for line.

    It is zero unless `evaluate`'s mop-up condition fires: the weaker side down to a king and
    at most two other men, *and* the stronger side at least `MOP_UP_MIN_ADVANTAGE` ahead. So
    it is zero in KPvK, and zero in every position the hand evaluation calls a dead draw
    (KBvK and KNvK are 330 and 320 ahead, the rook-pawn draw 100), which is what makes adding
    it safe: it can neither invent a win in a drawn position nor disturb one.

    The taper truncates toward zero, as `evaluate`'s does. `evaluate` truncates the *sum* of
    its two tapered halves once, so this can differ from the term's share of that score by a
    centipawn. `tests/test_fasteval.py` freezes the value on the endings the term was written
    for and checks the bound and the gates everywhere else.
    """
    white_men = 0
    black_men = 0
    white_material = 0
    black_material = 0
    white_heavy = 0
    black_heavy = 0
    minors = 0
    rooks = 0
    queens = 0
    for square in range(21, 99):
        piece = board[square]
        if piece == 0 or piece == 13:
            continue
        white = piece <= 6
        kind = piece if white else piece - 6
        value = PIECE_VALUE_BY_KIND[kind - 1]
        if white:
            white_men += 1
            white_material += value
        else:
            black_men += 1
            black_material += value
        if kind == 1:
            if white:
                white_heavy += 1
            else:
                black_heavy += 1
        elif kind == 2 or kind == 3:
            minors += 1
        elif kind == 4:
            rooks += 1
            if white:
                white_heavy += 1
            else:
                black_heavy += 1
        elif kind == 5:
            queens += 1
            if white:
                white_heavy += 1
            else:
                black_heavy += 1

    if white_men > MOP_UP_MAX_WEAK_PIECES and black_men > MOP_UP_MAX_WEAK_PIECES:
        return 0
    white_king = st[5]
    black_king = st[6]
    advantage = white_material - black_material
    sign = 0
    weak_king = white_king
    weak_count = white_men
    weak_heavy = white_heavy
    if advantage >= MOP_UP_MIN_ADVANTAGE:
        sign = 1
        weak_king = black_king
        weak_count = black_men
        weak_heavy = black_heavy
    elif advantage <= -MOP_UP_MIN_ADVANTAGE:
        sign = -1
    if sign == 0 or weak_count > MOP_UP_MAX_WEAK_PIECES:
        return 0
    bare = weak_count <= MOP_UP_BARE_PIECES and weak_heavy == 0
    if bare:
        centre_weight = MOP_UP_CMD
        close_weight = MOP_UP_CLOSE
    else:
        centre_weight = MOP_UP_LOOSE_CMD
        close_weight = MOP_UP_LOOSE_CLOSE
    separation: int = abs(FILE_OF[white_king] - FILE_OF[black_king]) + abs(
        RANK_OF[white_king] - RANK_OF[black_king]
    )
    endgame: int = 0
    endgame += sign * (
        centre_weight * CENTRE_DISTANCE[weak_king] + close_weight * (14 - separation)
    )

    phase = minors + PHASE_ROOK * rooks + PHASE_QUEEN * queens
    if phase > PHASE_MAX:
        phase = PHASE_MAX
    total = endgame * (PHASE_MAX - phase)
    score = total // PHASE_MAX if total >= 0 else -((-total) // PHASE_MAX)
    return score if st[0] == 0 else -score


def warm() -> None:
    """Run the jitted evaluation once so nothing compiles on the clock."""
    from fastboard import START_FEN, from_fen

    board, st, _ = from_fen(START_FEN)
    evaluate(board, st)
    mop_up(board, st)


warm()

COMPILE_SECONDS = time.perf_counter() - _STARTED
