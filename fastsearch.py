"""The v2.4 engine on the numba board: compiled evaluation, and the search that calls it.

`fastboard.py` is the substrate (board arrays, move generation, make/unmake, Zobrist keys).
This module is what runs on it: the tapered evaluation from `agent.py`, then alpha-beta with
quiescence, a transposition table, killers and history, all `@njit` with eager signatures so
everything compiles at import inside the platform's 90 second budget and nothing compiles on
the clock. `agent.py` owns the time management, the fallback and the game memory; it calls
in here once per iteration of its deepening loop.

The evaluation's numbers live in `agent.py`, where v2.4 wrote them and where a judge reads
them, and arrive here packed into two arrays by `pack_tables` and `pack_weights`. That keeps
one source of truth for the Python evaluation and the compiled one, and it means a broken
compile of this module leaves the Python engine untouched: `agent.py` imports this inside a
`try` and plays v2.4 if the import fails.

Conventions shared with `fastboard`: `board` is the 10x12 mailbox (a8 is 21, h1 is 98), piece
codes are 1..6 white P N B R Q K and 7..12 black, `st[0]` is the side to move, `st[5]` and
`st[6]` the king squares, `st[7]` the Zobrist key. Scores are centipawns from the side to
move, as in `agent.py`.
"""

import numpy as np
from numba import njit
from numba import types as nbt

import fastboard as fb

EMPTY = fb.EMPTY
OFF = fb.OFF

# Mailbox square to file (a is 0) and rank (the first rank is 0); -1 off the board.
FILE_OF = np.full(120, -1, dtype=np.int64)
RANK_OF = np.full(120, -1, dtype=np.int64)
for _sq in range(21, 99):
    if 1 <= _sq % 10 <= 8:
        FILE_OF[_sq] = (_sq - 21) % 10
        RANK_OF[_sq] = 7 - (_sq - 21) // 10

# Indices into the weights vector `pack_weights` builds. The first six are the piece values
# in fastboard's kind order (pawn 1 lands at index 0), so material is `w[kind - 1]`.
W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING = 0, 1, 2, 3, 4, 5
W_ISOLATED_MG, W_ISOLATED_EG = 6, 7
W_DOUBLED_MG, W_DOUBLED_EG = 8, 9
W_ROOK_OPEN_MG, W_ROOK_OPEN_EG = 10, 11
W_ROOK_SEMI_OPEN_MG, W_ROOK_SEMI_OPEN_EG = 12, 13
W_BISHOP_PAIR_MG, W_BISHOP_PAIR_EG = 14, 15
W_SHIELD_PENALTY, W_SHIELD_MAX_COVER = 16, 17
W_MOP_UP_CMD, W_MOP_UP_CLOSE = 18, 19
W_MOP_UP_LOOSE_CMD, W_MOP_UP_LOOSE_CLOSE = 20, 21
W_MOP_UP_MIN_ADVANTAGE, W_MOP_UP_MAX_WEAK_PIECES, W_MOP_UP_BARE_PIECES = 22, 23, 24
W_DRAWISH_MARGIN = 25
W_PHASE_ROOK, W_PHASE_QUEEN, W_PHASE_MAX = 26, 27, 28
# Eight entries each, indexed by the rank a passed pawn has reached from its own side.
W_PASSED_MG = 29
W_PASSED_EG = 37
W_COUNT = 45

_BOARD_T = nbt.int8[::1]
_ST_T = nbt.int64[::1]
_UNDO_T = nbt.int64[:, ::1]
_PST_T = nbt.int64[:, :, ::1]
_W_T = nbt.int64[::1]

PieceSquareTable = tuple[tuple[int, ...], ...]


def pack_tables(
    middlegame: tuple[PieceSquareTable, ...],
    endgame: tuple[PieceSquareTable, ...],
    piece_values: dict[int, int],
) -> np.ndarray:
    """Fold material into the piece-square tables, per piece code and mailbox square.

    `pst[phase, piece, square]` is what a piece of that code standing there is worth to White,
    material included: positive for White's pieces and negative for Black's, whose rows are
    mirrored exactly as `agent._flat_table` mirrors them. The evaluation's board pass is then
    one lookup and one add per man, with no sign to think about.
    """
    pst = np.zeros((2, 13, 120), dtype=np.int64)
    for kind in range(1, 7):
        value = piece_values[kind]
        for sq in range(21, 99):
            file_index, rank_index = int(FILE_OF[sq]), int(RANK_OF[sq])
            if rank_index < 0:
                continue
            for phase, tables in ((0, middlegame), (1, endgame)):
                table = tables[kind - 1]
                pst[phase, kind, sq] = value + table[rank_index][file_index]
                pst[phase, kind + 6, sq] = -(value + table[7 - rank_index][file_index])
    return pst


def pack_weights(
    *,
    piece_values: dict[int, int],
    passed_mg: tuple[int, ...],
    passed_eg: tuple[int, ...],
    isolated_mg: int,
    isolated_eg: int,
    doubled_mg: int,
    doubled_eg: int,
    rook_open_mg: int,
    rook_open_eg: int,
    rook_semi_open_mg: int,
    rook_semi_open_eg: int,
    bishop_pair_mg: int,
    bishop_pair_eg: int,
    shield_penalty: int,
    shield_max_cover: int,
    mop_up_cmd: int,
    mop_up_close: int,
    mop_up_loose_cmd: int,
    mop_up_loose_close: int,
    mop_up_min_advantage: int,
    mop_up_max_weak_pieces: int,
    mop_up_bare_pieces: int,
    drawish_margin: int,
    phase_rook: int,
    phase_queen: int,
    phase_max: int,
) -> np.ndarray:
    """The evaluation's scalar weights as one vector, in the order the `W_` indices name."""
    w = np.zeros(W_COUNT, dtype=np.int64)
    for kind in range(1, 7):
        w[kind - 1] = piece_values[kind]
    w[W_ISOLATED_MG], w[W_ISOLATED_EG] = isolated_mg, isolated_eg
    w[W_DOUBLED_MG], w[W_DOUBLED_EG] = doubled_mg, doubled_eg
    w[W_ROOK_OPEN_MG], w[W_ROOK_OPEN_EG] = rook_open_mg, rook_open_eg
    w[W_ROOK_SEMI_OPEN_MG], w[W_ROOK_SEMI_OPEN_EG] = rook_semi_open_mg, rook_semi_open_eg
    w[W_BISHOP_PAIR_MG], w[W_BISHOP_PAIR_EG] = bishop_pair_mg, bishop_pair_eg
    w[W_SHIELD_PENALTY], w[W_SHIELD_MAX_COVER] = shield_penalty, shield_max_cover
    w[W_MOP_UP_CMD], w[W_MOP_UP_CLOSE] = mop_up_cmd, mop_up_close
    w[W_MOP_UP_LOOSE_CMD], w[W_MOP_UP_LOOSE_CLOSE] = mop_up_loose_cmd, mop_up_loose_close
    w[W_MOP_UP_MIN_ADVANTAGE] = mop_up_min_advantage
    w[W_MOP_UP_MAX_WEAK_PIECES] = mop_up_max_weak_pieces
    w[W_MOP_UP_BARE_PIECES] = mop_up_bare_pieces
    w[W_DRAWISH_MARGIN] = drawish_margin
    w[W_PHASE_ROOK], w[W_PHASE_QUEEN], w[W_PHASE_MAX] = phase_rook, phase_queen, phase_max
    for rank_index in range(8):
        w[W_PASSED_MG + rank_index] = passed_mg[rank_index]
        w[W_PASSED_EG + rank_index] = passed_eg[rank_index]
    return w


@njit(nbt.int64(nbt.int64, nbt.int64), cache=False)
def _file_bits(files: int, file_index: int) -> int:
    """The eight rank bits of one file out of a packed pawn mask.

    A side's pawns are summarised as one int64: bit `8 * file + rank` is set for each pawn.
    Pawns never stand on the first or last rank, so bit 63 is never set and the shift is safe.
    """
    return (files >> (8 * file_index)) & 0xFF


@njit(nbt.int64(_BOARD_T, _ST_T, _PST_T, _W_T), cache=False)
def evaluate(board: np.ndarray, st: np.ndarray, pst: np.ndarray, w: np.ndarray) -> int:
    """`agent.evaluate`, term for term, on the mailbox. Returns the same integer.

    Two passes over the 64 squares and no allocation. The first sums the tapered tables and
    gathers the counts everything else needs, including each side's pawns packed by file so
    that the structure terms are shifts and masks. The second scores the pawns and the rooks
    against those masks. Then the king shield, mop-up, the phase blend and the no-pawn rules,
    in the order `agent.py` applies them.
    """
    middlegame = 0
    endgame = 0
    white_files = 0
    black_files = 0
    white_count = 0
    black_count = 0
    white_material = 0
    black_material = 0
    white_bishops = 0
    black_bishops = 0
    # Pawns, rooks and queens: the men that can still win material or make progress. A weak
    # side with none of them is "helpless" for the mop-up term.
    white_heavy = 0
    black_heavy = 0
    pawns = 0
    minors = 0
    rooks = 0
    queens = 0
    for sq in range(21, 99):
        piece = board[sq]
        if piece in (EMPTY, OFF):
            continue
        middlegame += pst[0, piece, sq]
        endgame += pst[1, piece, sq]
        white = piece <= 6
        kind = piece if white else piece - 6
        value = w[kind - 1]
        if white:
            white_count += 1
            white_material += value
        else:
            black_count += 1
            black_material += value
        if kind == 1:
            pawns += 1
            bit = 1 << (8 * FILE_OF[sq] + RANK_OF[sq])
            if white:
                white_files |= bit
                white_heavy += 1
            else:
                black_files |= bit
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
                white_heavy += 1
            else:
                black_heavy += 1
        elif kind == 5:
            queens += 1
            if white:
                white_heavy += 1
            else:
                black_heavy += 1

    # Pawn structure and rooks on open files, against the packed pawn masks. "Ahead" of a
    # white pawn on rank r is ranks r+1..7, so `bits >> (r + 1)`; ahead of a black pawn it is
    # ranks 0..r-1, so `bits & ((1 << r) - 1)`. A pawn with a friend ahead on its file is the
    # rear pawn of a doubled pair and is never passed, whatever the enemy files hold.
    for sq in range(21, 99):
        piece = board[sq]
        if piece == 1:
            file_index = FILE_OF[sq]
            rank_index = RANK_OF[sq]
            blocked = (_file_bits(white_files, file_index) >> (rank_index + 1)) != 0
            if not blocked:
                passed = True
                for near in range(file_index - 1, file_index + 2):
                    if 0 <= near <= 7 and (_file_bits(black_files, near) >> (rank_index + 1)) != 0:
                        passed = False
                        break
                if passed:
                    middlegame += w[W_PASSED_MG + rank_index]
                    endgame += w[W_PASSED_EG + rank_index]
            isolated = True
            if file_index > 0 and _file_bits(white_files, file_index - 1) != 0:
                isolated = False
            if file_index < 7 and _file_bits(white_files, file_index + 1) != 0:
                isolated = False
            if isolated:
                middlegame -= w[W_ISOLATED_MG]
                endgame -= w[W_ISOLATED_EG]
            if blocked:
                middlegame -= w[W_DOUBLED_MG]
                endgame -= w[W_DOUBLED_EG]
        elif piece == 7:
            file_index = FILE_OF[sq]
            rank_index = RANK_OF[sq]
            behind = (1 << rank_index) - 1
            blocked = (_file_bits(black_files, file_index) & behind) != 0
            if not blocked:
                passed = True
                for near in range(file_index - 1, file_index + 2):
                    if 0 <= near <= 7 and (_file_bits(white_files, near) & behind) != 0:
                        passed = False
                        break
                if passed:
                    middlegame -= w[W_PASSED_MG + 7 - rank_index]
                    endgame -= w[W_PASSED_EG + 7 - rank_index]
            isolated = True
            if file_index > 0 and _file_bits(black_files, file_index - 1) != 0:
                isolated = False
            if file_index < 7 and _file_bits(black_files, file_index + 1) != 0:
                isolated = False
            if isolated:
                middlegame += w[W_ISOLATED_MG]
                endgame += w[W_ISOLATED_EG]
            if blocked:
                middlegame += w[W_DOUBLED_MG]
                endgame += w[W_DOUBLED_EG]
        elif piece == 4 or piece == 10:
            file_index = FILE_OF[sq]
            sign = 1 if piece == 4 else -1
            own = white_files if piece == 4 else black_files
            if _file_bits(white_files | black_files, file_index) == 0:
                middlegame += sign * w[W_ROOK_OPEN_MG]
                endgame += sign * w[W_ROOK_OPEN_EG]
            elif _file_bits(own, file_index) == 0:
                middlegame += sign * w[W_ROOK_SEMI_OPEN_MG]
                endgame += sign * w[W_ROOK_SEMI_OPEN_EG]

    if white_bishops >= 2:
        middlegame += w[W_BISHOP_PAIR_MG]
        endgame += w[W_BISHOP_PAIR_EG]
    if black_bishops >= 2:
        middlegame -= w[W_BISHOP_PAIR_MG]
        endgame -= w[W_BISHOP_PAIR_EG]

    # King shield: own pawns on the three files around the king, two ranks in front of it,
    # counted up to a cap. Middlegame only.
    white_king = st[5]
    black_king = st[6]
    white_cover = 0
    king_file = FILE_OF[white_king]
    king_rank = RANK_OF[white_king]
    for near in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        for ahead in range(1, 3):
            rank_index = king_rank + ahead
            if rank_index <= 7 and board[21 + (7 - rank_index) * 10 + near] == 1:
                white_cover += 1
    black_cover = 0
    king_file = FILE_OF[black_king]
    king_rank = RANK_OF[black_king]
    for near in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        for ahead in range(1, 3):
            rank_index = king_rank - ahead
            if rank_index >= 0 and board[21 + (7 - rank_index) * 10 + near] == 7:
                black_cover += 1
    if white_cover > w[W_SHIELD_MAX_COVER]:
        white_cover = w[W_SHIELD_MAX_COVER]
    if black_cover > w[W_SHIELD_MAX_COVER]:
        black_cover = w[W_SHIELD_MAX_COVER]
    middlegame += (white_cover - black_cover) * w[W_SHIELD_PENALTY]

    # Mop-up: geometry for a won ending, exactly the gates `agent._mop_up` applies.
    max_weak = w[W_MOP_UP_MAX_WEAK_PIECES]
    if white_count <= max_weak or black_count <= max_weak:
        advantage = white_material - black_material
        sign = 0
        weak_king = black_king
        weak_count = black_count
        weak_heavy = black_heavy
        if advantage >= w[W_MOP_UP_MIN_ADVANTAGE]:
            sign = 1
        elif advantage <= -w[W_MOP_UP_MIN_ADVANTAGE]:
            sign = -1
            weak_king = white_king
            weak_count = white_count
            weak_heavy = white_heavy
        if sign != 0 and weak_count <= max_weak:
            bare = weak_count <= w[W_MOP_UP_BARE_PIECES] and weak_heavy == 0
            cmd = w[W_MOP_UP_CMD] if bare else w[W_MOP_UP_LOOSE_CMD]
            close = w[W_MOP_UP_CLOSE] if bare else w[W_MOP_UP_LOOSE_CLOSE]
            weak_file = FILE_OF[weak_king]
            weak_rank = RANK_OF[weak_king]
            centre = abs(2 * weak_file - 7) // 2 + abs(2 * weak_rank - 7) // 2
            separation = abs(FILE_OF[white_king] - FILE_OF[black_king]) + abs(
                RANK_OF[white_king] - RANK_OF[black_king]
            )
            endgame += sign * (cmd * centre + close * (14 - separation))

    phase = minors + w[W_PHASE_ROOK] * rooks + w[W_PHASE_QUEEN] * queens
    phase_max = w[W_PHASE_MAX]
    if phase > phase_max:
        phase = phase_max
    # Truncated toward zero, as agent.py does, so mirroring a position negates its score.
    total = middlegame * phase + endgame * (phase_max - phase)
    score = total // phase_max if total >= 0 else -((-total) // phase_max)
    if pawns == 0:
        if rooks + queens == 0 and minors <= 1:
            return 0
        advantage = white_material - black_material
        drawish = w[W_DRAWISH_MARGIN]
        if -drawish <= advantage <= drawish:
            score = score // 2 if score >= 0 else -((-score) // 2)
    return int(score) if st[0] == 0 else -int(score)
