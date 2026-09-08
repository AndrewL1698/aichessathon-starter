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

from collections.abc import Callable

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


# ----------------------------------------------------------------------------------------
# The search: agent.py's negamax, quiescence, table, killers and history, compiled.
# ----------------------------------------------------------------------------------------

# The same constants agent.py uses; tests/test_fastsearch.py asserts they agree.
MATE = 1_000_000
INFINITY = MATE + 1
MATE_FOUND = MATE - 1_000
QUIESCENCE_MAX_PLY = 8
STALEMATE_PIECE_LIMIT = 3
FIFTY_MOVE_PLIES = 100
EXACT, LOWER, UPPER = 0, 1, 2
TABLE_BONUS = 4_000_000
CAPTURE_BONUS = 1_000_000
PROMOTION_BONUS = 500_000
KILLER_BONUS = 400_000
HISTORY_CAP = KILLER_BONUS - 3

MAX_PLY = fb.MAX_PLY
# The transposition table is one int64 array of TT_SIZE rows and five columns, indexed by
# the low bits of the Zobrist key, replaced on every store. Two million rows of forty bytes
# is 84 MB, allocated once at import and kept for the game.
TT_BITS = 21
TT_SIZE = 1 << TT_BITS
TT_KEY, TT_DEPTH, TT_BOUND, TT_SCORE, TT_MOVE = 0, 1, 2, 3, 4
TT_COLUMNS = 5
# The game's positions, for repetition. A game is capped at 600 plies.
HIST_SIZE = 1024
# The control vector: counters the search keeps and switches the caller sets.
C_NODES = 0  # nodes searched this move
C_MAX_NODES = 1  # stop when C_NODES reaches this; the caller converts time into nodes
C_ABORT = 2  # set by the search when it stopped early; its result is then meaningless
C_CUTOFFS = 3
C_DRAWS = 4  # repetition and fifty-move draws returned; see negamax
C_CONTEMPT = 5  # what a draw is worth to the side to move at the root
C_N_HIST = 6  # how many of `hist` are filled; hist[C_N_HIST - 1] must be the root
C_USE_TABLE = 7  # 0 disables the transposition table (for the parity test)
C_USE_KILLERS = 8  # 0 disables killers and history (for the parity test)
C_COUNT = 9
UNLIMITED = 1 << 62

_ROW_T = nbt.int32[::1]
_BUFS_T = nbt.int32[:, ::1]
_RANKS_T = nbt.int64[:, ::1]
_VEC_T = nbt.int64[::1]
_KILLERS_T = nbt.int64[:, ::1]
_HISTORY_T = nbt.int64[:, :, ::1]
_TT_T = nbt.int64[:, ::1]


@njit(nbt.int64(_BOARD_T, nbt.int64), cache=False)
def men(board: np.ndarray, side: int) -> int:
    """How many men `side` has on the board, king included."""
    low = 1 + 6 * side
    high = low + 5
    count = 0
    for sq in range(21, 99):
        piece = board[sq]
        if low <= piece <= high:
            count += 1
    return count


@njit(nbt.boolean(_BOARD_T), cache=False)
def insufficient_material(board: np.ndarray) -> bool:
    """python-chess's `is_insufficient_material`, rule for rule.

    Any pawn, rook or queen is sufficient, and the scan leaves on the first one it meets, so
    a middlegame position costs a few loads. With none: a side with a knight is insufficient
    only when it has nothing else and the other side is a bare king; a side with bishops
    only when every bishop on the board, either colour, stands on one square colour and
    there are no knights; a bare king always.
    """
    white_men = 0
    black_men = 0
    white_knights = 0
    black_knights = 0
    white_bishops = 0
    black_bishops = 0
    dark = False
    light = False
    for sq in range(21, 99):
        piece = board[sq]
        if piece in (EMPTY, OFF):
            continue
        white = piece <= 6
        kind = piece if white else piece - 6
        if kind in (1, 4, 5):
            return False
        if white:
            white_men += 1
        else:
            black_men += 1
        if kind == 2:
            if white:
                white_knights += 1
            else:
                black_knights += 1
        elif kind == 3:
            if white:
                white_bishops += 1
            else:
                black_bishops += 1
            if (FILE_OF[sq] + RANK_OF[sq]) % 2 == 0:
                dark = True
            else:
                light = True
    knights = white_knights + black_knights
    if white_knights > 0:
        if white_men > 2 or black_men != 1:
            return False
    elif white_bishops > 0 and ((dark and light) or knights > 0):
        return False
    if black_knights > 0:
        if black_men > 2 or white_men != 1:
            return False
    elif black_bishops > 0 and ((dark and light) or knights > 0):
        return False
    return True


@njit(nbt.int64(nbt.int64, nbt.int64), cache=False)
def draw_score(contempt: int, ply: int) -> int:
    """A draw from the mover at `ply`; contempt is from the root's side, so odd plies flip."""
    return contempt if ply % 2 == 0 else -contempt


@njit(nbt.int64(nbt.int64, nbt.int64), cache=False)
def to_table(score: int, ply: int) -> int:
    """A mate score counted from this node rather than from the root, for storing."""
    if score > MATE_FOUND:
        return score + ply
    if score < -MATE_FOUND:
        return score - ply
    return score


@njit(nbt.int64(nbt.int64, nbt.int64), cache=False)
def from_table(score: int, ply: int) -> int:
    """Undo to_table for a node at this ply."""
    if score > MATE_FOUND:
        return score - ply
    if score < -MATE_FOUND:
        return score + ply
    return score


@njit(nbt.boolean(_BOARD_T, nbt.int32), cache=False)
def is_noisy(board: np.ndarray, move: int) -> bool:
    """A capture or a promotion, which order themselves and are never killers."""
    return board[(move >> 7) & 127] != EMPTY or (move & fb.FLAG_EP) != 0 or ((move >> 14) & 7) != 0


@njit(nbt.int64(_BOARD_T, nbt.int32, _W_T), cache=False)
def move_score(board: np.ndarray, move: int, w: np.ndarray) -> int:
    """`agent._move_score`: MVV-LVA for captures, plus a promotion bonus; zero for quiet moves."""
    frm = move & 127
    to = (move >> 7) & 127
    promo = (move >> 14) & 7
    score = 0
    captured = board[to]
    if (move & fb.FLAG_EP) != 0 or captured != EMPTY:
        victim = w[W_PAWN] if (move & fb.FLAG_EP) != 0 else w[(captured - 1) % 6]
        attacker = w[(board[frm] - 1) % 6]
        score += CAPTURE_BONUS + victim * 100 - attacker
    if promo != 0:
        score += PROMOTION_BONUS + w[promo - 1]
    return score


@njit(
    nbt.void(
        _BOARD_T,
        nbt.int64,
        _ROW_T,
        nbt.int64,
        nbt.int64,
        _VEC_T,
        _HISTORY_T,
        _W_T,
        _VEC_T,
        nbt.boolean,
    ),
    cache=False,
)
def rank_moves(
    board: np.ndarray,
    side: int,
    row: np.ndarray,
    count: int,
    table_move: int,
    killers: np.ndarray,
    history: np.ndarray,
    w: np.ndarray,
    ranks: np.ndarray,
    use_killers: bool,
) -> None:
    """`agent._order_fully`'s key for each move: table move, captures, promotions, killers,
    then quiet moves by history. The moves are picked highest first by `pick`."""
    for i in range(count):
        move = row[i]
        if move == table_move:
            ranks[i] = TABLE_BONUS
            continue
        rank = move_score(board, move, w)
        if rank == 0 and use_killers:
            if move == killers[0]:
                rank = KILLER_BONUS
            elif move == killers[1]:
                rank = KILLER_BONUS - 1
            else:
                rank = min(history[side, move & 127, (move >> 7) & 127], HISTORY_CAP)
        ranks[i] = rank


@njit(nbt.void(_ROW_T, _VEC_T, nbt.int64, nbt.int64), cache=False)
def pick(row: np.ndarray, ranks: np.ndarray, start: int, count: int) -> None:
    """Swap the highest-ranked move of `row[start:count]` into `row[start]`.

    Picking one move at a time rather than sorting the list costs nothing at a node that cuts
    off after its first move, which is most of them. Ties keep generation order.
    """
    best = start
    for i in range(start + 1, count):
        if ranks[i] > ranks[best]:
            best = i
    if best != start:
        move = row[start]
        row[start] = row[best]
        row[best] = move
        rank = ranks[start]
        ranks[start] = ranks[best]
        ranks[best] = rank


@njit(nbt.int64(_BOARD_T, _ST_T, _UNDO_T, _ROW_T), cache=False)
def gen_noisy(board: np.ndarray, st: np.ndarray, undo: np.ndarray, row: np.ndarray) -> int:
    """The legal captures and queen promotions: what quiescence searches when not in check.

    Pseudo-legal first, then the noisy ones are kept and only those are tested for legality,
    so a quiet leaf pays for a handful of make/unmake pairs rather than thirty.
    """
    count = fb.gen_moves(board, st, row)
    side = st[0]
    other = 1 - side
    kept = 0
    for i in range(count):
        move = row[i]
        if (
            board[(move >> 7) & 127] == EMPTY
            and (move & fb.FLAG_EP) == 0
            and ((move >> 14) & 7) != 5
        ):
            continue
        fb.make_move(board, st, undo, move)
        legal = not fb.is_square_attacked(board, st[5 + side], other)
        fb.unmake_move(board, st, undo, move)
        if legal:
            row[kept] = move
            kept += 1
    return kept


@njit(
    nbt.int64(
        _BOARD_T,
        _ST_T,
        _UNDO_T,
        _PST_T,
        _W_T,
        _BUFS_T,
        _RANKS_T,
        _VEC_T,
        nbt.int64,
        nbt.int64,
        nbt.int64,
        nbt.int64,
    ),
    cache=False,
)
def quiescence(
    board: np.ndarray,
    st: np.ndarray,
    undo: np.ndarray,
    pst: np.ndarray,
    w: np.ndarray,
    moves: np.ndarray,
    ranks: np.ndarray,
    ctl: np.ndarray,
    alpha: int,
    beta: int,
    ply: int,
    remaining: int,
) -> int:
    """`agent._quiescence`: captures and queen promotions until the position is quiet.

    In check every evasion is searched and there is no standing pat. Otherwise the side to
    move may stand on the static evaluation, and a side down to three men that has no legal
    move is stalemated rather than evaluated. Capped at `remaining` plies.
    """
    ctl[C_NODES] += 1
    if ctl[C_NODES] >= ctl[C_MAX_NODES]:
        ctl[C_ABORT] = 1
        return 0
    if ply >= MAX_PLY - 1:
        return evaluate(board, st, pst, w)
    row = moves[ply]
    if fb.in_check(board, st):
        count = fb.gen_legal(board, st, undo, row)
        if count == 0:
            return -MATE + ply
        if remaining == 0:
            return evaluate(board, st, pst, w)
        best = -INFINITY
    else:
        if men(board, st[0]) <= STALEMATE_PIECE_LIMIT and fb.gen_legal(board, st, undo, row) == 0:
            return draw_score(ctl[C_CONTEMPT], ply)
        best = evaluate(board, st, pst, w)
        if best >= beta or remaining == 0:
            return best
        if best > alpha:
            alpha = best
        count = gen_noisy(board, st, undo, row)
    rank_row = ranks[ply]
    for i in range(count):
        rank_row[i] = move_score(board, row[i], w)
    for i in range(count):
        pick(row, rank_row, i, count)
        move = row[i]
        fb.make_move(board, st, undo, move)
        score = -quiescence(
            board, st, undo, pst, w, moves, ranks, ctl, -beta, -alpha, ply + 1, remaining - 1
        )
        fb.unmake_move(board, st, undo, move)
        if ctl[C_ABORT] != 0:
            return 0
        if score > best:
            best = score
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break
    return best


@njit(
    nbt.int64(
        _BOARD_T,
        _ST_T,
        _UNDO_T,
        _PST_T,
        _W_T,
        _BUFS_T,
        _RANKS_T,
        _KILLERS_T,
        _HISTORY_T,
        _TT_T,
        _VEC_T,
        _VEC_T,
        _VEC_T,
        nbt.int64,
        nbt.int64,
        nbt.int64,
        nbt.int64,
    ),
    cache=False,
)
def negamax(
    board: np.ndarray,
    st: np.ndarray,
    undo: np.ndarray,
    pst: np.ndarray,
    w: np.ndarray,
    moves: np.ndarray,
    ranks: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    tt: np.ndarray,
    path: np.ndarray,
    hist: np.ndarray,
    ctl: np.ndarray,
    depth: int,
    ply: int,
    alpha: int,
    beta: int,
) -> int:
    """`agent._negamax`: fail-soft alpha-beta, step for step in the same order.

    Repetition and the fifty move rule first, then insufficient material, then quiescence at
    depth zero, the table probe, move generation, the ordered move loop, and the table store.
    Buffers are per ply (`moves[ply]`, `ranks[ply]`, `killers[ply]`), so a node never touches
    its children's. On abort every frame still unmakes its move, so the board the caller
    handed in is the board it gets back.
    """
    ctl[C_NODES] += 1
    if ctl[C_NODES] >= ctl[C_MAX_NODES]:
        ctl[C_ABORT] = 1
        return 0
    if ply >= MAX_PLY - 1:
        return evaluate(board, st, pst, w)
    key = st[7]
    contempt = ctl[C_CONTEMPT]

    # A position the game has stood in, or one already on this line, is a draw. Only the
    # last `clock` positions can match: a capture or a pawn move changes the board for good,
    # and the halfmove clock counts the plies since the last one. The ancestors on the path
    # come first, nearest first, then the game's positions, latest first; hist's last entry
    # is the root, which path[0] duplicates and is skipped.
    clock = st[3]
    if clock > 0:
        k = 1
        i = ply - 1
        while k <= clock and i >= 1:
            if path[i] == key:
                ctl[C_DRAWS] += 1
                return draw_score(contempt, ply)
            i -= 1
            k += 1
        i = ctl[C_N_HIST] - 1
        while k <= clock and i >= 0:
            if hist[i] == key:
                ctl[C_DRAWS] += 1
                return draw_score(contempt, ply)
            i -= 1
            k += 1
    row = moves[ply]
    # The fifty move rule does not rescue a side that is being mated.
    if clock >= FIFTY_MOVE_PLIES and (
        not fb.in_check(board, st) or fb.gen_legal(board, st, undo, row) > 0
    ):
        ctl[C_DRAWS] += 1
        return draw_score(contempt, ply)
    if insufficient_material(board):
        return draw_score(contempt, ply)
    if depth <= 0:
        return quiescence(
            board, st, undo, pst, w, moves, ranks, ctl, alpha, beta, ply, QUIESCENCE_MAX_PLY
        )

    table_move = 0
    slot = key & (TT_SIZE - 1)
    if ctl[C_USE_TABLE] != 0 and tt[slot, TT_KEY] == key:
        table_move = tt[slot, TT_MOVE]
        if tt[slot, TT_DEPTH] >= depth:
            score = from_table(tt[slot, TT_SCORE], ply)
            bound = tt[slot, TT_BOUND]
            if (
                bound == EXACT
                or (bound == LOWER and score >= beta)
                or (bound == UPPER and score <= alpha)
            ):
                return score

    count = fb.gen_legal(board, st, undo, row)
    if count == 0:
        return -MATE + ply if fb.in_check(board, st) else draw_score(contempt, ply)

    draws_before = ctl[C_DRAWS]
    window_alpha = alpha
    best = -INFINITY
    best_move = row[0]
    path[ply] = key
    side = st[0]
    use_killers = ctl[C_USE_KILLERS] != 0
    rank_row = ranks[ply]
    rank_moves(board, side, row, count, table_move, killers[ply], history, w, rank_row, use_killers)
    for i in range(count):
        pick(row, rank_row, i, count)
        move = row[i]
        noisy = is_noisy(board, move)
        fb.make_move(board, st, undo, move)
        score = -negamax(
            board,
            st,
            undo,
            pst,
            w,
            moves,
            ranks,
            killers,
            history,
            tt,
            path,
            hist,
            ctl,
            depth - 1,
            ply + 1,
            -beta,
            -alpha,
        )
        fb.unmake_move(board, st, undo, move)
        if ctl[C_ABORT] != 0:
            return 0
        if score > best:
            best = score
            best_move = move
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    ctl[C_CUTOFFS] += 1
                    if use_killers and not noisy:
                        if move != killers[ply, 0]:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = move
                        history[side, move & 127, (move >> 7) & 127] += depth * depth
                    break

    # A score that came out of a repetition or the fifty move rule belongs to the line, not
    # to the position, so it is not stored; see agent._negamax for the rest of the reasoning.
    if ctl[C_USE_TABLE] != 0 and ctl[C_DRAWS] == draws_before:
        bound = UPPER if best <= window_alpha else LOWER if best >= beta else EXACT
        tt[slot, TT_KEY] = key
        tt[slot, TT_DEPTH] = depth
        tt[slot, TT_BOUND] = bound
        tt[slot, TT_SCORE] = to_table(best, ply)
        tt[slot, TT_MOVE] = best_move
    return best


class Aborted(Exception):
    """The node budget ran out inside an iteration. `root_best` is the best root move that
    iteration had already proven, or 0 when its first move had not finished."""

    def __init__(self, root_best: int) -> None:
        super().__init__("node budget exhausted")
        self.root_best = root_best


class SearchState:
    """Every buffer the compiled search reads or writes, allocated once and reused for the game.

    `board`, `st` and `undo` are replaced by `set_position` each move; everything else is
    fixed. The table and history persist for the game, killers are cleared each move by
    `begin_move`, and `hist` holds every position the game has stood in, the current root
    last, appended by `remember`.
    """

    def __init__(self, pst: np.ndarray, w: np.ndarray) -> None:
        self.pst = pst
        self.w = w
        self.board, self.st, self.undo = fb.from_fen(fb.START_FEN)
        self.moves = np.zeros((MAX_PLY, fb.MAX_MOVES), dtype=np.int32)
        self.ranks = np.zeros((MAX_PLY, fb.MAX_MOVES), dtype=np.int64)
        self.killers = np.zeros((MAX_PLY, 2), dtype=np.int64)
        self.history = np.zeros((2, 120, 120), dtype=np.int64)
        self.tt = np.zeros((TT_SIZE, TT_COLUMNS), dtype=np.int64)
        self.path = np.zeros(MAX_PLY, dtype=np.int64)
        self.hist = np.zeros(HIST_SIZE, dtype=np.int64)
        self.ctl = np.zeros(C_COUNT, dtype=np.int64)
        self.ctl[C_MAX_NODES] = UNLIMITED
        self.ctl[C_USE_TABLE] = 1
        self.ctl[C_USE_KILLERS] = 1

    def set_position(self, fen: str) -> None:
        self.board, self.st, self.undo = fb.from_fen(fen)

    def key(self) -> int:
        return int(self.st[7])

    def legal_moves(self) -> list[int]:
        """The legal moves of the current position, for the root."""
        return fb.legal_moves(self.board, self.st, self.undo)

    def new_game(self) -> None:
        """Forget the table, the history counts and the game's positions."""
        self.tt[:, TT_KEY] = 0
        self.history[:] = 0
        self.ctl[C_N_HIST] = 0

    def decay_history(self) -> None:
        """Halve the history counts, as agent._observe does between moves."""
        self.history //= 2

    def remember(self, key: int) -> None:
        """Append a position the game has stood in. The root must be the last one appended
        before a search starts."""
        count = int(self.ctl[C_N_HIST])
        if count >= HIST_SIZE:
            self.hist[:-1] = self.hist[1:]
            count -= 1
        self.hist[count] = key
        self.ctl[C_N_HIST] = count + 1

    def begin_move(self, contempt: int) -> None:
        """Reset what belongs to one move: the killers, the counters and the abort flag."""
        self.killers[:] = 0
        self.ctl[C_NODES] = 0
        self.ctl[C_CUTOFFS] = 0
        self.ctl[C_DRAWS] = 0
        self.ctl[C_ABORT] = 0
        self.ctl[C_MAX_NODES] = UNLIMITED
        self.ctl[C_CONTEMPT] = contempt

    @property
    def nodes(self) -> int:
        return int(self.ctl[C_NODES])

    @property
    def cutoffs(self) -> int:
        return int(self.ctl[C_CUTOFFS])

    def table_move(self) -> int:
        """The table's move for the current position, or 0."""
        key = self.key()
        slot = key & (TT_SIZE - 1)
        if int(self.tt[slot, TT_KEY]) == key:
            return int(self.tt[slot, TT_MOVE])
        return 0

    def table_filled(self) -> int:
        """How many table rows hold an entry; a count for the log line, not a hot path."""
        return int(np.count_nonzero(self.tt[:, TT_KEY]))


def root(
    state: SearchState,
    depth: int,
    first: int,
    budget: Callable[[], int] | None = None,
) -> tuple[int, int]:
    """Search every root move at one depth, `first` first. Mirrors `agent._root`.

    Returns `(best_move, best_score)`. `budget`, when given, is called before each root move
    and returns the node count at which the search must stop; when the compiled search reaches
    it the iteration is abandoned and `Aborted` carries the best move it had proven. Because
    `first` is searched first, anything that replaced it has already outscored it.
    """
    board, st, undo = state.board, state.st, state.undo
    row = state.moves[0]
    count = fb.gen_legal(board, st, undo, row)
    ranks = state.ranks[0]
    rank_moves(
        board,
        int(st[0]),
        row,
        count,
        first,
        state.killers[0],
        state.history,
        state.w,
        ranks,
        state.ctl[C_USE_KILLERS] != 0,
    )
    order = sorted(range(count), key=lambda i: -int(ranks[i]))
    moves = [int(row[i]) for i in order]
    state.path[0] = state.key()
    best_move, best_score = moves[0], -INFINITY
    root_best = 0
    for move in moves:
        if budget is not None:
            limit = budget()
            if limit <= state.nodes:
                raise Aborted(root_best)
            state.ctl[C_MAX_NODES] = limit
        fb.make_move(board, st, undo, move)
        score = -int(
            negamax(
                board,
                st,
                undo,
                state.pst,
                state.w,
                state.moves,
                state.ranks,
                state.killers,
                state.history,
                state.tt,
                state.path,
                state.hist,
                state.ctl,
                depth - 1,
                1,
                -INFINITY,
                -best_score,
            )
        )
        fb.unmake_move(board, st, undo, move)
        if state.ctl[C_ABORT] != 0:
            raise Aborted(root_best)
        if score > best_score:
            best_move, best_score = move, score
            root_best = move
    return best_move, best_score


def warm(state: SearchState, depth: int = 3) -> int:
    """Run a short search from the start position so every compiled path has executed once.

    The eager signatures above compile at import; this proves the whole graph runs, before
    the clock starts. Returns the nodes it searched.
    """
    state.set_position(fb.START_FEN)
    state.new_game()
    state.remember(state.key())
    state.begin_move(0)
    moves = fb.legal_moves(state.board, state.st, state.undo)
    root(state, depth, moves[0])
    state.new_game()
    return state.nodes
