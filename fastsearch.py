"""The search of `agent.py`, rewritten as `@njit` code over the `fastboard` state.

Like `fasteval`, this is a port rather than a redesign: iterative deepening, fail-soft
alpha-beta negamax, quiescence over captures and queen promotions, MVV-LVA / killers / history
/ table-move ordering, a transposition table, repetition and fifty-move draws scored with
contempt, insufficient material, mate scored by ply, and the same budgets. Where a decision is
explained in `agent.py` it is explained there and not repeated; what follows is what had to
change to run without Python objects.

**The clock.** numba cannot call `time.perf_counter()` directly, but it can call it through
`numba.objmode`, and that costs about 340 nanoseconds. So the search reads the real clock every
1024 nodes exactly as `agent.py` does, and every 128 under a short budget. At a million nodes a
second that is a clock read every millisecond, twenty times finer than the twenty milliseconds
that would keep a move safe, and the reads cost about three parts in ten thousand of the
search. Driving the deadline from a node-count estimate instead would have meant guessing a
node rate that varies by position, and being wrong about it near the end of a long clock is a
flag; this way the budget arithmetic, the deadline and the abort keep exactly the meaning they
have in `agent.py`. Python still drives iterative deepening, one jitted call per depth, so an
abort also lands cleanly between iterations.

**Aborting.** A timeout cannot unwind through a `raise` here, so `stats[ABORTED]` is set and
every frame returns as soon as it sees it, *after* unmaking its move. The board, the undo stack
and the ply-indexed buffers are all consistent when the abort reaches Python, unlike
`agent.py`, which deliberately leaves its board mid-line. Nothing is stored to the table on the
way out, so no half-searched score is ever written.

**State.** numba freezes a module-level array as a read-only constant, so every array the
search writes to is allocated once at import and passed in. `state()` hands back the whole set
as one tuple and `think` unpacks it; nothing in this file allocates per node.

**The table** is one `int64[2**21, 2]` array indexed by `key & TT_MASK`: column 0 is the full
64-bit key, as verification against the two positions that share a slot, and column 1 packs the
depth, the bound, the score relative to the node's ply, and the move into the other 64 bits.
Packing is not tidiness. Five parallel arrays would put five cache lines between a probe and
its answer, and a transposition probe is a random access into 33 MB, so every one of them is a
miss; this way a probe touches one 16-byte entry and misses once. It is **replace-always**: a
store overwrites whatever shares the slot, which is what `agent.py` effectively does (it clears
the entire dictionary when it fills, which is strictly worse) and costs nothing on the store
path, where depth-preferred costs a load and a compare on every store.

**Repetition** is the one place the data structure changed the algorithm. `agent.py` keeps a
set of every position the game has stood in and a set of the current path, and asks both. Here
they are arrays, and scanning them per node would cost more than the node does. Instead the
lookback is bounded by the halfmove clock and stepped by two, and both of those are exact
rather than approximations: a repetition needs the same side to move, and a position on the far
side of a capture or a pawn move has different material or a different pawn structure, so a
different Zobrist key, and cannot repeat this one at all. The scan walks back at most `st[3]`
plies, through the search path first and then on into the game history, and finds exactly what
the two sets would have found. In the middlegame the halfmove clock is small and the loop is a
handful of comparisons.
"""

import time

import numpy as np
from numba import njit, objmode
from numba import types as nbt

import fastboard as fb
from fastboard import (
    FLAG_EP,
    MAX_MOVES,
    UNDO_SIZE,
    gen_moves,
    is_square_attacked,
    make_move,
    unmake_move,
)

# `PIECE_VALUES` is the evaluation's, read here only for MVV-LVA ordering.
from fasteval import PIECE_VALUES, STALEMATE_PIECE_LIMIT, evaluate

# --------------------------------------------------------------------------------------
# Every constant below is the one `agent.py` searches with. Changing one here and not
# there makes the two engines different engines, and the fallback in `agent.py` exists to
# make swapping between them invisible.
# --------------------------------------------------------------------------------------

MATE = 1_000_000
INFINITY = MATE + 1
MATE_FOUND = MATE - 1_000

MAX_DEPTH = 64
QUIESCENCE_MAX_PLY = 8
# The deepest ply a frame can reach, and the size every ply-indexed buffer needs.
MAX_SEARCH_PLY = MAX_DEPTH + QUIESCENCE_MAX_PLY + 2

NODE_CHECK_MASK = 1023
FINE_CHECK_MASK = 127
FINE_CHECK_BELOW_MS = 300

SOFT_DIVISOR = 25
SOFT_BONUS_MS = 400
HARD_DIVISOR = 8
SAFETY_MARGIN_MS = 300
PANIC_MS = 1_000
GROWTH_MIN = 2.0
GROWTH_MAX = 8.0
GROWTH_UNKNOWN = 5.0
NPS_FLOOR_MS = 5

TABLE_BONUS = 4_000_000
CAPTURE_BONUS = 1_000_000
PROMOTION_BONUS = 500_000
KILLER_BONUS = 400_000
HISTORY_CAP = KILLER_BONUS - 3

EXACT, LOWER, UPPER = 0, 1, 2

CONTEMPT = 50
CONTEMPT_THRESHOLD = 150

FIFTY_MOVE_PLIES = 100

# --------------------------------------------------------------------------------------
# The transposition table's packing. A move occupies bits 0..19 (`fastboard` puts its
# highest flag at bit 19), the depth is stored one higher than it is so that an all-zero
# entry reads as empty, and the score is shifted by INFINITY so it is never negative.
# --------------------------------------------------------------------------------------

TT_BITS = 21
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1

TT_MOVE_MASK = (1 << 20) - 1
TT_DEPTH_SHIFT = 20
TT_DEPTH_MASK = (1 << 8) - 1
TT_BOUND_SHIFT = 28
TT_BOUND_MASK = 3
TT_SCORE_SHIFT = 30
TT_SCORE_MASK = (1 << 22) - 1

# Where the counters and the two per-move settings live in the `stats` array.
NODES, CUTOFFS, DRAWS, ABORTED = 0, 1, 2, 3
CONTEMPT_AT, CHECK_MASK, BEST_MOVE, ROOT_BEST = 4, 5, 6, 7
TT_PROBES, TT_HITS, TT_STORES, GAME_COUNT = 8, 9, 10, 11
STATS_SIZE = 16

# Read-only, so numba can hold it as a global constant.
PIECE_VALUE_BY_KIND = np.array(PIECE_VALUES, dtype=np.int32)

_BOARD_T = nbt.int8[::1]
_ST_T = nbt.int64[::1]
_UNDO_T = nbt.int64[:, ::1]
_TT_T = nbt.int64[:, ::1]
_BUFS_T = nbt.int32[:, ::1]
_KILLERS_T = nbt.int32[:, ::1]
_HISTORY_T = nbt.int32[:, :, ::1]
_KEYS_T = nbt.int64[::1]
_STATS_T = nbt.int64[::1]


def state() -> tuple[np.ndarray, ...]:
    """Allocate the whole of the search's memory. Called once, at import.

    Returned as a tuple because every one of these is written from jitted code and so has to
    be an argument rather than a global: `(tt, bufs, scores, killers, history, path, game,
    stats)`.
    """
    return (
        np.zeros((TT_SIZE, 2), dtype=np.int64),
        np.zeros((MAX_SEARCH_PLY, MAX_MOVES), dtype=np.int32),
        np.zeros((MAX_SEARCH_PLY, MAX_MOVES), dtype=np.int32),
        np.zeros((MAX_SEARCH_PLY, 2), dtype=np.int32),
        # Mailbox from-square and to-square, so nothing sits between the board and this.
        np.zeros((2, 120, 120), dtype=np.int32),
        # The keys of the ancestors of the node being searched. Index 0 is never read: the
        # root is the last entry of the game history instead.
        np.zeros(MAX_SEARCH_PLY, dtype=np.int64),
        # Every position this game has stood in, oldest first. Sized like the undo stack,
        # which a 600-ply game cannot overflow.
        np.zeros(UNDO_SIZE, dtype=np.int64),
        np.zeros(STATS_SIZE, dtype=np.int64),
    )


TT, BUFS, SCORES, KILLERS, HISTORY, PATH, GAME_KEYS, STATS = state()


# --------------------------------------------------------------------------------------
# Leaf helpers.
# --------------------------------------------------------------------------------------


@njit(nbt.float64(), cache=False)
def clock() -> float:
    """`time.perf_counter()` from jitted code, at about 340 nanoseconds a call."""
    with objmode(seconds="float64"):
        seconds = time.perf_counter()
    return seconds


@njit(nbt.boolean(_BOARD_T), cache=False)
def insufficient_material(board: np.ndarray) -> bool:
    """python-chess's `is_insufficient_material`, over the mailbox.

    The scan returns on the first pawn, rook or queen, and squares 21 upward are the back
    ranks, so in anything but a bare endgame this costs a handful of loads. What survives that
    gate is python-chess's rule exactly: a lone knight is insufficient only against a bare
    king, and bishops are insufficient only when every bishop on the board stands on one
    square colour and there is no knight anywhere.
    """
    white_knights = 0
    white_bishops = 0
    black_knights = 0
    black_bishops = 0
    light = 0
    dark = 0
    for square in range(21, 99):
        piece = board[square]
        if piece == 0 or piece == 13:
            continue
        kind = piece if piece <= 6 else piece - 6
        if kind == 1 or kind == 4 or kind == 5:
            return False
        if kind == 2:
            if piece <= 6:
                white_knights += 1
            else:
                black_knights += 1
        elif kind == 3:
            if piece <= 6:
                white_bishops += 1
            else:
                black_bishops += 1
            offset = square - 21
            if ((offset % 10) + (offset // 10)) % 2 == 0:
                dark += 1
            else:
                light += 1
    knights = white_knights + black_knights
    one_colour = light == 0 or dark == 0
    if white_knights > 0:
        if white_knights > 1 or white_bishops > 0 or black_knights > 0 or black_bishops > 0:
            return False
    elif white_bishops > 0 and not (one_colour and knights == 0):
        return False
    if black_knights > 0:
        if black_knights > 1 or black_bishops > 0 or white_knights > 0 or white_bishops > 0:
            return False
    elif black_bishops > 0 and not (one_colour and knights == 0):
        return False
    return True


@njit(nbt.boolean(_BOARD_T, nbt.int64, nbt.int64), cache=False)
def men_at_most(board: np.ndarray, side: int, limit: int) -> bool:
    """Has `side` at most `limit` men? Counted from that side's own end of the board.

    The count stops at the limit and the scan starts where that side's pieces are, so in a
    middlegame this answers after a dozen loads instead of walking all 78 squares. That
    matters because the quiescence stalemate check asks it at every quiet leaf.
    """
    low = 1 + 6 * side
    high = low + 5
    count = 0
    if side == 0:
        start, stop, step = 98, 20, -1
    else:
        start, stop, step = 21, 99, 1
    for square in range(start, stop, step):
        piece = board[square]
        if low <= piece <= high:
            count += 1
            if count > limit:
                return False
    return True


@njit(nbt.int64(_BOARD_T, _ST_T, _UNDO_T, _BUFS_T, nbt.int64), cache=False)
def count_legal(
    board: np.ndarray, st: np.ndarray, undo: np.ndarray, bufs: np.ndarray, ply: int
) -> int:
    """Filter this ply's buffer down to the legal moves, in place, and return how many.

    Pseudo-legal, then make and look at the king, which is the loop `perft` runs.
    """
    out = bufs[ply]
    generated = gen_moves(board, st, out)
    side = st[0]
    other = 1 - side
    kept = 0
    for index in range(generated):
        move = out[index]
        make_move(board, st, undo, move)
        legal = not is_square_attacked(board, st[5 + side], other)
        unmake_move(board, st, undo, move)
        if legal:
            out[kept] = move
            kept += 1
    return kept


@njit(nbt.int64(_STATS_T, nbt.int64), cache=False)
def draw_score(stats: np.ndarray, ply: int) -> int:
    """What a draw is worth to the side to move here. See `_draw_score` in `agent.py`."""
    contempt = int(stats[CONTEMPT_AT])
    return contempt if ply % 2 == 0 else -contempt


@njit(nbt.boolean(_ST_T, _KEYS_T, _KEYS_T, _STATS_T, nbt.int64), cache=False)
def repeated(
    st: np.ndarray, path: np.ndarray, game: np.ndarray, stats: np.ndarray, ply: int
) -> bool:
    """Has this exact position already occurred, on this path or earlier in the game?

    Walking back `k` plies reads the search path while `k` is still inside the search, and
    the game history once it is past the root. See the module docstring for why the halfmove
    clock is an exact bound on `k` and why the step is two.
    """
    key = st[7]
    reach = st[3]
    played = stats[GAME_COUNT]
    back = 2
    while back <= reach:
        index = ply - back
        if index >= 1:
            other = path[index]
        else:
            # `index` at or below zero is `-index` plies before the root, and the root is
            # the last entry of the game history, so this counts back from the end of it.
            position = played - 1 + index
            if position < 0:
                return False
            other = game[position]
        if other == key:
            return True
        back += 2
    return False


@njit(nbt.int64(nbt.int64, nbt.int64), cache=False)
def to_table(score: int, ply: int) -> int:
    """Rewrite a mate score to be counted from this node rather than from the root."""
    if score > MATE_FOUND:
        return score + ply
    if score < -MATE_FOUND:
        return score - ply
    return score


@njit(nbt.int64(nbt.int64, nbt.int64), cache=False)
def from_table(score: int, ply: int) -> int:
    """Undo `to_table` for a node at this ply."""
    if score > MATE_FOUND:
        return score - ply
    if score < -MATE_FOUND:
        return score + ply
    return score


@njit(
    nbt.void(
        _TT_T, _STATS_T, nbt.int64, nbt.int64, nbt.int64, nbt.int64, nbt.int32, nbt.int64
    ),
    cache=False,
)
def store(
    tt: np.ndarray,
    stats: np.ndarray,
    key: int,
    depth: int,
    bound: int,
    score: int,
    move: int,
    ply: int,
) -> None:
    """Write what this node proved. Replace-always: the slot belongs to whoever wrote last."""
    slot = key & TT_MASK
    tt[slot, 0] = key
    tt[slot, 1] = (
        (move & TT_MOVE_MASK)
        | ((depth + 1) << TT_DEPTH_SHIFT)
        | (bound << TT_BOUND_SHIFT)
        | ((to_table(score, ply) + INFINITY) << TT_SCORE_SHIFT)
    )
    stats[TT_STORES] += 1


# --------------------------------------------------------------------------------------
# Move ordering.
# --------------------------------------------------------------------------------------


@njit(nbt.int64(_BOARD_T, nbt.int32), cache=False)
def move_score(board: np.ndarray, move: int) -> int:
    """MVV-LVA for captures, then promotions. Zero for a quiet move, as `_move_score` is."""
    score = 0
    victim = board[(move >> 7) & 127]
    en_passant = (move & FLAG_EP) != 0
    if victim != 0 or en_passant:
        # An en passant capture lands on an empty square; its victim is always a pawn.
        if en_passant:
            victim_value = PIECE_VALUE_BY_KIND[0]
        else:
            victim_value = PIECE_VALUE_BY_KIND[(victim if victim <= 6 else victim - 6) - 1]
        attacker = board[move & 127]
        attacker_value = PIECE_VALUE_BY_KIND[(attacker if attacker <= 6 else attacker - 6) - 1]
        # Victim value dominates; the attacker only breaks ties, cheapest attacker first.
        score += CAPTURE_BONUS + victim_value * 100 - attacker_value
    promo = (move >> 14) & 7
    if promo != 0:
        score += PROMOTION_BONUS + PIECE_VALUE_BY_KIND[promo - 1]
    return score


@njit(
    nbt.void(
        _BOARD_T,
        _ST_T,
        _BUFS_T,
        _BUFS_T,
        _KILLERS_T,
        _HISTORY_T,
        nbt.int64,
        nbt.int64,
        nbt.int32,
    ),
    cache=False,
)
def score_moves(
    board: np.ndarray,
    st: np.ndarray,
    bufs: np.ndarray,
    scores: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    ply: int,
    count: int,
    table_move: int,
) -> None:
    """Rank every move in this ply's buffer, exactly as `_order_fully` ranks one."""
    moves = bufs[ply]
    out = scores[ply]
    side = st[0]
    for index in range(count):
        move = moves[index]
        if move == table_move:
            out[index] = TABLE_BONUS
            continue
        rank = move_score(board, move)
        if rank != 0:  # non-zero means a capture or a promotion
            out[index] = rank
        elif move == killers[ply, 0]:
            out[index] = KILLER_BONUS
        elif move == killers[ply, 1]:
            out[index] = KILLER_BONUS - 1
        else:
            seen = history[side, move & 127, (move >> 7) & 127]
            out[index] = seen if seen < HISTORY_CAP else HISTORY_CAP


@njit(nbt.void(_BOARD_T, _BUFS_T, _BUFS_T, nbt.int64, nbt.int64), cache=False)
def score_captures(
    board: np.ndarray, bufs: np.ndarray, scores: np.ndarray, ply: int, count: int
) -> None:
    """Rank a quiescence move list, which holds only captures and promotions."""
    moves = bufs[ply]
    out = scores[ply]
    for index in range(count):
        out[index] = move_score(board, moves[index])


@njit(nbt.void(_BUFS_T, _BUFS_T, nbt.int64, nbt.int64, nbt.int64), cache=False)
def pick_best(bufs: np.ndarray, scores: np.ndarray, ply: int, first: int, count: int) -> None:
    """Swap the best of the moves from `first` onward into position `first`.

    Selecting one move at a time rather than sorting the list is what makes ordering cheap: a
    node that cuts off on its first or second move never ranks the rest against each other.
    The strict comparison leaves moves of equal score in the order they were generated, which
    is what `agent.py`'s stable sort does.
    """
    moves = bufs[ply]
    ranks = scores[ply]
    best = first
    for index in range(first + 1, count):
        if ranks[index] > ranks[best]:
            best = index
    if best != first:
        moves[first], moves[best] = moves[best], moves[first]
        ranks[first], ranks[best] = ranks[best], ranks[first]


# --------------------------------------------------------------------------------------
# Quiescence and negamax.
# --------------------------------------------------------------------------------------

_SEARCH_SIG = (
    _BOARD_T,
    _ST_T,
    _UNDO_T,
    _TT_T,
    _BUFS_T,
    _BUFS_T,
    _KILLERS_T,
    _HISTORY_T,
    _KEYS_T,
    _KEYS_T,
    _STATS_T,
    nbt.float64,
    nbt.int64,
    nbt.int64,
    nbt.int64,
    nbt.int64,
)


@njit(nbt.int64(*_SEARCH_SIG), cache=False)
def quiescence(
    board: np.ndarray,
    st: np.ndarray,
    undo: np.ndarray,
    tt: np.ndarray,
    bufs: np.ndarray,
    scores: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    path: np.ndarray,
    game: np.ndarray,
    stats: np.ndarray,
    deadline: float,
    remaining: int,
    ply: int,
    alpha: int,
    beta: int,
) -> int:
    """Search the noisy continuations so the evaluation is never read mid-exchange."""
    stats[NODES] += 1
    if (stats[NODES] & stats[CHECK_MASK]) == 0 and clock() > deadline:
        stats[ABORTED] = 1
        return 0

    side = st[0]
    other = 1 - side
    out = bufs[ply]

    if is_square_attacked(board, st[5 + side], other):
        # In check there is no standing pat and every evasion has to be looked at. That is
        # also the only place a checkmate can hide down here, and it makes it exact.
        count = count_legal(board, st, undo, bufs, ply)
        if count == 0:
            # Mate is scored by ply so a shorter mate outranks a longer one.
            return -MATE + ply
        if remaining == 0:
            return evaluate(board, st)
        best = -INFINITY
        score_moves(board, st, bufs, scores, killers, history, ply, count, 0)
    else:
        # A stalemate down here would otherwise score as the stand-pat. Proving it costs a
        # full move generation, far too much at every quiet leaf, so it is asked only when the
        # side to move is a king and at most two other men: exactly the side that gets
        # stalemated, and by then generating is nearly free.
        if men_at_most(board, side, STALEMATE_PIECE_LIMIT) and (
            count_legal(board, st, undo, bufs, ply) == 0
        ):
            stats[DRAWS] += 1
            return draw_score(stats, ply)
        best = evaluate(board, st)
        if best >= beta or remaining == 0:
            return best
        if best > alpha:
            alpha = best
        # Captures, plus the quiet promotions to a queen, which change material as much as a
        # capture does. Only the queen: an underpromotion is a way to avoid a stalemate or to
        # fork, and neither is something a search of the noisy moves alone can see.
        generated = gen_moves(board, st, out)
        count = 0
        for index in range(generated):
            move = out[index]
            quiet = board[(move >> 7) & 127] == 0 and (move & FLAG_EP) == 0
            if quiet and ((move >> 14) & 7) != 5:
                continue
            make_move(board, st, undo, move)
            legal = not is_square_attacked(board, st[5 + side], other)
            unmake_move(board, st, undo, move)
            if legal:
                out[count] = move
                count += 1
        score_captures(board, bufs, scores, ply, count)

    for index in range(count):
        pick_best(bufs, scores, ply, index, count)
        move = out[index]
        make_move(board, st, undo, move)
        score = -quiescence(
            board,
            st,
            undo,
            tt,
            bufs,
            scores,
            killers,
            history,
            path,
            game,
            stats,
            deadline,
            remaining - 1,
            ply + 1,
            -beta,
            -alpha,
        )
        unmake_move(board, st, undo, move)
        if stats[ABORTED] != 0:
            return 0
        if score > best:
            best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
    return best


@njit(nbt.int64(*_SEARCH_SIG), cache=False)
def negamax(
    board: np.ndarray,
    st: np.ndarray,
    undo: np.ndarray,
    tt: np.ndarray,
    bufs: np.ndarray,
    scores: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    path: np.ndarray,
    game: np.ndarray,
    stats: np.ndarray,
    deadline: float,
    depth: int,
    ply: int,
    alpha: int,
    beta: int,
) -> int:
    """Fail-soft alpha-beta by observing that both sides' scores are exact opposites."""
    stats[NODES] += 1
    if (stats[NODES] & stats[CHECK_MASK]) == 0 and clock() > deadline:
        stats[ABORTED] = 1
        return 0

    key = st[7]
    side = st[0]
    other = 1 - side

    # A position the game has already stood in, or one already on this path, is a draw: the
    # referee claims the third occurrence and the second is the move that offers it. Counting
    # two rather than three is the standard simplification; `agent.py` says what it costs. A
    # repeated position is never checkmate, since the game would have ended the first time.
    if repeated(st, path, game, stats, ply):
        stats[DRAWS] += 1
        return draw_score(stats, ply)
    # The fifty move rule does not rescue a side that is being mated: mate ends the game
    # first, so a position with no escape from check is scored below, not here.
    if st[3] >= FIFTY_MOVE_PLIES and (
        not is_square_attacked(board, st[5 + side], other)
        or count_legal(board, st, undo, bufs, ply) > 0
    ):
        stats[DRAWS] += 1
        return draw_score(stats, ply)
    # Before the depth check, because quiescence would score a dead draw off the tables
    # instead, and at depth 1 that is every leaf.
    if insufficient_material(board):
        return draw_score(stats, ply)
    if depth <= 0:
        return quiescence(
            board,
            st,
            undo,
            tt,
            bufs,
            scores,
            killers,
            history,
            path,
            game,
            stats,
            deadline,
            QUIESCENCE_MAX_PLY,
            ply,
            alpha,
            beta,
        )

    table_move = 0
    slot = key & TT_MASK
    stats[TT_PROBES] += 1
    entry = tt[slot, 1]
    if tt[slot, 0] == key and entry != 0:
        stats[TT_HITS] += 1
        table_move = entry & TT_MOVE_MASK
        stored_depth = ((entry >> TT_DEPTH_SHIFT) & TT_DEPTH_MASK) - 1
        if stored_depth >= depth:
            score = from_table(((entry >> TT_SCORE_SHIFT) & TT_SCORE_MASK) - INFINITY, ply)
            bound = (entry >> TT_BOUND_SHIFT) & TT_BOUND_MASK
            # An exact score settles the node. A bound only settles it when it already falls
            # outside the window we were asked about.
            if (
                bound == EXACT
                or (bound == LOWER and score >= beta)
                or (bound == UPPER and score <= alpha)
            ):
                return score

    count = count_legal(board, st, undo, bufs, ply)
    if count == 0:
        if is_square_attacked(board, st[5 + side], other):
            return -MATE + ply
        return draw_score(stats, ply)

    out = bufs[ply]
    draws_before = stats[DRAWS]
    window_alpha = alpha
    best = -INFINITY
    best_move = out[0]
    path[ply] = key
    score_moves(board, st, bufs, scores, killers, history, ply, count, table_move)
    for index in range(count):
        pick_best(bufs, scores, ply, index, count)
        move = out[index]
        make_move(board, st, undo, move)
        score = -negamax(
            board,
            st,
            undo,
            tt,
            bufs,
            scores,
            killers,
            history,
            path,
            game,
            stats,
            deadline,
            depth - 1,
            ply + 1,
            -beta,
            -alpha,
        )
        unmake_move(board, st, undo, move)
        if stats[ABORTED] != 0:
            return 0
        if score > best:
            best = score
            best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                stats[CUTOFFS] += 1
                # Captures already order themselves by what they win, so only quiet moves are
                # remembered. Squared, because a cutoff found deep in the tree stood up to far
                # more refutations than one found at a leaf.
                if board[(move >> 7) & 127] == 0 and (move & (FLAG_EP | (7 << 14))) == 0:
                    if move != killers[ply, 0]:
                        killers[ply, 1] = killers[ply, 0]
                        killers[ply, 0] = move
                    history[side, move & 127, (move >> 7) & 127] += depth * depth
                break

    # A score that came out of a repetition belongs to the path, not to the position, so it
    # is not written. `agent.py` has the rest of the reasoning, including why contempt is
    # allowed to colour the entries that are written.
    if stats[DRAWS] == draws_before:
        if best <= window_alpha:
            bound = UPPER
        elif best >= beta:
            bound = LOWER
        else:
            bound = EXACT
        store(tt, stats, key, depth, bound, best, best_move, ply)
    return best


@njit(
    nbt.int64(
        _BOARD_T,
        _ST_T,
        _UNDO_T,
        _TT_T,
        _BUFS_T,
        _BUFS_T,
        _KILLERS_T,
        _HISTORY_T,
        _KEYS_T,
        _KEYS_T,
        _STATS_T,
        nbt.float64,
        nbt.int64,
        nbt.int32,
    ),
    cache=False,
)
def search_root(
    board: np.ndarray,
    st: np.ndarray,
    undo: np.ndarray,
    tt: np.ndarray,
    bufs: np.ndarray,
    scores: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    path: np.ndarray,
    game: np.ndarray,
    stats: np.ndarray,
    deadline: float,
    depth: int,
    first: int,
) -> int:
    """Search every root move at one depth, trying the previous iteration's best move first.

    Writes the best move to `stats[BEST_MOVE]` and, separately, the move that most recently
    improved on the one before it to `stats[ROOT_BEST]`. Because `first` is searched first,
    anything in `ROOT_BEST` has already outscored it at this depth, so an aborted iteration
    only ever hands back a move it has proven better.
    """
    stats[ROOT_BEST] = 0
    count = count_legal(board, st, undo, bufs, 0)
    if count == 0:
        stats[BEST_MOVE] = 0
        return 0
    # The root's own killers are always empty: beta is infinity here, so no root move ever
    # cuts off and nothing is recorded at ply 0. Passing them keeps one ordering path.
    score_moves(board, st, bufs, scores, killers, history, 0, count, first)

    out = bufs[0]
    best_score = -INFINITY
    pick_best(bufs, scores, 0, 0, count)
    best_move = out[0]
    for index in range(count):
        pick_best(bufs, scores, 0, index, count)
        move = out[index]
        make_move(board, st, undo, move)
        score = -negamax(
            board,
            st,
            undo,
            tt,
            bufs,
            scores,
            killers,
            history,
            path,
            game,
            stats,
            deadline,
            depth - 1,
            1,
            -INFINITY,
            -best_score,
        )
        unmake_move(board, st, undo, move)
        if stats[ABORTED] != 0:
            stats[BEST_MOVE] = best_move
            return best_score
        if score > best_score:
            best_score = score
            best_move = move
            stats[ROOT_BEST] = move
    stats[BEST_MOVE] = best_move
    return best_score


# --------------------------------------------------------------------------------------
# The Python driver: iterative deepening, the budgets, and the game history across moves.
# --------------------------------------------------------------------------------------

# The position we handed back last move, as a fen. The next one has to be one move on.
_EXPECTED: str | None = None


def reset() -> None:
    """Forget everything. A process is meant to live for one game; this is another game."""
    global _EXPECTED
    TT.fill(0)
    HISTORY.fill(0)
    KILLERS.fill(0)
    STATS[GAME_COUNT] = 0
    _EXPECTED = None


def budgets(time_left_ms: int) -> tuple[float, float]:
    """The soft and hard budgets in milliseconds, as `_budgets` computes them."""
    soft = time_left_ms / SOFT_DIVISOR + SOFT_BONUS_MS
    hard = max(min(time_left_ms / HARD_DIVISOR, time_left_ms - SAFETY_MARGIN_MS), 0.0)
    return min(soft, hard), hard


def contempt_for(root_score: int) -> int:
    """What a draw is worth to us, given how the root position stands from our side."""
    if root_score > CONTEMPT_THRESHOLD:
        return -CONTEMPT
    if root_score < -CONTEMPT_THRESHOLD:
        return CONTEMPT
    return 0


def projected(last_ms: float, previous_ms: float) -> float:
    """What the next iteration costs, from the last one and how fast cost is growing."""
    growth = last_ms / previous_ms if previous_ms > 0.0 else GROWTH_UNKNOWN
    return last_ms * min(max(growth, GROWTH_MIN), GROWTH_MAX)


def reachable(previous: str, key: int) -> bool:
    """Is the position with this key one legal move on from the one we handed back?"""
    board, st, undo = fb.from_fen(previous)
    for move in fb.legal_moves(board, st, undo):
        make_move(board, st, undo, move)
        found = int(st[7]) == key
        unmake_move(board, st, undo, move)
        if found:
            return True
    return False


def observe(key: int) -> None:
    """Fold the position we were handed into the game history, forgetting another game.

    The rule is `_observe`'s: the first fen we are given is where the game starts for
    repetition and fifty-move purposes, and a position that is not one legal move on from
    what we handed back belongs to a game we are not in.
    """
    if _EXPECTED is None or not reachable(_EXPECTED, key):
        reset()
    else:
        # Halved rather than kept or cleared: which quiet moves cut off is still largely true
        # two plies later, but it decays, and halving stops a long game's counts running away.
        np.floor_divide(HISTORY, 2, out=HISTORY)
    remember(key)


def remember(key: int) -> None:
    """Append a position to the game history, if there is room left in it."""
    played = int(STATS[GAME_COUNT])
    if played < GAME_KEYS.shape[0]:
        GAME_KEYS[played] = key
        STATS[GAME_COUNT] = played + 1


def think(fen: str, time_left_ms: int) -> str:
    """Deepen until the budget is spent, keeping the best move proven so far.

    Returns a UCI move, or `0000` when the position has none. Raises on a fen `fastboard`
    cannot parse, which is the caller's cue to fall back.
    """
    global _EXPECTED
    started = time.perf_counter()
    board, st, undo = fb.from_fen(fen)
    moves = fb.legal_moves(board, st, undo)
    if not moves:
        return "0000"

    observe(int(st[7]))
    soft_ms, hard_ms = budgets(time_left_ms)
    for counter in (NODES, CUTOFFS, DRAWS, ABORTED, TT_PROBES, TT_HITS, TT_STORES):
        STATS[counter] = 0
    STATS[CHECK_MASK] = NODE_CHECK_MASK if hard_ms >= FINE_CHECK_BELOW_MS else FINE_CHECK_MASK
    # Contempt is set once, from the static evaluation, and left alone; see `_think` for why
    # reading it off the previous iteration's score feeds back on itself.
    STATS[CONTEMPT_AT] = contempt_for(int(evaluate(board, st)))
    KILLERS.fill(0)
    deadline = started + hard_ms / 1000.0

    # Before the first iteration the only ordering there is is the static one, plus whatever
    # the table already knows about this position from the search two plies ago.
    best = max(moves, key=lambda move: move_score(board, move))
    slot = int(st[7]) & TT_MASK
    if TT[slot, 0] == st[7] and TT[slot, 1] != 0:
        stored = int(TT[slot, 1] & TT_MOVE_MASK)
        if stored in moves:
            best = stored

    best_score = 0
    reached = 0
    partial = False
    last_ms, previous_ms = 0.0, 0.0

    # A zero budget means the clock is under the safety margin, and then even the first
    # hundred nodes are time we do not have: the ordered first move is the whole reply.
    deepest = 0 if hard_ms <= 0.0 else 1 if time_left_ms < PANIC_MS else MAX_DEPTH
    for depth in range(1, deepest + 1):
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Start an iteration only when the soft budget is projected to fall inside it; see
        # the same gate in `agent.py` for why this is not "have we spent the budget yet".
        if depth > 1 and elapsed_ms + projected(last_ms, previous_ms) / 2.0 > soft_ms:
            break
        iteration_started = time.perf_counter()
        score = int(
            search_root(
                board, st, undo, TT, BUFS, SCORES, KILLERS, HISTORY, PATH, GAME_KEYS, STATS,
                deadline, depth, best,
            )
        )
        if STATS[ABORTED] != 0:
            if STATS[ROOT_BEST] != 0:
                best = int(STATS[ROOT_BEST])
                partial = True
            break
        best = int(STATS[BEST_MOVE])
        best_score = score
        previous_ms = last_ms
        last_ms = (time.perf_counter() - iteration_started) * 1000.0
        reached = depth
        if best_score >= MATE_FOUND:
            break  # A forced mate is in hand; searching deeper cannot shorten it.

    spent_ms = (time.perf_counter() - started) * 1000.0
    depth_text = f"d{reached} score {best_score:+d}" if reached else "d0"
    move_text = f"move {fb.move_to_uci(best)}"
    if partial:
        move_text += f" from partial d{reached + 1}"
    rate_text = ""
    if spent_ms >= NPS_FLOOR_MS:
        rate_text = f"nps {STATS[NODES] / spent_ms * 1000.0:.0f} "
    hit_rate = f"{STATS[TT_HITS] / STATS[TT_PROBES] * 100.0:.0f}%" if STATS[TT_PROBES] else "-"
    print(
        f"{depth_text} {move_text} nodes {STATS[NODES]} "
        f"{rate_text}{spent_ms:.0f}ms soft {soft_ms:.0f} hard {hard_ms:.0f} "
        f"clock {time_left_ms} tt {hit_rate} cut {STATS[CUTOFFS]} "
        f"contempt {STATS[CONTEMPT_AT]:+d}",
        flush=True,
    )

    # We are handed the position after the opponent's reply next, so both this position and
    # the one we are about to make are part of the game's history.
    after_board, after_st, after_undo = fb.from_fen(fen)
    make_move(after_board, after_st, after_undo, best)
    remember(int(after_st[7]))
    _EXPECTED = fb.to_fen(after_board, after_st)
    return fb.move_to_uci(best)


def search_fixed(
    fen: str, depth: int, fresh: bool = True, contempt: int = 0
) -> tuple[str, int, int]:
    """Search one position to a fixed depth with no deadline. For tests and benchmarks.

    `fresh` clears the table and the game history first, so a measurement is not quietly
    helped by whatever the previous call left behind; passing False keeps a game history a
    caller has set up by hand, which is how the repetition tests reach this path.
    """
    if fresh:
        reset()
    board, st, undo = fb.from_fen(fen)
    for counter in (NODES, CUTOFFS, DRAWS, ABORTED, TT_PROBES, TT_HITS, TT_STORES):
        STATS[counter] = 0
    STATS[CHECK_MASK] = NODE_CHECK_MASK
    STATS[CONTEMPT_AT] = contempt
    KILLERS.fill(0)
    remember(int(st[7]))
    score = int(
        search_root(
            board, st, undo, TT, BUFS, SCORES, KILLERS, HISTORY, PATH, GAME_KEYS, STATS,
            time.perf_counter() + 86_400.0, depth, 0,
        )
    )
    return fb.move_to_uci(int(STATS[BEST_MOVE])), score, int(STATS[NODES])


def warm() -> None:
    """Run every jitted function once with the types it will really see.

    The eager signatures above compile at import; this proves the whole graph runs and warms
    the two recursive specialisations, which is where a first call on the clock would cost
    the most. The bare-king position at the end runs the mop-up, the quiescence stalemate
    check and the insufficient-material return, so a failure in any of them shows up here
    rather than in a game.
    """
    far = time.perf_counter() + 3_600.0
    board, st, undo = fb.from_fen(fb.START_FEN)
    clock()
    insufficient_material(board)
    men_at_most(board, 0, STALEMATE_PIECE_LIMIT)
    count_legal(board, st, undo, BUFS, 0)
    draw_score(STATS, 0)
    repeated(st, PATH, GAME_KEYS, STATS, 1)
    to_table(0, 0)
    from_table(0, 0)
    store(TT, STATS, int(st[7]), 1, EXACT, 0, 0, 0)
    first = fb.legal_moves(board, st, undo)[0]
    move_score(board, first)
    score_moves(board, st, BUFS, SCORES, KILLERS, HISTORY, 0, 1, 0)
    score_captures(board, BUFS, SCORES, 0, 1)
    pick_best(BUFS, SCORES, 0, 0, 1)
    arrays = (TT, BUFS, SCORES, KILLERS, HISTORY, PATH, GAME_KEYS, STATS)
    quiescence(board, st, undo, *arrays, far, QUIESCENCE_MAX_PLY, 0, -INFINITY, INFINITY)
    negamax(board, st, undo, *arrays, far, 2, 1, -INFINITY, INFINITY)
    search_root(board, st, undo, *arrays, far, 2, 0)
    board, st, undo = fb.from_fen("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
    search_root(board, st, undo, *arrays, far, 3, 0)
    reset()


warm()
