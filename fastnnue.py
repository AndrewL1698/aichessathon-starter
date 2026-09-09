"""Runtime inference for the learned evaluation: numba over the integer weights `tools/nnue`
exported. This is the shipped half of the NNUE; nothing here imports torch, ever.

`tools/nnue/nnue_ref.py` is the specification and this file is its reimplementation, so the
arithmetic below is that arithmetic in the same order and the same width:

    acc = l1_bias + sum(l1_weight[i] for each active feature i)   int16   scale qa
    a1  = clip(acc, 0, qa)                                        int16   scale qa
    z2  = l2_bias + a1 . l2_weight                                int32   scale qa*qb
    a2  = clip(z2 // qb, 0, qa)                                   int16   scale qa
    z3  = l3_bias + a2 . l3_weight                                int32   scale qa*qc
    cp  = (z3 * cp_scale) // (qa * qc)                            int32   centipawns

Every division is floor division. numba's `//` on integers has Python's semantics, which
`tests/test_nnue.py` asserts on negative numerators rather than trusting: `z2 // qb` is
clipped to zero straight after so truncation would agree there, but the final divide has no
clip after it and truncating would be a centipawn out on every negative score.

**Two perspectives, one feature scheme.** The scheme in `tools/nnue/features.py` is relative
to the side to move: our men on planes 0-5, theirs on 6-11, and the square flipped vertically
when Black is to move. Written that way the whole feature set changes sign-of-perspective on
every move, which no incremental accumulator can follow. So this keeps *two* accumulators, one
per perspective: index 0 is the White-perspective set (White's men are "ours", squares
unflipped) and index 1 the Black-perspective set (Black's men are "ours", squares flipped).
Neither depends on whose turn it is, so a move changes only the handful of features belonging
to the men that moved, and `evaluate` reads the accumulator of the side to move -- which is
exactly the side-to-move-relative set the net was trained on. `FEATURE` below is that index,
precomputed for every (perspective, piece, mailbox square).

**The stack.** `acc[ply, perspective, hidden]` is int16 and allocated once. `push` copies
`acc[ply]` into `acc[ply + 1]` and applies the deltas for the men the move touched; unmaking
needs no work at all, because `acc[ply]` was never written. A null move moves nothing, so it
is the copy alone. `refresh` builds a perspective pair from the board, and the root does that
once per search; every node below it is incremental.

**Mop-up.** The net is a static evaluation trained on positions with men on the board, and it
has no idea that in KRvK the only thing left to score is geometry: push their king to a corner
and walk ours up. Every move there ties on material and the game is drawn by the fifty-move
rule. So `mop_up` is `fasteval`'s own mop-up term, over the same constants, added on top of
the net's centipawns in exactly the positions `fasteval` fires it in. It reads `fasteval`'s
numbers rather than restating them, so there is one source of truth for the weights.

**Missing or broken weights.** `weights/nnue.npz` arrives by a separate PR and is not in the
tree here. If it is absent, unreadable, or fails the shape and scale checks, `LOADED` is False
and `active()` is False, and the engine runs the hand evaluation exactly as v3.1 does. The
jitted functions are compiled either way, over a zeroed stand-in net of the right dtypes, so
that a weight file appearing does not move any compilation onto the clock.
"""

import time
from pathlib import Path

import numpy as np
from numba import njit
from numba import types as nbt

from fastboard import FLAG_CASTLE, FLAG_EP
from fasteval import (
    CENTRE_DISTANCE,
    FILE_OF,
    MOP_UP_BARE_PIECES,
    MOP_UP_CLOSE,
    MOP_UP_CMD,
    MOP_UP_LOOSE_CLOSE,
    MOP_UP_LOOSE_CMD,
    MOP_UP_MAX_WEAK_PIECES,
    MOP_UP_MIN_ADVANTAGE,
    PHASE_MAX,
    PHASE_QUEEN,
    PHASE_ROOK,
    PIECE_VALUE_BY_KIND,
    RANK_OF,
)

# Spans this module's own compilation, like `fasteval` and `fastsearch` do.
_STARTED = time.perf_counter()

# The weight file the packager ships. Resolved against this file so it is found wherever the
# zip was unpacked, never against the working directory.
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "nnue.npz"

# `tools/nnue/nnue_ref.SCHEME_VERSION`. A weight file from a different feature scheme or a
# different quantisation layout has to be refused, not read with the wrong shapes.
SCHEME_VERSION = 1
# 12 planes of 64 squares, and the second and third layers' widths, all fixed by the scheme.
NUM_FEATURES = 768
LAYER2_WIDTH = 32
LAYER3_WIDTH = 32
# The export proves the int16 accumulator cannot overflow by bounding it with the 32 largest
# weights in each neuron's column; at most 32 men can stand on a board. Checked again here,
# because the file is the thing that ships and the export is the thing that ran once.
MAX_ACTIVE = 32
INT16_MAX = 32767

# A module-level switch so tests and the bench can run either evaluation. `active()` is what
# anything else should ask: this on its own says nothing about whether there are weights.
USE_NNUE = True

# --------------------------------------------------------------------------------------
# The feature index, precomputed. Read-only, so numba holds it as a constant.
#
# FEATURE[perspective, piece, mailbox square] is the row of `l1_weight` that piece on that
# square activates, for that perspective. `fastboard` numbers pieces 1..6 white and 7..12
# black; index 0 (empty) and 13 (off board) are never asked for, and are filled with 0 rather
# than a sentinel because numba does not bounds-check and a plausible wrong answer is a far
# cheaper failure than a read past the end of the weights.
# --------------------------------------------------------------------------------------

FEATURE = np.zeros((2, 14, 120), dtype=np.int16)
for _persp in range(2):
    for _mb in range(21, 99):
        if FILE_OF[_mb] < 0:
            continue
        # `tools/nnue/features.py` indexes squares the way python-chess does, a1 = 0 and
        # h8 = 63, and flips with `square ^ 56` for the Black perspective.
        _square = int(RANK_OF[_mb]) * 8 + int(FILE_OF[_mb])
        if _persp == 1:
            _square ^= 56
        for _piece in range(1, 13):
            _colour = 0 if _piece <= 6 else 1
            _kind = _piece if _piece <= 6 else _piece - 6
            _plane = (0 if _colour == _persp else 6) + _kind - 1
            FEATURE[_persp, _piece, _mb] = _plane * 64 + _square

_BOARD_T = nbt.int8[::1]
_ST_T = nbt.int64[::1]
# acc[ply, perspective, hidden].
ACC_T = nbt.int16[:, :, ::1]

# The whole net as one argument. numba freezes a module-level array into a compiled function
# as a read-only constant, so weights that arrive after import have to be passed in; bundling
# them keeps the search's already-long argument lists readable. `l2_weight` is stored
# transposed to [32, hidden] so the second layer walks contiguous memory.
NET_T = nbt.Tuple(
    (
        nbt.int16[:, ::1],  # l1_weight [768, hidden]
        nbt.int16[::1],  # l1_bias   [hidden]
        nbt.int16[:, ::1],  # l2_weight [32, hidden], transposed
        nbt.int32[::1],  # l2_bias   [32]
        nbt.int16[::1],  # l3_weight [32]
        nbt.int64,  # l3_bias
        nbt.int64,  # qa
        nbt.int64,  # qb
        nbt.int64,  # qc
        nbt.int64,  # cp_scale
    )
)

Net = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int
]


# --------------------------------------------------------------------------------------
# Accumulators.
# --------------------------------------------------------------------------------------


@njit(nbt.void(_BOARD_T, ACC_T, nbt.int64, NET_T), cache=False)
def refresh(board: np.ndarray, acc: np.ndarray, ply: int, net: Net) -> None:
    """Build both perspectives at `ply` from the board. The root does this once per search."""
    l1_weight = net[0]
    l1_bias = net[1]
    hidden = l1_bias.shape[0]
    for perspective in range(2):
        target = acc[ply, perspective]
        for unit in range(hidden):
            target[unit] = l1_bias[unit]
    for square in range(21, 99):
        piece = board[square]
        if piece == 0 or piece == 13:
            continue
        for perspective in range(2):
            row = l1_weight[FEATURE[perspective, piece, square]]
            target = acc[ply, perspective]
            for unit in range(hidden):
                target[unit] += row[unit]


@njit(nbt.void(_BOARD_T, nbt.int64, ACC_T, nbt.int64, nbt.int32, NET_T), cache=False)
def push(
    board: np.ndarray, side: int, acc: np.ndarray, ply: int, move: int, net: Net
) -> None:
    """Copy `acc[ply]` to `acc[ply + 1]` and apply `move`'s deltas, for both perspectives.

    Called with the board as it stands *before* `make_move`, because everything the deltas
    need -- the moving piece, the captured piece, the castling rook -- is readable there and
    nowhere else once the move has been played. `side` is `st[0]` before the move.

    Every intermediate state written here is one a board could really stand in (the same men,
    one of them on a different square, or one fewer), so the export's bound on the int16
    accumulator covers each of them and not merely the final value.
    """
    l1_weight = net[0]
    hidden = net[1].shape[0]
    frm = move & 127
    to = (move >> 7) & 127
    promotion = (move >> 14) & 7
    piece = board[frm]
    # `make_move`'s own arithmetic: a promotion lands as that piece type in this side's range.
    landed = promotion + 6 * side if promotion != 0 else piece

    # Every move has a piece leaving one square and arriving on another, so the copy from
    # `acc[ply]` is fused with that pair rather than run as a pass of its own. The captures
    # and the castling rook below are the rare cases and stay as their own loops.
    for perspective in range(2):
        left = l1_weight[FEATURE[perspective, piece, frm]]
        arrived = l1_weight[FEATURE[perspective, landed, to]]
        source = acc[ply, perspective]
        target = acc[ply + 1, perspective]
        for unit in range(hidden):
            target[unit] = np.int16(source[unit] + arrived[unit] - left[unit])

    if (move & FLAG_EP) != 0:  # the victim is not on the square landed on
        captured_square = to + 10 if side == 0 else to - 10
        captured = board[captured_square]
        for perspective in range(2):
            row = l1_weight[FEATURE[perspective, captured, captured_square]]
            target = acc[ply + 1, perspective]
            for unit in range(hidden):
                target[unit] -= row[unit]
    else:
        captured = board[to]
        if captured != 0:
            for perspective in range(2):
                row = l1_weight[FEATURE[perspective, captured, to]]
                target = acc[ply + 1, perspective]
                for unit in range(hidden):
                    target[unit] -= row[unit]

    if (move & FLAG_CASTLE) != 0:  # the rook moves too
        if to > frm:
            rook_from = frm + 3
            rook_to = frm + 1
        else:
            rook_from = frm - 4
            rook_to = frm - 1
        rook = board[rook_from]
        for perspective in range(2):
            left = l1_weight[FEATURE[perspective, rook, rook_from]]
            arrived = l1_weight[FEATURE[perspective, rook, rook_to]]
            target = acc[ply + 1, perspective]
            for unit in range(hidden):
                target[unit] += arrived[unit] - left[unit]


@njit(nbt.void(ACC_T, nbt.int64, NET_T), cache=False)
def push_null(acc: np.ndarray, ply: int, net: Net) -> None:
    """A null move moves no men, so both perspectives carry over unchanged.

    The side to move changes, but neither accumulator depends on that; which of the two
    `evaluate` reads does, and that is decided at the leaf.
    """
    hidden = net[1].shape[0]
    for perspective in range(2):
        source = acc[ply, perspective]
        target = acc[ply + 1, perspective]
        for unit in range(hidden):
            target[unit] = source[unit]


# --------------------------------------------------------------------------------------
# Inference and the mop-up term.
# --------------------------------------------------------------------------------------


@njit(nbt.int64(ACC_T, nbt.int64, nbt.int64, NET_T), cache=False)
def infer(acc: np.ndarray, ply: int, side: int, net: Net) -> int:
    """The net's centipawns for the position at `ply`, from `side`'s point of view.

    `side` picks the perspective, which is what makes the answer side-to-move relative: the
    accumulator it reads holds exactly the feature set `tools/nnue/features.py` would build
    for this position with this side to move.

    The second layer is walked output by output over the transposed weights so the inner loop
    is contiguous, which also means no scratch array for `z2` and so no allocation per leaf.

    Three details in that loop are worth 4.6x and are not stylistic. The accumulation is
    int32, not int64, because an int16 by int16 product summed into int32 is one SIMD
    instruction and summing into int64 is a widening no vector unit does for free -- the
    values are the same either way, `z2` peaks near 1.4e6 and `nnue_ref` accumulates in int32
    too. The clip is `min`/`max` rather than a pair of `if`s, because a branch in the inner
    loop stops the vectoriser dead. And it is recomputed once per output rather than hoisted
    into a scratch row: measured, hoisting it saved 29 of 610 nanoseconds, which is not worth
    another array threaded through every frame of the search.
    """
    l1_bias = net[1]
    l2_weight = net[2]
    l2_bias = net[3]
    l3_weight = net[4]
    third = net[5]
    qa = net[6]
    qb = net[7]
    qc = net[8]
    cp_scale = net[9]
    hidden = l1_bias.shape[0]
    accumulator = acc[ply, side]
    floor = np.int16(0)
    ceiling = np.int16(qa)

    for index in range(LAYER2_WIDTH):
        row = l2_weight[index]
        total = np.int32(l2_bias[index])
        for unit in range(hidden):
            value = min(max(accumulator[unit], floor), ceiling)
            total += np.int32(value) * np.int32(row[unit])
        second = np.int64(total) // qb
        if second < 0:
            second = 0
        elif second > qa:
            second = qa
        third += second * l3_weight[index]
    return (third * cp_scale) // (qa * qc)


@njit(nbt.boolean(_BOARD_T, nbt.int64, nbt.int64), cache=False)
def _men_at_most(board: np.ndarray, side: int, limit: int) -> bool:
    """Has `side` at most `limit` men? Counted from that side's own end of the board.

    `fastsearch.men_at_most` is this same test, written out again here because `fastsearch`
    imports this module and not the other way round. It is what keeps `mop_up` off the
    middlegame path: the count stops at the limit and the scan starts where that side's men
    are, so a full board answers after a dozen loads instead of walking all 78 squares.
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


@njit(nbt.int64(_BOARD_T, _ST_T), cache=False)
def mop_up(board: np.ndarray, st: np.ndarray) -> int:
    """`fasteval`'s mop-up term alone, tapered and side-to-move relative.

    The net cannot learn this. It is a static evaluation of the men on the board, and in KRvK
    or KQvK every move leaves the same men on the board, so the net ties every move and the
    fifty-move rule ends the game. This is the same geometry `fasteval` scores -- their king
    towards a corner, ours towards theirs -- over the same constants, and it fires in exactly
    the positions `fasteval` fires it in.

    The two scalings that sit around it in `fasteval` are deliberately not reproduced, because
    they cannot both apply: the bare-minor zero and the drawish halving need the material
    advantage inside a minor piece and within `DRAWISH_MARGIN`, and mop-up needs it at
    `MOP_UP_MIN_ADVANTAGE` or more, which is twice that. The two conditions are disjoint.
    """
    if not (
        _men_at_most(board, 0, MOP_UP_MAX_WEAK_PIECES)
        or _men_at_most(board, 1, MOP_UP_MAX_WEAK_PIECES)
    ):
        return 0

    white_men = 0
    black_men = 0
    white_material = 0
    black_material = 0
    # A side's pawns, rooks and queens together: zero is "cannot make progress".
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

    advantage = white_material - black_material
    sign = 0
    weak_king = st[5]
    weak_count = white_men
    weak_heavy = white_heavy
    if advantage >= MOP_UP_MIN_ADVANTAGE:
        sign = 1
        weak_king = st[6]
        weak_count = black_men
        weak_heavy = black_heavy
    elif advantage <= -MOP_UP_MIN_ADVANTAGE:
        sign = -1
    # The gate above only proved one of the two sides is small. If the small one is the side
    # that is ahead, there is nothing to mop up: the other side still has an army.
    if sign == 0 or weak_count > MOP_UP_MAX_WEAK_PIECES:
        return 0
    if weak_count <= MOP_UP_BARE_PIECES and weak_heavy == 0:
        centre_weight = MOP_UP_CMD
        close_weight = MOP_UP_CLOSE
    else:
        centre_weight = MOP_UP_LOOSE_CMD
        close_weight = MOP_UP_LOOSE_CLOSE
    separation = abs(FILE_OF[st[5]] - FILE_OF[st[6]]) + abs(RANK_OF[st[5]] - RANK_OF[st[6]])
    endgame = sign * (
        centre_weight * CENTRE_DISTANCE[weak_king] + close_weight * (14 - separation)
    )

    phase = minors + PHASE_ROOK * rooks + PHASE_QUEEN * queens
    if phase > PHASE_MAX:
        phase = PHASE_MAX
    # Tapered as `fasteval` tapers it: an endgame-only term, truncated toward zero so that
    # mirroring the board negates the score exactly.
    total = endgame * (PHASE_MAX - phase)
    score = total // PHASE_MAX if total >= 0 else -((-total) // PHASE_MAX)
    return score if st[0] == 0 else -score


@njit(nbt.int64(_BOARD_T, _ST_T, ACC_T, nbt.int64, NET_T), cache=False)
def evaluate_nnue(
    board: np.ndarray, st: np.ndarray, acc: np.ndarray, ply: int, net: Net
) -> int:
    """The learned evaluation as the search reads it: the net's centipawns plus mop-up."""
    return infer(acc, ply, st[0], net) + mop_up(board, st)


# --------------------------------------------------------------------------------------
# Loading, and the stand-in net that keeps compilation off the clock either way.
# --------------------------------------------------------------------------------------


class WeightError(Exception):
    """The weight file is not one this runtime can evaluate with."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise WeightError(message)


def _load(path: Path) -> Net:
    """Read and validate `weights/nnue.npz`, or raise `WeightError` saying what is wrong.

    Every shape, dtype and scale is checked rather than assumed. A file from a different
    scheme version, a transposed layer or a bias on the wrong scale would otherwise produce
    an evaluation that looks like a number and plays like noise, and the failure would show up
    as lost games rather than as a load error. The accumulator bound the export proves is
    re-proved here, because the file is what ships.
    """
    with np.load(path) as data:
        missing = {
            "version", "hidden", "qa", "qb", "qc", "cp_scale",
            "l1_weight", "l1_bias", "l2_weight", "l2_bias", "l3_weight", "l3_bias",
        } - set(data.files)
        _check(not missing, f"the weight file is missing {sorted(missing)}")
        version = int(data["version"])
        _check(
            version == SCHEME_VERSION,
            f"the weight file is scheme version {version}, this runtime speaks "
            f"{SCHEME_VERSION}",
        )
        hidden = int(data["hidden"])
        _check(0 < hidden <= 4096, f"a hidden width of {hidden} is not believable")
        qa, qb, qc = int(data["qa"]), int(data["qb"]), int(data["qc"])
        cp_scale = int(data["cp_scale"])
        for name, scale in (("qa", qa), ("qb", qb), ("qc", qc), ("cp_scale", cp_scale)):
            _check(scale > 0, f"{name} is {scale}, which cannot be divided by")
        l1_weight = data["l1_weight"]
        l1_bias = data["l1_bias"]
        l2_weight = data["l2_weight"]
        l2_bias = data["l2_bias"]
        l3_weight = data["l3_weight"]
        l3_bias = int(data["l3_bias"])
        for name, array, shape, dtype in (
            ("l1_weight", l1_weight, (NUM_FEATURES, hidden), np.int16),
            ("l1_bias", l1_bias, (hidden,), np.int16),
            ("l2_weight", l2_weight, (hidden, LAYER2_WIDTH), np.int16),
            ("l2_bias", l2_bias, (LAYER2_WIDTH,), np.int32),
            ("l3_weight", l3_weight, (LAYER3_WIDTH,), np.int16),
        ):
            _check(
                array.shape == shape,
                f"{name} has shape {array.shape}, this runtime wants {shape}",
            )
            _check(
                array.dtype == dtype,
                f"{name} has dtype {array.dtype}, this runtime wants {np.dtype(dtype).name}",
            )

    # The int16 accumulator, proved rather than hoped for: for every hidden neuron the bias
    # plus the 32 largest weight magnitudes in its column has to fit. At most 32 men can stand
    # on a board, so that bound covers every position a search can reach.
    magnitudes = np.abs(l1_weight.astype(np.int32))
    worst = int(
        (np.sort(magnitudes, axis=0)[-MAX_ACTIVE:].sum(axis=0)
         + np.abs(l1_bias.astype(np.int32))).max()
    )
    _check(
        worst <= INT16_MAX,
        f"the int16 accumulator can overflow: the worst hidden neuron reaches {worst} "
        f"against a limit of {INT16_MAX}",
    )
    return (
        np.ascontiguousarray(l1_weight),
        np.ascontiguousarray(l1_bias),
        # Transposed so `infer`'s inner loop is contiguous. Pure layout: the sum it computes
        # is `a1 . l2_weight` either way.
        np.ascontiguousarray(l2_weight.T),
        np.ascontiguousarray(l2_bias),
        np.ascontiguousarray(l3_weight),
        l3_bias,
        qa,
        qb,
        qc,
        cp_scale,
    )


def _stand_in(hidden: int = 128) -> Net:
    """A zeroed net of the right dtypes, so the jitted graph compiles with no weight file."""
    return (
        np.zeros((NUM_FEATURES, hidden), dtype=np.int16),
        np.zeros(hidden, dtype=np.int16),
        np.zeros((LAYER2_WIDTH, hidden), dtype=np.int16),
        np.zeros(LAYER2_WIDTH, dtype=np.int32),
        np.zeros(LAYER3_WIDTH, dtype=np.int16),
        0,
        1024,
        512,
        128,
        400,
    )


LOADED = False
STATUS = ""
try:
    NET = _load(WEIGHTS_PATH)
    LOADED = True
    STATUS = (
        f"nnue h{NET[1].shape[0]} qa{NET[6]} qb{NET[7]} qc{NET[8]} cp{NET[9]} "
        f"from {WEIGHTS_PATH.name}"
    )
except FileNotFoundError:
    NET = _stand_in()
    STATUS = f"hand: no weight file at {WEIGHTS_PATH}"
except (WeightError, OSError, ValueError, KeyError) as _failure:
    NET = _stand_in()
    STATUS = f"hand: {WEIGHTS_PATH.name} rejected ({_failure})"

# The hidden width every accumulator has to be sized for.
HIDDEN = int(NET[1].shape[0])


def active() -> bool:
    """Is the learned evaluation the one that will be used? Both halves have to be true."""
    return USE_NNUE and LOADED


def accumulators(plies: int) -> np.ndarray:
    """The accumulator stack for a search `plies` deep: `acc[ply, perspective, hidden]`."""
    return np.zeros((plies, 2, HIDDEN), dtype=np.int16)


def warm() -> None:
    """Run every jitted function once with the types it will really see.

    The eager signatures above already compile at import; this proves the whole graph runs,
    with or without a weight file, so a failure lands in the init budget rather than on the
    clock. Two plies of a real move are pushed and a promotion, an en passant capture and a
    castle are all reached by the positions below.
    """
    from fastboard import START_FEN, from_fen, legal_moves, make_move, uci_to_move

    acc = accumulators(4)
    board, st, undo = from_fen(START_FEN)
    refresh(board, acc, 0, NET)
    infer(acc, 0, 0, NET)
    mop_up(board, st)
    evaluate_nnue(board, st, acc, 0, NET)
    push_null(acc, 0, NET)
    first = legal_moves(board, st, undo)[0]
    push(board, int(st[0]), acc, 0, first, NET)
    make_move(board, st, undo, first)
    push(board, int(st[0]), acc, 1, legal_moves(board, st, undo)[0], NET)

    # A castle, an en passant capture and a promotion, so every branch of `push` is compiled
    # and run before a game reaches one.
    for fen, uci in (
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("8/8/8/3pP3/8/8/8/K6k w - d6 0 1", "e5d6"),
        ("8/4P3/8/8/8/8/8/K6k w - - 0 1", "e7e8q"),
    ):
        board, st, undo = from_fen(fen)
        refresh(board, acc, 0, NET)
        push(board, int(st[0]), acc, 0, uci_to_move(board, st, undo, uci), NET)
    _men_at_most(board, 0, MOP_UP_MAX_WEAK_PIECES)


warm()

COMPILE_SECONDS = time.perf_counter() - _STARTED
