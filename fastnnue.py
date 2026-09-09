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

**Where the net stops.** `bare_endgame` is the one position class this evaluation refuses, and
`fastsearch.leaf` scores it with the hand tables instead. A network trained on positions games
reach has effectively never seen KRvK or KPvK, and it shows: it scores every legal move in
KRRvK within a few centipawns of every other, because every one of them leaves the same men on
the board. That function has the measurements and the reasoning.

**Missing or broken weights.** `weights/nnue.npz` arrives by a separate PR and is not in the
tree here. If it is absent, unreadable, or fails the shape and scale checks, `LOADED` is False
and `active()` is False, and the engine runs the hand evaluation exactly as v3.1 does. The
jitted functions are compiled either way, over a zeroed stand-in net of the right dtypes, so
that a weight file appearing does not move any compilation onto the clock.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit
from numba import types as nbt

from fastboard import FLAG_CASTLE, FLAG_EP
from fasteval import FILE_OF, MOP_UP_MAX_WEAK_PIECES, RANK_OF

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

# --------------------------------------------------------------------------------------
# The switch, and the four ways a leaf can be scored.
#
# Off by default, and that is now a decision for the orchestrator rather than a verdict from
# the measurements. Blended with the hand evaluation, the 52M-position net is **Elo +191** over
# v3.2 across 64 games with the interval well clear of zero (`docs/BENCH_LOG.md`), so the
# evidence says turn it on; what it waits on is the orchestrator choosing which weight file
# ships and re-running the bench against it. Setting this `True` is the whole change, and with
# a file that declares itself absolute it selects the configuration that was measured.
# --------------------------------------------------------------------------------------

USE_NNUE = False

# How a leaf is scored. `HAND` is `fasteval` alone, and it is also what every position past
# `bare_endgame` gets whatever else is set.
HAND = 0
# The net's centipawns alone. What an absolute net -- one trained to predict the evaluation
# itself -- is for.
ABSOLUTE = 1
# The mean of the two, `(hand + net) // 2`. An experiment rather than a design: if the net is
# noisy but carries signal the hand evaluation lacks, averaging should beat both, and if it
# lands between them the net carries nothing new. `docs/BENCH_LOG.md` has the measurement.
BLEND = 2
# `hand + net`, for a net trained on the *residual* -- Stockfish's centipawns minus
# `fasteval`'s. Such a file says so itself, with `target='residual'` in the npz, because
# scoring a residual net as though it were absolute produces an evaluation that is wrong by
# the whole hand evaluation and still looks like centipawns.
RESIDUAL = 3

POLICY_NAMES = {HAND: "hand", ABSOLUTE: "absolute", BLEND: "blend", RESIDUAL: "residual"}

# What an npz may say it was trained to predict, and how to score it. `tools/nnue/export.py`
# writes `target` as `cp` or `residual` (files exported before the key existed carry none, and
# are absolute); `absolute` is accepted as a synonym for `cp` because that is the word this file
# and the docs use for it and a weight file should not be refused over a vocabulary difference.
# Anything else is refused rather than assumed, in `load`.
TARGETS = {"cp": ABSOLUTE, "absolute": ABSOLUTE, "residual": RESIDUAL}

# How to score an *absolute* net -- one whose file says it predicts the evaluation itself.
# `BLEND`, not `ABSOLUTE`, and that is a measurement rather than a preference: over 64 games at
# 10 s + 0.1 s against v3.2, with the opening book on both sides, the same weight file scored
# **75.0% (Elo +191, interval +109 to +298) blended** against the hand evaluation and **48.4%
# (Elo -11)** alone. `docs/BENCH_LOG.md` has every row. A residual net is never blended: its
# output is a correction that only means anything added whole.
ABSOLUTE_POLICY = BLEND

# Overrides both of the above. `None` means "whatever the file asks for", which is the only
# setting that ships.
POLICY: int | None = None

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
NET_T = nbt.Tuple(  # type: ignore[no-untyped-call]
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


@njit(nbt.int64(nbt.int64, nbt.int64), cache=False)
def divide(numerator: int, denominator: int) -> int:
    """`numerator // denominator` as the jitted code above computes it.

    Exposed only so `tests/test_nnue.py` can assert on negative numerators that numba's
    integer `//` floors, which is what `nnue_ref` does and what the final rescale needs. It is
    not called from anything hot; the divisions in `infer` are written out there.
    """
    return numerator // denominator


@njit(nbt.void(_BOARD_T, ACC_T, nbt.int64, NET_T), cache=False)
def refresh(board: np.ndarray, acc: np.ndarray, ply: int, net: Net) -> None:
    """Build both perspectives at `ply` from the board. The root does this once per search.

    Once per search is also cheap enough to check that the stack is the right width for the
    net, which nothing on the incremental path can afford to. Getting that wrong writes past
    the end of a perspective row, and numba does not bounds-check.
    """
    l1_weight = net[0]
    l1_bias = net[1]
    hidden = l1_bias.shape[0]
    if acc.shape[2] != hidden:
        raise ValueError("fastnnue: the accumulator stack is not this net's hidden width")
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
    `infer` reads does, and that is decided at the leaf.
    """
    hidden = net[1].shape[0]
    for perspective in range(2):
        source = acc[ply, perspective]
        target = acc[ply + 1, perspective]
        for unit in range(hidden):
            target[unit] = source[unit]


# --------------------------------------------------------------------------------------
# Inference, and the one position class it refuses.
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
    instruction and summing into int64 is a widening no vector unit does for free. int32 is
    also the width `nnue_ref` accumulates in: at the shipped scales `z2` peaks under 1e8
    against int32's 2.1e9, so neither overflows. (They would not wrap identically if one did,
    because numba widens the running total to int64; that shows up as a failing parity test,
    not a wrong game.) The clip is `min`/`max` rather than a pair of `if`s, because a
    branch in the inner loop stops the vectoriser dead. And it is recomputed once per output
    rather than hoisted into a scratch row: measured, hoisting it saved 29 of 610 nanoseconds,
    which is not worth another array threaded through every frame of the search.
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
        # `min`/`max` rather than two `if`s, so mypy sees one type here and numba emits the
        # same saturating pair either way. Outside the hot loop, so it costs nothing.
        second = min(max(np.int64(total) // qb, np.int64(0)), np.int64(qa))
        third += second * l3_weight[index]
    return (third * cp_scale) // (qa * qc)


@njit(nbt.boolean(_BOARD_T, nbt.int64, nbt.int64), cache=False)
def _men_at_most(board: np.ndarray, side: int, limit: int) -> bool:
    """Has `side` at most `limit` men? Counted from that side's own end of the board.

    `fastsearch.men_at_most` is this same test, written out again here because `fastsearch`
    imports this module and not the other way round. The count stops at the limit and the
    scan starts where that side's men are, so a full board answers after a dozen loads
    instead of walking all 78 squares, which is what keeps `bare_endgame` free.
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


@njit(nbt.boolean(_BOARD_T), cache=False)
def bare_endgame(board: np.ndarray) -> bool:
    """Has either side been reduced to a king and at most two other men?

    This is the line the learned evaluation is not allowed across, and `fastsearch.leaf`
    scores everything past it with the hand tables instead. Two reasons, one measured:

    The network has no idea what to do here. Its training set is positions a real game
    reached, and KRvK, KRRvK and KPvK are a vanishing fraction of those, so its output in
    them is close to arbitrary -- it scores every legal move in KRRvK within a few centipawns
    of every other, because every one of them leaves the same men on the board. Played out
    with the network scoring the leaves and `fasteval`'s mop-up term added on top, KRRvK drew
    by repetition and KPvK never promoted; `fasteval` alone mates in nine plies from the same
    KRRvK. Adding the mop-up term to the network was the first policy tried and it is not
    enough, because the term is worth at most 120 centipawns and the network's own variation
    across these positions is larger than that.

    And there is nothing to gain. Everything a learned evaluation knows -- pawn structure,
    king safety, piece coordination -- is about positions with men on the board. What decides
    a bare endgame is geometry and the fifty-move clock, which is exactly what `fasteval`'s
    mop-up, drawish scaling and bare-minor zero were written for and tested on.

    `MOP_UP_MAX_WEAK_PIECES` is `fasteval`'s own bound, three men including the king, so the
    line is drawn where the hand evaluation's endgame terms start firing rather than at a
    number of this file's own choosing. The material-advantage half of `fasteval`'s mop-up
    condition is deliberately *not* applied: KPvK is a pawn ahead, not four hundred
    centipawns ahead, and it is one of the positions that needs this.
    """
    return _men_at_most(board, 0, MOP_UP_MAX_WEAK_PIECES) or _men_at_most(
        board, 1, MOP_UP_MAX_WEAK_PIECES
    )


# --------------------------------------------------------------------------------------
# Loading, and the stand-in net that keeps compilation off the clock either way.
# --------------------------------------------------------------------------------------


class WeightError(Exception):
    """The weight file is not one this runtime can evaluate with."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise WeightError(message)


def _open(path: Path) -> Any:
    """`np.load` with every way a damaged file can fail turned into a `WeightError`.

    A truncated copy raises `zipfile.BadZipFile`, an empty one `EOFError`, a corrupt deflate
    stream `zlib.error`, and none of those is an `OSError` or a `ValueError`. Left alone they
    escape the import of this module and the agent never starts, which on the platform is
    every game lost to a file that a `git show` interrupted halfway. A missing file is the
    one exception passed through unchanged, because that is the ordinary case, not damage.
    """
    try:
        return np.load(path)
    except FileNotFoundError:
        raise
    except Exception as failure:
        raise WeightError(f"not a readable npz ({type(failure).__name__}: {failure})") from failure


def _target(path: Path) -> str:
    """What the weight file says it was trained to predict: `absolute` or `residual`.

    A file with no `target` key is absolute. That is not a guess about the future: every file
    exported before the key existed is absolute, and the key was added when the first residual
    net was trained. The string is returned as the file spells it -- `cp` and `absolute` are
    the same thing to `TARGETS` -- so the init log says what the file actually said.
    """
    with _open(path) as data:
        if "target" not in data.files:
            return "absolute"
        return str(data["target"])


def _load(path: Path) -> Net:
    """Read and validate `weights/nnue.npz`, or raise `WeightError` saying what is wrong.

    Every shape, dtype and scale is checked rather than assumed. A file from a different
    scheme version, a transposed layer or a bias on the wrong scale would otherwise produce
    an evaluation that looks like a number and plays like noise, and the failure would show up
    as lost games rather than as a load error. The accumulator bound the export proves is
    re-proved here, because the file is what ships.
    """
    with _open(path) as data:
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
        # The clip ceiling is held as an int16, so a qa past that would wrap negative and
        # `min(max(v, 0), ceiling)` would return nonsense where `nnue_ref` clips correctly.
        _check(qa <= 32767, f"qa is {qa}, past the int16 ceiling the clip is held in")
        l1_weight = data["l1_weight"]
        l1_bias = data["l1_bias"]
        l2_weight = data["l2_weight"]
        l2_bias = data["l2_bias"]
        l3_weight = data["l3_weight"]
        l3_bias = int(data["l3_bias"])
        # Refused rather than defaulted: a marker this runtime does not understand means the
        # file was trained against something it cannot compose correctly, and scoring it the
        # wrong way round is an evaluation wrong by the whole hand evaluation.
        if "target" in data.files:
            marker = str(data["target"])
            _check(
                marker in TARGETS,
                f"the weight file says target={marker!r}, which this runtime does not know "
                f"how to score; it understands {sorted(TARGETS)}",
            )
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


def _guarded(read: Any, path: Path) -> Any:
    """Run one of the readers above with every failure but a missing file as a `WeightError`.

    `_open` covers the container, but a corrupt compressed stream only fails when the array
    is read, inside the reader, as `zlib.error`; this is the same conversion one level up.
    """
    try:
        return read(path)
    except (WeightError, FileNotFoundError):
        raise
    except Exception as failure:
        raise WeightError(f"unreadable ({type(failure).__name__}: {failure})") from failure


def target(path: Path) -> str:
    """What the file says it was trained to predict; see `_target`."""
    return str(_guarded(_target, path))


def load(path: Path) -> Net:
    """The validated network from `path`; see `_load`. Raises `WeightError` on any damage."""
    net: Net = _guarded(_load, path)
    return net


LOADED = False
FILE_TARGET = "absolute"
STATUS = ""
try:
    NET = load(WEIGHTS_PATH)
    FILE_TARGET = target(WEIGHTS_PATH)
    LOADED = True
    STATUS = (
        f"nnue h{NET[1].shape[0]} qa{NET[6]} qb{NET[7]} qc{NET[8]} cp{NET[9]} "
        f"{FILE_TARGET} from {WEIGHTS_PATH.name}"
    )
except FileNotFoundError:
    NET = _stand_in()
    STATUS = f"hand: no weight file at {WEIGHTS_PATH}"
except Exception as _failure:
    NET = _stand_in()
    STATUS = f"hand: {WEIGHTS_PATH.name} rejected ({_failure})"

# The hidden width every accumulator has to be sized for.
HIDDEN = int(NET[1].shape[0])


def active() -> bool:
    """Is the learned evaluation the one that will be used? Both halves have to be true."""
    return USE_NNUE and LOADED


def file_policy() -> int:
    """How to score a leaf when the net is on: what the file asks for, or the override.

    A residual file is scored as a residual and nothing else. An absolute one is scored the way
    `ABSOLUTE_POLICY` says, which is blended, because that is what measured 200 Elo better than
    using it alone.
    """
    if POLICY is not None:
        return POLICY
    if TARGETS[FILE_TARGET] == RESIDUAL:
        return RESIDUAL
    return ABSOLUTE_POLICY


def policy() -> int:
    """How a leaf will actually be scored. `HAND` when there is no net or the switch is off."""
    return file_policy() if active() else HAND


def accumulators(plies: int, hidden: int = HIDDEN) -> np.ndarray:
    """The accumulator stack for a search `plies` deep: `acc[ply, perspective, hidden]`.

    `hidden` defaults to the width of the net that loaded, which is what the search wants. It
    is an argument because a caller can hold a net this module did not load -- the tests
    evaluate every weight file in `weights/` -- and a stack sized for the wrong width is not a
    wrong answer, it is a write past the end of a row into the next perspective's memory.
    `refresh` refuses that outright rather than leaving it to be found as a wrong evaluation.
    """
    return np.zeros((plies, 2, hidden), dtype=np.int16)


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
    bare_endgame(board)
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
