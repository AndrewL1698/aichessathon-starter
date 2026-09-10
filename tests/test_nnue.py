"""Prove the numba runtime evaluates the network `tools/nnue` exported, exactly.

`uv run python -m tests.test_nnue [--full]`.

Two claims carry this file, and neither is a tolerance.

The first is that `fastnnue` and `tools/nnue/nnue_ref.py` return the *same integer* for the
same position. `nnue_ref` is the executable specification the export was verified against, so
anything short of equality means the runtime is evaluating a different network from the one
that was trained and checked: a scale read from the wrong place, a layer transposed, an
accumulator that wraps, a truncating divide where the specification floors. Every one of those
produces plausible centipawns and unplayable chess.

The second is that the incremental accumulators equal the from-scratch ones at every ply. The
search maintains `acc[ply]` by applying deltas as it makes moves, and a delta that is wrong
for, say, an en passant capture leaves an accumulator that is wrong for the whole subtree below
it and correct again the moment the move is unmade, which no game-level test would ever isolate.
So this drives the same calls the search makes, in the same order, and rebuilds the answer from
the board at every single ply.

The rest is what a correct evaluation still loses games by doing: costing so much per node that
the search gives up two plies, taking so long to import that the platform's budget is spent, or
turning a won KRRvK into a repetition draw because the network has never seen one.
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chess
import numpy as np

import fastboard as fb
import fasteval as fe
import fastnnue as fn
import fastsearch as fs
import tests.test_fastsearch as tfs
from tests.test_fastboard import STRESS_SEEDS, positions
from tests.test_fasteval import ENDGAME_SEEDS, load_reference
from tools.nnue import features, nnue_ref

ROOT = Path(__file__).resolve().parent.parent

# The endings past `fastnnue.bare_endgame`, where the hand tables score the leaf. Every legal
# move in each of these leaves the same men on the board, so a network trained on positions
# games reach ties all of them and the fifty-move rule ends the game.
BARE_ENDGAMES: tuple[tuple[str, str], ...] = (
    ("KRvK", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1"),
    ("KQvK", "8/8/8/3k4/8/8/8/3KQ3 w - - 0 1"),
    ("KRRvK", "8/8/8/4k3/8/8/8/R3K2R w - - 0 1"),
    ("KPvK", "8/4k3/8/8/8/8/4P3/4K3 w - - 0 1"),
    ("KBBvK", "8/8/8/4k3/8/8/8/2B1KB2 w - - 0 1"),
)

# Rated round 102, where the handover cost 47 moves. Both are one capture short of
# `fastnnue.bare_endgame`, so the move that wins crosses the line; `check_round_102` says what
# v4.2 played instead and why.
ROUND_102: tuple[tuple[str, str, str], ...] = (
    ("m54 Kxd3", "8/8/8/3B4/3p1p2/2kP1P2/7r/3K4 b - - 2 54", "c3d3"),
    ("m92 Kxd3", "8/8/4B3/2r5/3p1p2/3PkP2/8/6K1 b - - 78 92", "e3d3"),
)

# Rules the feature scheme and the deltas only meet in particular positions. Random play from
# the openings reaches an en passant capture roughly never and a promotion rarely.
DELTA_SEEDS: tuple[str, ...] = (
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # castling, all four ways
    "8/2P1P3/8/8/8/8/2p1p3/K6k w - - 0 1",  # promotions, both colours, all four pieces
    "8/2P1P3/8/8/8/8/2p1p1n1/K5bk w - - 0 1",  # promotions that are also captures
    "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",  # a live en passant
    "8/8/8/2pP4/8/8/8/K6k w - c6 0 1",  # en passant with nothing else on the board
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",  # a pawn ending that promotes
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",  # castling with a full board
)

# The end of the pipeline in one number. The pawn and rook figures landing near 100 and 500
# is every stage of it agreeing: features, perspective, sign, target transform, read-back
# scale, quantisation and this runtime.
SANITY: tuple[tuple[str, str], ...] = (
    ("start", fb.START_FEN),
    ("+1 pawn (white)", "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("+1 rook (white)", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"),
)


class Failure(Exception):
    """The runtime disagreed with the network it is meant to be evaluating."""


def weight_files(given: list[str]) -> list[Path]:
    """Every weight file to check. All of `weights/nnue*.npz` unless told otherwise.

    More than one hidden width ships over the life of this branch, and a runtime that only
    works at 128 is a runtime that fails on the day someone trains a wider net. Nothing here
    reads a shape it did not take from the file.
    """
    if given:
        return [Path(path) for path in given]
    found = sorted((ROOT / "weights").glob("nnue*.npz"))
    if not found:
        raise Failure(
            f"no weight file to test against: put one at {ROOT / 'weights' / 'nnue.npz'} "
            f"or pass --weights"
        )
    return found


def check_floor_division() -> str:
    """The final rescale floors, on negative numerators, exactly as Python does.

    `nnue_ref` divides with numpy and Python integers, both of which floor. C truncates toward
    zero, and if numba did too, every negative evaluation would come back one centipawn high:
    small enough to pass any tolerance and large enough to break the equality this file is
    built on. So it is asserted rather than assumed, here and once for every scale a shipped
    weight file actually uses.
    """
    numerators = [-1, -7, -8, -511, -512, -513, -131_071, -131_072, -131_073, 1, 7, 8, 131_073]
    denominators = [512, 511, 131_072, 65_536, 3]
    checked = 0
    for numerator in numerators:
        for denominator in denominators:
            got = int(fn.divide(numerator, denominator))
            want = numerator // denominator
            if got != want:
                raise Failure(
                    f"numba computes {numerator} // {denominator} as {got}, Python says {want}"
                )
            checked += 1
    return f"{checked} divisions, negative numerators included, all floor"


def check_load(path: Path) -> str:
    """A weight file is read for its own shapes and scales, never for assumed ones."""
    net = fn.load(path)
    with np.load(path) as data:
        for index, name in ((6, "qa"), (7, "qb"), (8, "qc"), (9, "cp_scale")):
            if net[index] != int(data[name]):
                raise Failure(f"{path.name}: {name} loaded as {net[index]}, file says {data[name]}")
        hidden = int(data["hidden"])
    if net[1].shape[0] != hidden:
        raise Failure(f"{path.name}: loaded hidden {net[1].shape[0]}, file says {hidden}")
    return f"h{hidden} qa{net[6]} qb{net[7]} qc{net[8]} cp{net[9]}"


def check_rejection() -> str:
    """A weight file this runtime cannot evaluate has to be refused, not read hopefully.

    The engine falls back to the hand evaluation when loading fails, which is a lost few Elo.
    Reading a wrong-shaped file as if it were right is a lost game, so every check is here and
    each one is proven to fire rather than assumed to.
    """
    source = fn.load(ROOT / "weights" / "nnue.npz")
    hidden = source[1].shape[0]
    base: dict[str, Any] = {
        "version": np.int32(fn.SCHEME_VERSION),
        "hidden": np.int32(hidden),
        "qa": np.int32(source[6]),
        "qb": np.int32(source[7]),
        "qc": np.int32(source[8]),
        "cp_scale": np.int32(source[9]),
        "l1_weight": source[0],
        "l1_bias": source[1],
        # `load` transposes layer two, so the round trip has to transpose it back.
        "l2_weight": np.ascontiguousarray(source[2].T),
        "l2_bias": source[3],
        "l3_weight": source[4],
        "l3_bias": np.int32(source[5]),
    }
    corruptions: tuple[tuple[str, dict[str, Any]], ...] = (
        ("a future scheme version", {"version": np.int32(fn.SCHEME_VERSION + 1)}),
        ("a hidden size the arrays contradict", {"hidden": np.int32(hidden + 1)}),
        ("a zero scale", {"qb": np.int32(0)}),
        ("a transposed first layer", {"l1_weight": np.ascontiguousarray(source[0].T[:, :1])}),
        ("a bias at the wrong width", {"l2_bias": np.zeros(1, dtype=np.int32)}),
        ("a bias at the wrong dtype", {"l3_weight": source[4].astype(np.int32)}),
        # The int16 accumulator proof: 32 men at the largest weights must still fit.
        ("an accumulator that can overflow", {"l1_weight": np.full_like(source[0], 4000)}),
    )
    workspace = Path(fs.__file__).parent / ".test_nnue_rejects"
    workspace.mkdir(exist_ok=True)
    try:
        for label, override in corruptions:
            path = workspace / "corrupt.npz"
            np.savez_compressed(path, **(base | override))
            try:
                fn.load(path)
            except fn.WeightError:
                continue
            except Exception as unexpected:
                raise Failure(
                    f"{label} raised {unexpected!r} rather than a WeightError"
                ) from unexpected
            raise Failure(f"{label} was accepted")
        # A damaged container, not merely wrong contents: a copy cut off halfway, an empty
        # file, and a corrupt compressed stream. Each raises something that is not an OSError
        # or a ValueError out of `np.load`, and each has to come back as a WeightError so the
        # import falls back to the hand evaluation instead of never finishing.
        intact = workspace / "intact.npz"
        np.savez_compressed(intact, **base)
        whole = intact.read_bytes()
        damaged: tuple[tuple[str, bytes], ...] = (
            ("a file cut off halfway", whole[: len(whole) // 2]),
            ("a file cut off after 100 bytes", whole[:100]),
            ("an empty file", b""),
            ("a corrupt compressed stream", whole[:60] + bytes(64) + whole[124:]),
        )
        for label, content in damaged:
            path = workspace / "damaged.npz"
            path.write_bytes(content)
            try:
                fn.load(path)
            except fn.WeightError:
                continue
            except Exception as unexpected:
                raise Failure(
                    f"{label} raised {unexpected!r} rather than a WeightError"
                ) from unexpected
            raise Failure(f"{label} was accepted")
        # A missing file is the ordinary case, not a corruption, and it must not raise here.
        missing = workspace / "absent.npz"
        try:
            fn.load(missing)
        except FileNotFoundError:
            pass
        else:
            raise Failure("a missing weight file did not raise FileNotFoundError")
    finally:
        for leftover in workspace.glob("*"):
            leftover.unlink()
        workspace.rmdir()
    return f"{len(corruptions)} broken weight files all refused, a missing one reported as missing"


def check_against_reference(path: Path, fens: list[str]) -> dict[str, int]:
    """`fastnnue` must return the integer `nnue_ref` returns, on every position.

    The reference is driven off python-chess and `tools/nnue/features.py`, so this compares two
    entirely separate routes to the feature set as well as two routes to the arithmetic: a
    mailbox scan with a precomputed index table here, `board.piece_map()` and the index formula
    there. A perspective flip that is wrong one way round shows up as half the sample failing.
    """
    reference = nnue_ref.load(path)
    net = fn.load(path)
    acc = fn.accumulators(2, net[1].shape[0])
    tally = {"positions": 0, "black to move": 0, "en passant": 0, "promotions": 0, "endgames": 0}
    for fen in fens:
        board, st, _ = fb.from_fen(fen)
        fn.refresh(board, acc, 0, net)
        got = int(fn.infer(acc, 0, int(st[0]), net))
        reference_board = chess.Board(fen)
        want = int(nnue_ref.evaluate(reference, features.features(reference_board)))
        if got != want:
            raise Failure(
                f"{path.name}: fastnnue evaluates {fen!r} at {got}cp, nnue_ref says {want}cp"
            )
        tally["positions"] += 1
        tally["black to move"] += reference_board.turn == chess.BLACK
        tally["en passant"] += reference_board.has_legal_en_passant()
        tally["promotions"] += any(move.promotion for move in reference_board.legal_moves)
        tally["endgames"] += len(reference_board.piece_map()) <= 10
    return tally


def check_increments(path: Path, rng: random.Random, wanted: int) -> dict[str, int]:
    """`acc[ply]` after `push` has to equal a from-scratch build, at every ply.

    Driven through the calls the search makes, in the order it makes them: `push` before
    `make_move`, `push_null` before `make_null`, nothing at all on the way back out. That last
    one is the property the whole design rests on -- unmaking is free because `acc[ply]` was
    never written -- so it is checked too, by snapshotting the parent before the move and
    comparing after the unmake.

    Both perspectives are compared in full. Comparing only the side to move's would pass with
    the other one wrong for a whole subtree, and the other one is the one the reply reads.
    """
    net = fn.load(path)
    hidden = net[1].shape[0]
    acc = fn.accumulators(fs.MAX_SEARCH_PLY + 1, hidden)
    scratch = fn.accumulators(1, hidden)
    tally = {"plies": 0, "captures": 0, "en passant": 0, "promotions": 0, "castles": 0, "nulls": 0}
    seeds = [fb.START_FEN, *DELTA_SEEDS, *STRESS_SEEDS, *ENDGAME_SEEDS]

    # Depth-first with a branching factor of three fills ten thousand plies out of a single
    # seed if it is allowed to run to the search's real depth, and one seed is one material
    # distribution: the first version of this test reached two captures and no promotion in
    # ten thousand plies. So the descent is shallow and the seeds are many.
    depth_limit = 8
    per_seed = max(wanted // 40, 40)

    def descend(board: np.ndarray, st: np.ndarray, undo: np.ndarray, ply: int, passes: int) -> None:
        if tally["plies"] >= budget or ply >= depth_limit:
            return
        moves = fb.legal_moves(board, st, undo)
        if not moves:
            return
        # A pass, sometimes, and never two running: that is the search's own rule, and the
        # accumulator has to survive the ply parity it leaves behind either way.
        if passes == 0 and rng.random() < 0.10:
            parent = acc[ply].copy()
            fn.push_null(acc, ply, net)
            fs.make_null(st, undo)
            fn.refresh(board, scratch, 0, net)
            if not np.array_equal(acc[ply + 1], scratch[0]):
                raise Failure(f"a null move at ply {ply} left the accumulator wrong")
            tally["plies"] += 1
            tally["nulls"] += 1
            descend(board, st, undo, ply + 1, 1)
            fs.unmake_null(st, undo)
            if not np.array_equal(acc[ply], parent):
                raise Failure(f"unmaking a null move at ply {ply} did not leave acc[{ply}] alone")
            return

        for move in rng.sample(moves, min(len(moves), 3)):
            if tally["plies"] >= budget:
                return
            side = int(st[0])
            target = board[(move >> 7) & 127]
            parent = acc[ply].copy()
            fn.push(board, side, acc, ply, move, net)
            fb.make_move(board, st, undo, move)
            fn.refresh(board, scratch, 0, net)
            if not np.array_equal(acc[ply + 1], scratch[0]):
                for perspective in range(2):
                    if np.array_equal(acc[ply + 1, perspective], scratch[0, perspective]):
                        continue
                    unit = int(np.argmax(acc[ply + 1, perspective] != scratch[0, perspective]))
                    raise Failure(
                        f"{fb.move_to_uci(move)} at ply {ply} in {fb.to_fen(board, st)!r}: "
                        f"perspective {perspective} unit {unit} of {hidden} is "
                        f"{acc[ply + 1, perspective, unit]}, from scratch it is "
                        f"{scratch[0, perspective, unit]}"
                    )
            tally["plies"] += 1
            tally["captures"] += target != 0
            tally["en passant"] += (move & fb.FLAG_EP) != 0
            tally["promotions"] += ((move >> 14) & 7) != 0
            tally["castles"] += (move & fb.FLAG_CASTLE) != 0
            descend(board, st, undo, ply + 1, 0)
            fb.unmake_move(board, st, undo, move)
            if not np.array_equal(acc[ply], parent):
                raise Failure(
                    f"unmaking {fb.move_to_uci(move)} at ply {ply} did not leave acc[{ply}] alone"
                )

    while tally["plies"] < wanted:
        budget = min(wanted, tally["plies"] + per_seed)
        board, st, undo = fb.from_fen(rng.choice(seeds))
        # Walk a few plies in before descending, so the sample is not all seed positions.
        for _ in range(rng.randint(0, 16)):
            moves = fb.legal_moves(board, st, undo)
            if not moves:
                break
            fb.make_move(board, st, undo, rng.choice(moves))
        fn.refresh(board, acc, 0, net)
        descend(board, st, undo, 0, 0)
    return tally


def check_policies(fens: list[str]) -> str:
    """Each of `leaf`'s four ways of scoring a leaf composes the two evaluations as claimed.

    Composition is the whole content of the policies, and getting it wrong is not a crash: an
    absolute net scored as a residual one is out by the entire hand evaluation and still reads
    like centipawns. So each is checked against the arithmetic done in Python, over positions
    that include the bare endgames -- because the handover overrides every policy and that has
    to be true of each of them, not just of the one that shipped first.

    Past the handover the composition is the blend plus a whole mop-up term, and the half of
    that term the blend already carries through its hand half is why the arithmetic here adds
    only the other half. Two positions past it are still the hand evaluation whole: a pawn
    ending, where the net is reversed rather than merely unsure, and a position the hand
    evaluation scores exactly 0, which is a draw it has proved from the material and not
    something to average with a network that has never seen the position.

    `//` on the blend is floor division. Both inputs are side-to-move relative and both are
    mirror-invariant, so their mean is too, and the search never needs the evaluation to be an
    odd function; there is nothing for a floor to break here.
    """
    acc = fn.accumulators(1)
    tally = {"positions": 0, "bare": 0}
    for fen in fens:
        board, st, _ = fb.from_fen(fen)
        hand = int(fe.evaluate(board, st))
        bare = bool(fn.bare_endgame(board))
        fn.refresh(board, acc, 0, fn.NET)
        net = int(fn.infer(acc, 0, int(st[0]), fn.NET))
        term = int(fe.mop_up(board, st))
        half = term // 2 if term >= 0 else -((-term) // 2)
        whole = hand == 0 or fn.pawns_only(board)
        past = hand if whole else (hand + net) // 2 + half
        wanted = {
            fn.HAND: hand,
            fn.ABSOLUTE: past if bare else net,
            fn.BLEND: past if bare else (hand + net) // 2,
            fn.RESIDUAL: past if bare else hand + net,
        }
        for chosen, want in wanted.items():
            fs.STATS[fs.NNUE_POLICY] = chosen
            fs.refresh_root(board)
            fn.refresh(board, fs.ACC, 0, fn.NET)
            got = int(fs.leaf(board, st, fs.ACC, 0, fs.STATS, fn.NET))
            if got != want:
                raise Failure(
                    f"policy {fn.POLICY_NAMES[chosen]} scores {fen!r} at {got}, the "
                    f"composition of hand {hand} and net {net} is {want}"
                )
        tally["positions"] += 1
        tally["bare"] += bare
    fs.STATS[fs.NNUE_POLICY] = fn.policy()
    return (
        f"{len(fn.POLICY_NAMES)} policies over {tally['positions']:,} positions "
        f"({tally['bare']} of them past the bare-endgame handover), all compose exactly"
    )


def check_target_marker() -> str:
    """A weight file says what it was trained to predict, and the loader believes only two.

    A residual net scored as an absolute one is wrong by the whole hand evaluation, so the
    marker is not something to infer from the numbers. This builds the files rather than
    waiting for one: an absolute file with no key at all (every file exported before the key
    existed), one that says so explicitly, a residual one, and one with a marker from the
    future, which has to be refused rather than defaulted.
    """
    source = fn.load(ROOT / "weights" / "nnue.npz")
    hidden = source[1].shape[0]
    base: dict[str, Any] = {
        "version": np.int32(fn.SCHEME_VERSION),
        "hidden": np.int32(hidden),
        "qa": np.int32(source[6]),
        "qb": np.int32(source[7]),
        "qc": np.int32(source[8]),
        "cp_scale": np.int32(source[9]),
        "l1_weight": source[0],
        "l1_bias": source[1],
        "l2_weight": np.ascontiguousarray(source[2].T),
        "l2_bias": source[3],
        "l3_weight": source[4],
        "l3_bias": np.int32(source[5]),
    }
    workspace = ROOT / ".test_nnue_targets"
    workspace.mkdir(exist_ok=True)
    checked = []
    try:
        for label, extra, want, policy in (
            ("no key", {}, "absolute", fn.ABSOLUTE),
            # `cp` is what `tools/nnue/train.py --target` actually writes for an absolute net.
            ("cp", {"target": np.str_("cp")}, "cp", fn.ABSOLUTE),
            ("absolute", {"target": np.str_("absolute")}, "absolute", fn.ABSOLUTE),
            ("residual", {"target": np.str_("residual")}, "residual", fn.RESIDUAL),
        ):
            path = workspace / "marked.npz"
            np.savez_compressed(path, **(base | extra))
            fn.load(path)  # a legal marker must not be refused
            got = fn.target(path)
            if got != want:
                raise Failure(f"a {label} file reads as target={got!r}, want {want!r}")
            if fn.TARGETS[got] != policy:
                raise Failure(
                    f"a {label} file maps to policy "
                    f"{fn.POLICY_NAMES[fn.TARGETS[got]]}, want {fn.POLICY_NAMES[policy]}"
                )
            checked.append(f"{label} -> {got} -> {fn.POLICY_NAMES[policy]}")
            # And what the engine would actually do with such a file: a residual is scored
            # whole, an absolute one is blended, because blending measured 200 Elo better.
            wanted = fn.RESIDUAL if policy == fn.RESIDUAL else fn.ABSOLUTE_POLICY
            if fn.TARGETS[got] == fn.RESIDUAL:
                if wanted != fn.RESIDUAL:
                    raise Failure("a residual file must never be scored any other way")
            elif wanted != fn.ABSOLUTE_POLICY:
                raise Failure(f"an absolute file should be scored {fn.POLICY_NAMES[wanted]}")
        path = workspace / "marked.npz"
        np.savez_compressed(path, **(base | {"target": np.str_("wdl")}))
        try:
            fn.load(path)
        except fn.WeightError:
            checked.append("unknown marker refused")
        else:
            raise Failure("a file with an unknown target marker was accepted")
    finally:
        for leftover in workspace.glob("*"):
            leftover.unlink()
        workspace.rmdir()
    return "; ".join(checked)


def check_bare_endgames(depth: int) -> str:
    """Past `fastnnue.bare_endgame` the leaf is the blend plus a whole mop-up term, and these
    still convert.

    Two assertions, one per evaluation. Each of these endings has to reach mate with the
    network switched on, which is what the handover exists for: before it, KRRvK drew by
    repetition and KPvK never promoted, because a network trained on positions games reach
    scores every move in KRRvK the same. And each has to reach mate with the network off,
    because that is the hand policy `agent.py` still runs on and the conversion is the hand
    tables' own.

    A position with a piece on the board is no longer required to play out *identically* with
    the network on and off, and that is the change round 102 forced: scoring these positions
    with the hand tables alone put a step in the evaluation at the handover, and the engine
    spent 47 moves refusing to capture across it. `fastnnue.bare_endgame` has the numbers.
    The two evaluations now differ past the line, so the two playouts may pick different
    mates; what is asserted is that both mate, and `check_policies` above pins the
    composition exactly.

    A *pawn* ending keeps v4.2's handover whole, and what that buys is checked one level down
    rather than on the playout: a search whose leaves are all pawn endings has to return the
    same move and the same score with the network on and off, because every one of those
    leaves is the hand evaluation either way. The playout itself is not identical and should
    not be -- the pawn promotes, and the KQvK it promotes into is a piece ending scored by the
    blend, which is the seam `fastnnue.bare_endgame` says is deliberately left there.

    Fixed depth rather than a clock, because a conversion that depends on how loaded the
    machine was is not a test. `contempt_for` is recomputed each ply, as `think` does, so the
    engine still refuses a draw it is winning.
    """
    middlegame, _, _ = fb.from_fen(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    if fn.bare_endgame(middlegame):
        raise Failure("bare_endgame fired in a middlegame, so the network never gets used")
    was = fn.USE_NNUE
    results = []
    try:
        fn.USE_NNUE = True
        if not fn.active():
            raise Failure("the network is not active, so this measures the wrong evaluation")
        for name, fen in BARE_ENDGAMES:
            board, _, _ = fb.from_fen(fen)
            if not fn.bare_endgame(board):
                raise Failure(f"bare_endgame is false for {name}, which is one of its cases")
            with_net = playout(fen, depth, True)
            without = playout(fen, depth, False)
            if not with_net.startswith("mate in "):
                raise Failure(
                    f"with the network on, {name} ended in {with_net}: nothing is steering "
                    f"the search towards a mate"
                )
            if not without.startswith("mate in "):
                raise Failure(
                    f"with the network off, {name} ended in {without}: the hand policy "
                    f"`agent.py` runs on no longer converts it"
                )
            if fn.pawns_only(board):
                # Deep enough to be a real search and shallow enough that no line in it
                # reaches the eighth rank, so every leaf is still a pawn ending and the two
                # evaluations are the same evaluation. A single leaf scored with the network
                # in it moves the score.
                fs.reset()
                on = fs.search_fixed(fen, depth, nnue=True)[:2]
                fs.reset()
                off = fs.search_fixed(fen, depth, nnue=False)[:2]
                if on != off:
                    raise Failure(
                        f"{name} is a pawn ending, where the network is handed over to the "
                        f"hand tables whole, but at depth {depth} the search plays {on} with "
                        f"it and {off} without it"
                    )
            results.append(f"{name} {with_net} (hand alone: {without})")
    finally:
        fn.USE_NNUE = was
        fs.reset()
    return f"false in a middlegame; at depth {depth}, {', '.join(results)}"


def check_round_102(depth: int) -> str:
    """The capture across the handover is played, which in rated round 102 it was not.

    A rook up in a won ending, v4.2 shuffled from move 53 to move 100 -- 47 moves, halfmove
    clock 94 -- rather than take the d3 pawn with its king, because the position it was in was
    scored by the blend and the position after the capture, past `fastnnue.bare_endgame`, was
    scored by the hand tables alone. Two evaluations on two scales, so the capture read as a
    450-centipawn loss: at these depths v4.2 answers Rd2 and Rh3 at +869 and +857 from move
    54, and c5c2 at +978 from move 92. Stockfish at depth 30 mates in 14 from move 54.

    These are the two positions from that game, and they are here rather than only in
    `positions.epd` because what they test is the seam itself: any policy that scores the two
    sides of `bare_endgame` on different scales fails them, however good either scale is.
    """
    was = fn.USE_NNUE
    found = []
    try:
        fn.USE_NNUE = True
        if not fn.active():
            raise Failure("the network is not active, so this measures the wrong evaluation")
        for label, fen, want in ROUND_102:
            fs.reset()
            move, score, _ = fs.search_fixed(fen, depth)
            if move != want:
                raise Failure(
                    f"round 102 {label}: at depth {depth} the search plays {move} at {score}, "
                    f"not the capture {want} -- the evaluation has a step at bare_endgame "
                    f"again and the engine will shuffle rather than cross it"
                )
            found.append(f"{label} {move} {int(score):+d}")
    finally:
        fn.USE_NNUE = was
        fs.reset()
    return f"at depth {depth}, {', '.join(found)}"


def playout(fen: str, depth: int, enabled: bool, limit: int = 120) -> str:
    """Play a position out against itself at a fixed depth and say how it ended."""
    fs.reset()
    board = chess.Board(fen)
    for ply in range(1, limit + 1):
        position, st, _ = fb.from_fen(board.fen())
        contempt = fs.root_contempt(position, st)
        uci, _, _ = fs.search_fixed(
            board.fen(), depth, fresh=False, contempt=contempt, nnue=enabled
        )
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise Failure(f"playing out {fen!r}, {uci} is not legal in {board.fen()!r}")
        board.push(move)
        if board.is_checkmate():
            return f"mate in {ply} plies"
        if board.is_game_over():
            outcome = board.outcome()
            return outcome.termination.name.lower() if outcome else "over"
    return f"no mate in {limit} plies"


def check_sanity(path: Path) -> str:
    """The centipawns the whole pipeline agrees on, printed rather than asserted.

    A pawn near 100 and a rook near 500 is every stage agreeing with every other. It is a
    property of the net that was trained, though, not of this runtime, so it is reported and
    the exact-match test above is what fails a bad build.
    """
    net = fn.load(path)
    acc = fn.accumulators(1, net[1].shape[0])
    parts = []
    for label, fen in SANITY:
        board, st, _ = fb.from_fen(fen)
        fn.refresh(board, acc, 0, net)
        parts.append(f"{label} {int(fn.infer(acc, 0, int(st[0]), net)):+d}cp")
    return "  |  ".join(parts)


def check_search(reference_module: ModuleType, fens: list[str]) -> str:
    """The search still obeys the rules and the clock with the network scoring its leaves.

    These are `tests/test_fastsearch.py`'s own checks, run again with the evaluation swapped.
    The equality-to-`agent.py` one is run with the network *off*, where it still has to hold
    exactly: that is what says this branch changed the evaluation and nothing else.
    """
    was = fn.USE_NNUE
    try:
        fn.USE_NNUE = True
        if not fn.active():
            raise Failure("the network is not active, so none of this measures it")
        legal = tfs.check_legal(fens[:20], 5)
        timed = check_timed(fens[:18])
        backstop = tfs.check_backstop(fens[:4])

        fn.USE_NNUE = False
        equality = tfs.check_against_reference(reference_module, fens[:10], (2, 3, 4))
        off_timed = tfs.check_timed(fens[:12], reference_module)
    finally:
        fn.USE_NNUE = was
        fs.reset()
    return (
        f"network on: {legal} fixed-depth searches all legal; {timed}; {backstop}\n"
        f"    network off: {equality['searches']} searches score exactly what agent.py scores, "
        f"and {off_timed.split(';')[0]}"
    )


def check_timed(fens: list[str]) -> str:
    """`tests/test_fastsearch.py`'s timeout check, against this evaluation's own score.

    That check ends by asserting the depth-4 score equals `agent.py`'s, which is the property
    the whole port rests on and which cannot hold with the evaluation swapped -- it is a
    different evaluation, and if it scored the same it would not be worth shipping. So it is
    run with the network *off* as well, unchanged, and this is the same check with the one
    assertion that has to change: the score has to equal *its own* clean score, which is what
    actually catches an abort that left a half-written table entry, a stale killer or a move
    buffer in the wrong state.
    """
    probe = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    _, clean_score, _ = fs.search_fixed(probe, 4)
    aborts = 0
    worst_overrun = 0.0
    for index, fen in enumerate(fens):
        clock_ms = (12, 40, 120, 400, 1_500, 9_000)[index % 6]
        fs.reset()
        started = time.perf_counter()
        move = fs.think(fen, clock_ms)
        spent_ms = (time.perf_counter() - started) * 1000.0
        if move not in [candidate.uci() for candidate in chess.Board(fen).legal_moves]:
            raise Failure(f"a {clock_ms} ms search from {fen!r} returned {move}, not legal")
        aborts += int(fs.STATS[fs.ABORTED] != 0)
        _, hard_ms = fs.budgets(clock_ms)
        if hard_ms > 0.0:
            worst_overrun = max(worst_overrun, spent_ms - hard_ms)
    forced = 0
    for fen in fens[:6]:
        legal = [candidate.uci() for candidate in chess.Board(fen).legal_moves]
        move, _, _ = fs.search_fixed(fen, 40, deadline=time.perf_counter() + 0.02, first=0)
        if not fs.STATS[fs.ABORTED]:
            raise Failure(f"depth 40 from {fen!r} in 20 ms did not abort")
        forced += 1
        if move not in legal:
            raise Failure(f"an aborted search from {fen!r} returned {move}, not legal")
    fs.reset()
    _, again_score, _ = fs.search_fixed(probe, 4)
    if again_score != clean_score:
        raise Failure(
            f"after {aborts} aborted searches the same depth-4 search scores {again_score:+d} "
            f"instead of {clean_score:+d}: an abort corrupted the state"
        )
    fs.reset()
    return (
        f"{len(fens)} timed searches, {aborts} aborted on the clock, worst overrun of the "
        f"hard budget {worst_overrun:.0f} ms; {forced} forced mid-iteration aborts all legal; "
        f"depth-4 score unchanged at {again_score:+d}"
    )


def check_speed(depth: int) -> list[tuple[str, float, float]]:
    """The node rate with the network on against the hand evaluation, on the same positions.

    The whole cost of a learned evaluation is here. It is paid twice per node -- an accumulator
    update on the way down and an inner product at the leaf -- and the only question that
    matters is whether what it buys in accuracy is worth what it costs in depth.
    """
    rates = []
    for name, fen in tfs.BENCH:
        measured = []
        for enabled in (True, False):
            fs.search_fixed(fen, 2, nnue=enabled)
            started = time.perf_counter()
            _, _, nodes = fs.search_fixed(fen, depth, nnue=enabled)
            measured.append(nodes / (time.perf_counter() - started) / 1e6)
        rates.append((name, measured[0], measured[1]))
    return rates


def check_import() -> str:
    """Import time and peak memory, in a fresh process, because that is what the platform sees.

    The platform allows ninety seconds before the clock starts and kills the process at 2 GB.
    Measuring in *this* process would measure an import that already happened, so a subprocess
    imports `agent` from scratch and reports what it cost.
    """
    script = (
        "import resource, sys, time\n"
        "started = time.perf_counter()\n"
        "import agent\n"
        "elapsed = time.perf_counter() - started\n"
        "divisor = 1024.0 * 1024.0 if sys.platform == 'darwin' else 1024.0\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor\n"
        "import fastnnue\n"
        "print(f'RESULT {elapsed:.2f} {peak:.0f} {fastnnue.LOADED}', file=sys.stderr)\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        raise Failure(f"importing agent in a fresh process failed:\n{finished.stderr}")
    line = next(
        (line for line in finished.stderr.splitlines() if line.startswith("RESULT")), None
    )
    if line is None:
        raise Failure(f"the import probe printed no result:\n{finished.stderr}")
    _, seconds, megabytes, enabled = line.split()
    if enabled != "True":
        raise Failure("the import probe found no weight file, so it measures the wrong thing")
    return f"{float(seconds):.1f} s with warm-up, {float(megabytes):.0f} MB peak resident"


def sample(rng: random.Random, wanted: int) -> list[str]:
    """Random playouts plus the hand-built endgames, both sides to move throughout."""
    played = positions(rng, wanted // 2)
    endings: list[str] = []
    seeds = [*ENDGAME_SEEDS, *DELTA_SEEDS, *(fen for _, fen in BARE_ENDGAMES)]
    while len(endings) < wanted - len(played):
        board = chess.Board(rng.choice(seeds))
        for _ in range(rng.randint(1, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            endings.append(board.fen())
            board.push(rng.choice(moves))
    fens = played + endings[: wanted - len(played)]
    rng.shuffle(fens)
    return fens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="more positions and more plies")
    parser.add_argument(
        "--weights", nargs="*", default=[], help="weight files to check (default weights/nnue*.npz)"
    )
    arguments = parser.parse_args()

    files = weight_files(arguments.weights)
    print(f"weight files: {', '.join(path.name for path in files)}")
    print(f"loaded by the engine: {fn.STATUS}")
    print(f"shipped switch: USE_NNUE={fn.USE_NNUE}, so a leaf is scored "
          f"{fn.POLICY_NAMES[fn.policy()]!r} unless a test says otherwise")
    print(f"\nfloor division: {check_floor_division()}")
    print(f"rejection: {check_rejection()}")

    rng = random.Random(0x11EE)
    wanted = 10_000 if arguments.full else 2_000
    fens = sample(rng, wanted)

    for path in files:
        print(f"\n{path.name}: {check_load(path)}")
        started = time.perf_counter()
        tally = check_against_reference(path, fens)
        print(
            f"  exact match against tools/nnue/nnue_ref.py: {tally['positions']:,} positions, "
            f"0 mismatches, in {time.perf_counter() - started:.1f} s"
        )
        for name in ("black to move", "en passant", "promotions", "endgames"):
            print(f"    {name:<14} {tally[name]:,}")
        plies = 30_000 if arguments.full else 10_000
        started = time.perf_counter()
        moved = check_increments(path, random.Random(0x22FF), plies)
        print(
            f"  incremental equals from-scratch: {moved['plies']:,} make/unmake sequences, "
            f"0 mismatches, in {time.perf_counter() - started:.1f} s"
        )
        for name in ("captures", "en passant", "promotions", "castles", "nulls"):
            print(f"    {name:<14} {moved[name]:,}")
        print(f"  sanity: {check_sanity(path)}")

    print(f"\npolicies: {check_policies(fens[:400] + [fen for _, fen in BARE_ENDGAMES])}")
    print(f"target marker: {check_target_marker()}")
    print(f"\nbare endgames: {check_bare_endgames(6)}")
    print(f"round 102: {check_round_102(8)}")

    reference_module = load_reference()
    search_rng = random.Random(0x5EA2)
    print(f"\nsearch: {check_search(reference_module, positions(search_rng, 24))}")

    depth = 8 if arguments.full else 7
    print(f"\nnode rate at depth {depth}, network on against the hand evaluation")
    on_total = off_total = 0.0
    for name, on, off in check_speed(depth):
        on_total += on
        off_total += off
        print(f"  {name:<16} {on:6.2f}M nodes/s   against {off:6.2f}M   ({on / off:.0%})")
    count = len(tfs.BENCH)
    print(f"  {'mean':<16} {on_total / count:6.2f}M nodes/s   against {off_total / count:6.2f}M")

    print(f"\nimport: {check_import()}")
    print("\nfastnnue evaluates the network tools/nnue exported, exactly.")


if __name__ == "__main__":
    try:
        main()
    except Failure as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
