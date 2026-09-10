"""Prove `fastsearch` searches the same tree as the engine it was ported from.

`uv run python -m tests.test_fastsearch [--full]`.

The load-bearing test is the first one. `fastsearch` and `agent.py`'s search are fail-soft
alpha-beta over the same evaluation, and the value alpha-beta returns at the root is the exact
minimax value of the position at that depth, whatever order it happened to look at the moves
in and whatever its transposition table remembered. So the two must return the *same score* at
the same depth, and if they do not, the port has changed the tree: a mis-ordered promotion, a
draw scored in the wrong place, a bound stored under the wrong ply. Node counts within a few
per cent of each other say the ordering survived too, but only the score is an assertion,
because ordering is allowed to differ and the score is not.

The rest is what a wrong search still passes the score test by doing: playing an illegal move,
failing to see a mate it is one ply from, walking into a repetition while a queen up, or
leaving the board corrupted after a timeout. Each of those has cost a game somewhere.

This file runs with the learned evaluation switched **off**, throughout. That is not it being
avoided: the score equality above is a claim about the *search*, and it is only a claim at all
while both engines read the same evaluation. `tests/test_nnue.py` runs these same legality,
timeout and backstop checks with the network on, and the node rate below is the hand
evaluation's, which is the number the network's is worth comparing against.
"""

import argparse
import itertools
import random
import sys
import time
from types import ModuleType

import chess

import agent
import fastboard as fb
import fastnnue
import fastsearch as fs
from tests.test_fastboard import STRESS_SEEDS, positions
from tests.test_fasteval import load_reference

# Mates verified against python-chess, not asserted from memory. The mate-in-two positions
# need three plies, which is why they are searched at depth three: two of ours and one of
# theirs. Anything shallower cannot see the second move and would pass by luck.
MATE_IN_ONE: tuple[tuple[str, str], ...] = (
    ("3k4/8/3K4/8/8/8/8/7R w - - 0 1", "h1h8"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
    ("7k/6pp/8/8/8/8/8/R6K w - - 0 1", "a1a8"),
    ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", ""),  # two mates; any of them will do
)

MATE_IN_TWO: tuple[str, ...] = (
    "2k5/8/1K6/8/8/8/8/6R1 w - - 0 1",
    "8/2k5/7R/8/8/8/8/K5R1 w - - 4 3",
    "8/8/8/8/8/8/K4Q2/3k4 w - - 6 4",
    "8/8/8/8/8/k7/8/K1R3Q1 w - - 6 4",
    "8/8/8/8/3Q4/8/2k5/K6R w - - 2 2",
    "8/7R/8/8/8/k7/8/K5Q1 w - - 2 2",
    "8/8/1Q6/8/8/k7/8/K6R w - - 2 2",
    "6k1/pp4p1/2p5/2bp4/8/P5Pb/1P3rrP/2BRRN1K b - - 0 1",
)

# Rated round 85 vs NajeebA, move 56, Black to move. The game searched to depth nine, where
# `g2g1q`, `g2g1r` and `f1e1` all score +21, and played `g2g1r`. The depth asserted below is
# eight rather than nine because that is the one this file can measure: it runs the hand
# evaluation, and at depth nine the hand evaluation happens to separate the two promotions,
# while at depth eight the three-way tie at +14 is the same with the network on and off.
PROMO_TIEBREAK_FEN = "8/6R1/3P4/4K3/Pp6/1P5k/6p1/5r2 b - - 0 56"
PROMO_TIEBREAK_DEPTH = 8

# Saavedra, at the move the study turns on: 6.c8=R wins, 6.c8=Q is stalemate after Rc4+ Qxc4.
# The counterpart to the position above: an under-promotion the search must still play.
SAAVEDRA_FEN = "8/2P5/8/8/3r4/8/2K5/k7 w - - 0 1"

# Middlegame positions for the node-rate figure, which is the whole reason this exists.
BENCH: tuple[tuple[str, str], ...] = (
    ("start", fb.START_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("queens gambit", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"),
    ("sicilian", "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
)


class Failure(Exception):
    """The search did something a chess engine may not do."""


def reference_root(reference: ModuleType, fen: str, depth: int) -> tuple[str, int, int]:
    """Run `agent.py`'s search at a fixed depth from a clean memory, as a reference."""
    reference._MEMORY.table.clear()
    reference._MEMORY.seen.clear()
    reference._MEMORY.history = [0] * len(reference._MEMORY.history)
    reference._MEMORY.expected = None
    board = chess.Board(fen)
    search = reference._Search(deadline=time.perf_counter() + 86_400.0, contempt=0)
    move, score = reference._root(board, depth, None, search)
    return move.uci(), score, search.nodes


def check_against_reference(reference: ModuleType, fens: list[str], depths: tuple[int, ...]) -> (
    dict[str, int]
):
    """The root score of the port must equal the root score of the engine it came from.

    Null-move pruning and late move reductions are switched off here, and only here. They
    are the two things this search does that `agent.py`'s does not, and both are deliberately
    inexact: null move is allowed to miss a line, and a reduction is allowed to be wrong about
    a move it ordered late. With both off the two searches are the same algorithm and the
    scores have to agree exactly; `check_null_move` and `check_late_moves` cover them being
    on.

    The principal variation search is *not* switched off, and this is the test that covers
    it. `agent.py` searches every move at the full window; `fastsearch` searches all but the
    first at a window one wide and re-searches whatever beats it. That is a claim about cost,
    not about value, and the claim is exactly what these scores being equal proves. It has no
    off-switch on purpose: an off-switch here would mean the equality was never measuring the
    search that plays.
    """
    tally = {"searches": 0, "our nodes": 0, "their nodes": 0, "same move": 0}
    for fen in fens:
        for depth in depths:
            move, score, nodes = fs.search_fixed(
                fen, depth, null_move=False, late_moves=False
            )
            their_move, their_score, their_nodes = reference_root(reference, fen, depth)
            if score != their_score:
                raise Failure(
                    f"depth {depth} from {fen!r}: fastsearch scores {score:+d} playing "
                    f"{move}, agent.py scores {their_score:+d} playing {their_move}"
                )
            if move not in [m.uci() for m in chess.Board(fen).legal_moves]:
                raise Failure(f"depth {depth} from {fen!r}: {move} is not legal")
            tally["searches"] += 1
            tally["our nodes"] += nodes
            tally["their nodes"] += their_nodes
            tally["same move"] += move == their_move
    return tally


def check_legal(fens: list[str], depth: int) -> int:
    """Every fixed-depth search returns a move python-chess agrees is legal."""
    for fen in fens:
        move, _, _ = fs.search_fixed(fen, depth)
        if move not in [candidate.uci() for candidate in chess.Board(fen).legal_moves]:
            raise Failure(f"depth {depth} from {fen!r} returned {move}, which is not legal")
    return len(fens)


def check_mates() -> tuple[int, int]:
    """A mate in one and a mate in two are both found at depth three, and scored by distance.

    The score matters as much as the move: `MATE - 1` for a mate on the first ply and
    `MATE - 3` for one on the third is what makes the engine prefer the shorter mate and
    actually finish the game, rather than shuffling between two winning lines.
    """
    for fen, expected in MATE_IN_ONE:
        move, score, _ = fs.search_fixed(fen, 3)
        if score != fs.MATE - 1:
            raise Failure(f"mate in one from {fen!r} scored {score:+d}, want {fs.MATE - 1:+d}")
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(move))
        if not board.is_checkmate():
            raise Failure(f"mate in one from {fen!r} played {move}, which is not mate")
        if expected and move != expected:
            raise Failure(f"mate in one from {fen!r} played {move}, want {expected}")
    for fen in MATE_IN_TWO:
        move, score, _ = fs.search_fixed(fen, 3)
        if score != fs.MATE - 3:
            raise Failure(f"mate in two from {fen!r} scored {score:+d}, want {fs.MATE - 3:+d}")
        if move not in [candidate.uci() for candidate in chess.Board(fen).legal_moves]:
            raise Failure(f"mate in two from {fen!r} returned an illegal {move}")
    return len(MATE_IN_ONE), len(MATE_IN_TWO)


def check_promotion_tiebreak() -> str:
    """A promotion that only ties the queen promotion never gets played as an under-promotion.

    Why it matters: an under-promotion on a tie is a free half point given away. Rated round 85
    reached `8/6R1/3P4/4K3/Pp6/1P5k/6p1/5r2 b` at move 56 and played `g2g1r`. The rook is
    captured exactly as the queen would be, so the two score the same, but a rook is not a
    queen in any line the search did not reach, and the game was drawn.

    The mechanism is ordering, not evaluation. `think` hands the previous iteration's answer to
    `search_root` as `first`, `score_moves` ranks it `TABLE_BONUS` so it is searched before
    everything else, and `search_root` improves on it only with a strict `>`. So once a shallow
    iteration returns the rook - at depth two in that game, where the rook really did outscore
    the queen - every deeper iteration inherits it and no equal score can ever take it back.
    `queen_first` keeps an under-promotion out of that slot.

    So the first half asserts the tie is real and that the bare root still resolves it by
    order, because if the tie ever stops existing this test stops measuring anything. The
    second half asserts what changed: `think` picks the queen promotion or the rook move, never
    the under-promotion, at every clock. The third is the position that says the fix is a
    tie-break and not a ban: Saavedra's `c7c8r` is strictly better than `c7c8q`, and it is
    still played even when the queen promotion is the move handed in as `first`.
    """
    board, st, undo = fb.from_fen(PROMO_TIEBREAK_FEN)
    moves = fb.legal_moves(board, st, undo)
    by_uci = {fb.move_to_uci(move): move for move in moves}
    scores = {}
    depth = PROMO_TIEBREAK_DEPTH
    for name in ("g2g1q", "g2g1r", "f1e1"):
        move, score, _ = fs.search_fixed(PROMO_TIEBREAK_FEN, depth, first=by_uci[name])
        if move != name:
            raise Failure(
                f"the root at depth {depth} was handed {name} first and returned {move}: "
                f"the three moves no longer tie, so this test measures nothing"
            )
        scores[name] = score
    if len(set(scores.values())) != 1:
        raise Failure(f"round 85 move 56 no longer ties: {scores}")

    for clock_ms in (2_000, 8_000, 20_000, 120_000):
        fs.reset()
        played = fs.think(PROMO_TIEBREAK_FEN, clock_ms)
        if played == "g2g1r":
            raise Failure(
                f"round 85 move 56 at a {clock_ms} ms clock played g2g1r again: an "
                f"under-promotion chosen over an equal-scoring queen promotion"
            )
        if played not in ("g2g1q", "f1e1"):
            raise Failure(f"round 85 move 56 at a {clock_ms} ms clock played {played}")

    saavedra, st, undo = fb.from_fen(SAAVEDRA_FEN)
    queen = {fb.move_to_uci(move): move for move in fb.legal_moves(saavedra, st, undo)}["c7c8q"]
    for study in (5, 7):
        move, _, _ = fs.search_fixed(SAAVEDRA_FEN, study, first=queen)
        if move != "c7c8r":
            raise Failure(
                f"Saavedra at depth {study} played {move}, not c7c8r: an under-promotion that "
                f"is strictly better must still be found when the queen promotion goes first"
            )
    fs.reset()
    if fs.think(SAAVEDRA_FEN, 10_000) != "c7c8r":
        raise Failure("Saavedra: think() no longer plays the winning rook promotion")
    tie = scores["g2g1q"]
    return f"round 85 move 56 ties at {tie:+d}, no under-promotion; Saavedra still c7c8r"


def check_null_move(fens: list[str], depth: int) -> str:
    """Null-move pruning must save nodes without losing a mate or returning an illegal move.

    Being unsound about quiet lines is the point of it; being unsound about mates is not, and
    a null move inside a mate line is how an engine reports a forced win it cannot deliver. So
    the mate suites are re-run with it on and have to score exactly what they scored with it
    off, and the saving is reported rather than asserted, because the number is a property of
    the positions and the bench is what decides whether it is worth having.
    """
    for fen, _ in MATE_IN_ONE:
        _, score, _ = fs.search_fixed(fen, 3, null_move=True)
        if score != fs.MATE - 1:
            raise Failure(f"with null move on, mate in one from {fen!r} scored {score:+d}")
    for fen in MATE_IN_TWO:
        _, score, _ = fs.search_fixed(fen, 3, null_move=True)
        if score != fs.MATE - 3:
            raise Failure(f"with null move on, mate in two from {fen!r} scored {score:+d}")
    on_nodes = off_nodes = 0
    agreed = 0
    for fen in fens:
        legal = [candidate.uci() for candidate in chess.Board(fen).legal_moves]
        move_on, _, nodes_on = fs.search_fixed(fen, depth, null_move=True)
        move_off, _, nodes_off = fs.search_fixed(fen, depth, null_move=False)
        if move_on not in legal:
            raise Failure(f"null move on, depth {depth} from {fen!r} returned {move_on}")
        on_nodes += nodes_on
        off_nodes += nodes_off
        agreed += move_on == move_off
    saved = 1.0 - on_nodes / max(off_nodes, 1)
    return (
        f"mates unaffected; over {len(fens)} positions at d{depth} it searched "
        f"{on_nodes:,} nodes against {off_nodes:,} ({saved:.0%} fewer) and chose the same "
        f"move {agreed}/{len(fens)} times"
    )


def check_late_moves(fens: list[str], depth: int) -> str:
    """Reductions must save nodes without losing a mate or returning an illegal move.

    The same shape as `check_null_move`, and for the same reason: what a reduction is allowed
    to cost is a quiet move searched in the wrong order, and what it is not allowed to cost is
    a mate. A reduction is always re-searched at the full depth when it beats the bound, so a
    mate on the reduced side of the tree still has to come back with its exact distance; if
    the re-search were dropped these two suites are what would notice.
    """
    for fen, _ in MATE_IN_ONE:
        _, score, _ = fs.search_fixed(fen, 3, late_moves=True)
        if score != fs.MATE - 1:
            raise Failure(f"with reductions on, mate in one from {fen!r} scored {score:+d}")
    for fen in MATE_IN_TWO:
        _, score, _ = fs.search_fixed(fen, 3, late_moves=True)
        if score != fs.MATE - 3:
            raise Failure(f"with reductions on, mate in two from {fen!r} scored {score:+d}")
    on_nodes = off_nodes = 0
    agreed = 0
    for fen in fens:
        legal = [candidate.uci() for candidate in chess.Board(fen).legal_moves]
        move_on, _, nodes_on = fs.search_fixed(fen, depth, late_moves=True)
        move_off, _, nodes_off = fs.search_fixed(fen, depth, late_moves=False)
        if move_on not in legal:
            raise Failure(f"reductions on, depth {depth} from {fen!r} returned {move_on}")
        on_nodes += nodes_on
        off_nodes += nodes_off
        agreed += move_on == move_off
    saved = 1.0 - on_nodes / max(off_nodes, 1)
    return (
        f"mates unaffected; over {len(fens)} positions at d{depth} it searched "
        f"{on_nodes:,} nodes against {off_nodes:,} ({saved:.0%} fewer) and chose the same "
        f"move {agreed}/{len(fens)} times"
    )


def check_fallback() -> str:
    """A fallback must not cost the fast engine everything it knows about the game.

    `fastsearch` decides whether it is still in the same game by asking whether the position
    it has been handed is one legal move on from the one it expects. Only the move that was
    actually played can set that expectation, so if the python-chess engine plays a move and
    nobody tells the fast engine, the next position looks like a different game: the table is
    cleared and every position the game has stood in is forgotten, for the rest of the game.
    One fallback, and no repetition is ever seen again.

    So this plays a short game through `agent.get_move` with the fast engine broken on one
    move, and asserts the game history only ever grows. Before the fix it went 2, 4, 2, 4:
    two entries is a game that has just started.
    """
    # A quiet middlegame, so the game lasts long enough to have a middle to fall back in.
    # The winning fixture below mates in two or three and never gets that far.
    opening = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    fen = "6k1/5ppp/8/8/8/8/5PPP/Q5KR w - - 8 30"
    fs.reset()
    agent._MEMORY.table.clear()
    agent._MEMORY.seen.clear()
    agent._MEMORY.expected = None

    def broken(fen: str, time_left_ms: int) -> str:
        raise RuntimeError(f"injected: the numba engine failed on {fen!r} at {time_left_ms}")

    board = chess.Board(opening)
    real_think = fs.think
    counts, fell_back = [], 0
    for index in range(3):
        if index == 1:  # mid-sequence, so there is a history to lose on both sides of it
            fs.think = broken
        before = int(fs.STATS[fs.GAME_COUNT])
        move = agent.get_move(board.fen(), 2_000)
        if fs.think is broken:
            fs.think = real_think
            fell_back += 1
            if int(fs.STATS[fs.GAME_COUNT]) <= before:
                raise Failure("the fallback move was never recorded in the game history")
        if move not in [candidate.uci() for candidate in board.legal_moves]:
            raise Failure(f"get_move returned {move}, which is not legal in {board.fen()!r}")
        counts.append(int(fs.STATS[fs.GAME_COUNT]))
        board.push(chess.Move.from_uci(move))
        replies = list(board.legal_moves)
        if not replies:
            break
        board.push(replies[0])  # a fixed opponent, so the fens are one legal move on
    fs.think = real_think
    # Three moves is the minimum that says anything: one before the fallback, the fallback
    # itself, and the one after it, which is the move that used to find the history gone.
    if len(counts) < 3:
        raise Failure(f"the fixture ended after {len(counts)} moves; it needs at least 3")
    if fell_back != 1:
        raise Failure(f"the test never forced a fallback: {fell_back}")
    for earlier, later in itertools.pairwise(counts):
        if later <= earlier:
            raise Failure(f"the game history was thrown away across the fallback: {counts}")
    if fs._EXPECTED is None:
        raise Failure("nothing set the expected position after the last move")

    # And the repetition it knows about still binds afterwards. Same construction as
    # `check_repetition`, run on the history this game actually built rather than a fresh one.
    seen_before = int(fs.STATS[fs.GAME_COUNT])
    wanted, _, _ = fs.search_fixed(fen, 5)
    board2, st2, undo2 = fb.from_fen(fen)
    fb.make_move(board2, st2, undo2, fb.uci_to_move(board2, st2, undo2, wanted))
    fs.reset()
    fs.GAME_KEYS[0] = int(st2[7])
    fs.STATS[fs.GAME_COUNT] = 1
    avoided, _, _ = fs.search_fixed(fen, 5, fresh=False, contempt=-fs.CONTEMPT)
    fs.reset()
    if avoided == wanted:
        raise Failure("after a fallback the engine walked into a position it had seen")
    return (
        f"{fell_back} forced fallback, history grew {counts} and was never reset "
        f"(it held {seen_before} positions after); still refuses {wanted} for {avoided}"
    )


def check_repetition() -> str:
    """Winning, the search must refuse the move that repeats a position the game has seen.

    Built rather than hand-written, so it cannot rot: search the position once with an empty
    game history to see what the engine wants to play, then put the position that move reaches
    into the game history and search again. The move is now a draw offer worth the contempt
    score instead of the win it was, and anything else the engine picks proves the lookback
    reached past the root into the game and that contempt was applied with the right sign.
    """
    # White is a queen and a rook up with the move; every reasonable line is winning.
    fen = "6k1/5ppp/8/8/8/8/5PPP/Q5KR w - - 8 30"
    wanted, winning_score, _ = fs.search_fixed(fen, 5)
    if winning_score < 500:
        raise Failure(f"the repetition fixture is not winning: {winning_score:+d}")

    board, st, undo = fb.from_fen(fen)
    fb.make_move(board, st, undo, fb.uci_to_move(board, st, undo, wanted))
    repeated_key = int(st[7])

    # The game history is read backwards from its end, and the last entry is the position we
    # are searching from, so the position to avoid goes in one before it. `search_fixed` adds
    # the root itself, which is why only the earlier one is seeded here.
    fs.reset()
    fs.GAME_KEYS[0] = repeated_key
    fs.STATS[fs.GAME_COUNT] = 1
    # Contempt is negative when we are winning: a draw is a loss of the half point in hand.
    avoided, score, _ = fs.search_fixed(fen, 5, fresh=False, contempt=-fs.CONTEMPT)
    fs.reset()
    if avoided == wanted:
        raise Failure(f"still played {wanted} into a position the game has already stood in")
    if score < 500:
        raise Failure(f"avoiding the repetition threw the win away: {score:+d}")
    return f"{wanted} avoided for {avoided}, still {score:+d}"


def check_table() -> str:
    """The table has to be answering. A hit rate near zero means it is not being read.

    A deep search from one position re-reaches the same positions by many move orders, so a
    healthy table answers a large fraction of probes. This is a smoke test with a wide band,
    not a tuning target: what it catches is a key that never matches or a slot that is never
    written, both of which leave the search correct and several plies weaker.
    """
    fs.reset()
    fs.search_fixed(fb.START_FEN, 8)
    probes = int(fs.STATS[fs.TT_PROBES])
    hits = int(fs.STATS[fs.TT_HITS])
    stores = int(fs.STATS[fs.TT_STORES])
    if probes == 0 or stores == 0:
        raise Failure(f"the table was not used at all: {probes} probes, {stores} stores")
    rate = hits / probes
    if rate < 0.10:
        raise Failure(f"table hit rate {rate:.1%} is too low to be working")
    if rate > 0.95:
        raise Failure(f"table hit rate {rate:.1%} is too high to be believable")
    return f"{probes:,} probes, {hits:,} hits ({rate:.0%}), {stores:,} stores"


def check_reserve() -> str:
    """The soft budget keeps a reserve off the clock, and the reserve belongs to one game.

    The point of the reserve is the increment: a long game at 120 s + 0.5 s used to walk the
    clock down until every move cost about what the increment paid back, and the 172-move
    replay settled at 5.3 s. What has to hold is that a move made on a clock inside the reserve
    plans to spend *less* than the increment, so the clock climbs back out instead of sitting
    there. `SOFT_BONUS_MS` is the floor under that, and it is 400 ms against a 500 ms increment.

    The rest is the reserve being a fact about this game and no other. It is inferred from the
    first clock, because the platform tells the agent neither the base clock nor the increment,
    and the only thing that says a new game has started is `observe` deciding the position it
    was handed is not one legal move on from the one we handed back. So a 45 s game that starts
    in a process which just played a 120 s game must derive its reserve from 45 s, or its every
    move is budgeted against three times the reserve it should have and it plays down a ply for
    the whole game.
    """
    increment_ms = 500.0
    fs.reset()
    fs.budgets(120_000)
    soft_ms, hard_ms = fs.budgets(6_000)
    if soft_ms >= increment_ms:
        raise Failure(
            f"at a 6 s clock after a 120 s first clock the soft budget is {soft_ms:.0f} ms, "
            f"not under the {increment_ms:.0f} ms increment: the clock cannot recover"
        )
    if soft_ms < fs.SOFT_BONUS_MS:
        raise Failure(f"the soft budget fell to {soft_ms:.0f} ms, below the bonus floor")
    planned_ms = soft_ms
    if hard_ms != max(min(6_000 / fs.HARD_DIVISOR, 6_000 - fs.SAFETY_MARGIN_MS), 0.0):
        raise Failure(f"the hard budget moved: {hard_ms:.0f} ms at a 6 s clock")

    # A second game in the same process, reached the way a real one is: we hand back a move,
    # and the next position we are given belongs to a game we are not in.
    fs.remember_played(fb.START_FEN, "e2e4")
    other = "8/8/8/4k3/8/4K3/8/8 w - - 0 1"
    _, st, _ = fb.from_fen(other)
    fs.observe(int(st[7]))
    if fs._FIRST_CLOCK_MS is not None:
        raise Failure("a position from another game left the first clock behind")
    fresh, _ = fs.budgets(45_000)
    expected = (45_000 - 45_000 / fs.RESERVE_DIVISOR) / fs.SOFT_DIVISOR + fs.SOFT_BONUS_MS
    if abs(fresh - expected) > 0.5:
        raise Failure(
            f"the 45 s game's first move budgets {fresh:.0f} ms, not the {expected:.0f} ms a "
            f"45 s reserve gives: it inherited the 120 s game's reserve"
        )

    # Degenerate first clocks. A reserve is only ever subtracted, never divided by, so the
    # harness's short-clock tests and a zero clock have to come out as the plain formula or a
    # small reserve, and never as an exception or a negative budget.
    for first_ms, clock_ms in ((0, 9_000), (1_500, 1_500), (1_500, 40), (120_000, 0)):
        fs.reset()
        fs.budgets(first_ms)
        soft_ms, hard_ms = fs.budgets(clock_ms)
        if soft_ms < 0.0 or hard_ms < 0.0 or soft_ms > max(hard_ms, 0.0) + 0.5:
            raise Failure(
                f"a {first_ms} ms first clock and a {clock_ms} ms clock give soft "
                f"{soft_ms:.0f} ms and hard {hard_ms:.0f} ms"
            )
    # A game whose first clock is zero has no reserve to take, and then this is prod's formula
    # exactly. Nothing about a reserve may make a budget smaller than the one it replaced by
    # any route other than a reserve there was a clock to pay for.
    fs.reset()
    fs.budgets(0)
    zero_soft, _ = fs.budgets(9_000)
    if abs(zero_soft - (9_000 / fs.SOFT_DIVISOR + fs.SOFT_BONUS_MS)) > 0.5:
        raise Failure("a zero first clock does not fall back to the budget without a reserve")
    fs.reset()
    return (
        f"reserve 1/{fs.RESERVE_DIVISOR} of the first clock, 6 s clock in a 120 s game plans "
        f"{planned_ms:.0f} ms against a 500 ms increment"
    )


def check_timed(fens: list[str], reference: ModuleType) -> str:
    """A search that runs out of time returns a legal move and leaves the shared arrays sane.

    The abort path never runs in a fixed-depth test and always runs in a game. Budgets small
    enough to abort mid-iteration are used deliberately here, and the check that matters comes
    after them: a fixed-depth search whose score is already known must still return that exact
    score. A timeout that left a half-written entry in the table, a stale killer or a move
    buffer in the wrong state would show up as a different number, and nowhere else.
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
    # The clocks above give hard budgets the iteration gate simply does not overrun, so on a
    # quiet machine none of them may actually abort. The abort path is the one that always
    # runs in a real game, so it is forced here: a depth nothing finishes, against a deadline
    # a few milliseconds out, has to come back aborted, legal, and with the partial-iteration
    # rule intact - anything in ROOT_BEST has been proven better than the move it replaced.
    forced = 0
    for fen in fens[:6]:
        legal = [candidate.uci() for candidate in chess.Board(fen).legal_moves]
        move, _, _ = fs.search_fixed(
            fen, 40, deadline=time.perf_counter() + 0.02, first=0
        )
        if not fs.STATS[fs.ABORTED]:
            raise Failure(f"depth 40 from {fen!r} in 20 ms did not abort")
        forced += 1
        if move not in legal:
            raise Failure(f"an aborted search from {fen!r} returned {move}, not legal")
        root_best = int(fs.STATS[fs.ROOT_BEST])
        if root_best and fb.move_to_uci(root_best) not in legal:
            raise Failure(f"an aborted search left an illegal move in ROOT_BEST from {fen!r}")
    fs.reset()

    again_move, again_score, _ = fs.search_fixed(probe, 4)
    if again_score != clean_score:
        raise Failure(
            f"after {aborts} aborted searches the same depth-4 search scores "
            f"{again_score:+d} instead of {clean_score:+d}: an abort corrupted the state"
        )
    _, their_score, _ = reference_root(reference, probe, 4)
    if again_score != their_score:
        raise Failure(f"depth-4 probe scores {again_score:+d}, agent.py says {their_score:+d}")
    fs.reset()
    return (
        f"{len(fens)} timed searches, {aborts} of them aborted on the clock, worst overrun "
        f"of the hard budget {worst_overrun:.0f} ms; {forced} forced mid-iteration aborts all "
        f"legal; depth-4 score unchanged at {again_score:+d} playing {again_move}"
    )


def check_backstop(fens: list[str]) -> str:
    """The timer thread stops a search that the node-counted clock read cannot.

    The clock is read every `CHECK_MASK + 1` nodes, so a search only stops on it as often as
    nodes come, and a slow subtree near the deadline is an overrun. This defeats that read
    outright, with a mask no node count ever satisfies, and asks for depth 40, which nothing
    finishes, so that only the thread can end the search. Each search then has to stop close
    to its deadline, flag both the expiry and the abort, and still return a legal move. With
    the thread inert every one of these runs until the heat death of the test, so it cannot
    pass by accident.
    """
    masks = fs.NODE_CHECK_MASK, fs.FINE_CHECK_MASK
    never = (1 << 62) - 1
    fs.NODE_CHECK_MASK = fs.FINE_CHECK_MASK = never
    worst_late = 0.0
    try:
        for index, fen in enumerate(fens):
            budget_ms = (150, 300, 500)[index % 3]
            legal = [candidate.uci() for candidate in chess.Board(fen).legal_moves]
            fs.reset()
            started = time.perf_counter()
            deadline = started + budget_ms / 1000.0
            backstop = fs._arm_backstop(deadline)
            try:
                move, _, _ = fs.search_fixed(fen, 40, deadline=deadline, first=0)
            finally:
                backstop.cancel()
                backstop.join()
            spent_ms = (time.perf_counter() - started) * 1000.0
            if not fs.STATS[fs.EXPIRED] or not fs.STATS[fs.ABORTED]:
                raise Failure(f"the backstop never fired on a {budget_ms} ms budget from {fen!r}")
            if move not in legal:
                raise Failure(f"a backstopped search from {fen!r} returned {move}, not legal")
            worst_late = max(worst_late, spent_ms - budget_ms)
            if spent_ms > budget_ms + 100.0:
                raise Failure(
                    f"the backstop fired {spent_ms - budget_ms:.0f} ms late on a {budget_ms} ms "
                    f"budget from {fen!r}"
                )
    finally:
        fs.NODE_CHECK_MASK, fs.FINE_CHECK_MASK = masks
        fs.reset()
    return (
        f"{len(fens)} depth-40 searches with the clock read disabled all stopped on the "
        f"thread, at most {worst_late:.0f} ms after the deadline, all legal"
    )


def check_speed(depth: int) -> list[tuple[str, float, int]]:
    """The node rate in the middlegame, which is the number this whole port exists for."""
    rates = []
    for name, fen in BENCH:
        fs.search_fixed(fen, 2)  # touch the position once so the table is not the variable
        started = time.perf_counter()
        _, _, nodes = fs.search_fixed(fen, depth)
        elapsed = time.perf_counter() - started
        rates.append((name, nodes / elapsed / 1e6, nodes))
    return rates


def check_perft() -> int:
    """The board this search sits on still counts what it counted."""
    suite = (
        (fb.START_FEN, 5, 4_865_609),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4_085_603),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624),
    )
    for fen, depth, want in suite:
        got = fb.run_perft(fen, depth)
        if got != want:
            raise Failure(f"perft({fen!r}, {depth}) = {got:,}, want {want:,}")
    return len(suite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="more positions and more depth")
    arguments = parser.parse_args()

    # One variable at a time: this file measures the search over `fasteval`, which is the
    # search `agent.py` describes. Leaving the network on would make every score comparison
    # below a comparison between two different engines.
    fastnnue.USE_NNUE = False

    rng = random.Random(0x5EA2)
    sample = positions(rng, 200 if arguments.full else 60)
    seeds = [fb.START_FEN, *STRESS_SEEDS]

    print(f"perft: {check_perft()} positions still match")

    reference = load_reference()
    depths = (2, 3, 4, 5) if arguments.full else (2, 3, 4)
    against = sample[: 40 if arguments.full else 16] + seeds
    print(f"\nsame root score as {reference.__file__}")
    started = time.perf_counter()
    tally = check_against_reference(reference, against, depths)
    if not arguments.full:
        # Depths two to four barely engage the table, the killers or the history, so the
        # default run would establish the equality claim only for a search that has not
        # started using its memory yet. `--full` does depth five over everything; this does
        # it over enough positions that the default run says something about it too.
        deep = check_against_reference(reference, against[:8], (5,))
        for name in tally:
            tally[name] += deep[name]
        depths = (*depths, 5)
    print(
        f"  {tally['searches']} searches over {len(against)} positions at depths "
        f"{depths}, all scores equal, in {time.perf_counter() - started:.0f} s"
    )
    print(
        f"    nodes {tally['our nodes']:,} here against {tally['their nodes']:,} there "
        f"({tally['our nodes'] / max(tally['their nodes'], 1):.2f}x), "
        f"same move chosen {tally['same move']}/{tally['searches']}"
    )

    print(f"\nlegal at a fixed depth: {check_legal(sample[:30], 5)} positions searched at d5")

    ones, twos = check_mates()
    print(f"mates at depth three: {ones} mates in one and {twos} mates in two, all found")

    print(f"promotion tie-break: {check_promotion_tiebreak()}")
    print(f"null move: {check_null_move(sample[:20], 6)}")
    print(f"late moves: {check_late_moves(sample[:20], 6)}")
    print(f"fallback: {check_fallback()}")
    print(f"repetition: {check_repetition()}")
    print(f"table: {check_table()}")
    print(f"reserve: {check_reserve()}")
    print(f"timeouts: {check_timed(sample[:24], reference)}")
    print(f"backstop: {check_backstop(sample[:6])}")

    depth = 8 if arguments.full else 7
    print(f"\nnode rate at depth {depth}")
    for name, rate, nodes in check_speed(depth):
        print(f"  {name:<16} {rate:6.2f}M nodes/s   {nodes:>12,} nodes")

    print("\nfastsearch searches the same tree as agent.py, faster.")


if __name__ == "__main__":
    try:
        main()
    except Failure as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
