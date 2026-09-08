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
"""

import argparse
import random
import sys
import time
from types import ModuleType

import chess

import fastboard as fb
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

    Null-move pruning is switched off here, and only here. It is the one thing this search
    does that `agent.py`'s does not, and it is deliberately unsound: it is allowed to miss a
    line, which is the trade that buys the depth. With it off the two searches are the same
    algorithm and the scores have to agree exactly; `check_null_move` covers it being on.
    """
    tally = {"searches": 0, "our nodes": 0, "their nodes": 0, "same move": 0}
    for fen in fens:
        for depth in depths:
            move, score, nodes = fs.search_fixed(fen, depth, null_move=False)
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


def check_null_move(fens: list[str], depth: int) -> str:
    """Null-move pruning must save nodes without losing a mate or returning an illegal move.

    Being unsound about quiet lines is the point of it; being unsound about mates is not, and
    a null move inside a mate line is how an engine reports a forced win it cannot deliver. So
    the mate suites are re-run with it on and have to score exactly what they scored with it
    off, and the saving is reported rather than asserted, because the number is a property of
    the positions and the bench is what decides whether it is worth having.
    """
    for fen, _ in MATE_IN_ONE:
        _, score, _ = fs.search_fixed(fen, 3)
        if score != fs.MATE - 1:
            raise Failure(f"with null move on, mate in one from {fen!r} scored {score:+d}")
    for fen in MATE_IN_TWO:
        _, score, _ = fs.search_fixed(fen, 3)
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
        f"of the hard budget {worst_overrun:.0f} ms; depth-4 score unchanged at "
        f"{again_score:+d} playing {again_move}"
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

    print(f"null move: {check_null_move(sample[:20], 6)}")
    print(f"repetition: {check_repetition()}")
    print(f"table: {check_table()}")
    print(f"timeouts: {check_timed(sample[:24], reference)}")

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
