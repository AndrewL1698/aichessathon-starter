"""Prove `fastsearch` against the Python engine in `agent.py`.

    uv run python -m tests.test_fastsearch [--positions 10000] [--search-positions 300]
                                           [--search-depth 3] [--deep-positions 40] [--speed]

Three gates and one measurement. The compiled evaluation has to return exactly the integer
`agent.evaluate` returns, on thousands of random positions generated the way
`tests.test_fastboard` generates them and on hand-built endings that random play rarely
reaches. The compiled search, at a fixed depth with the table, killers and history switched
off on both sides, has to return the same score as `agent._root`, and the same move unless
the two moves tie, which is checked by scoring the compiled move with the Python search. With
everything switched on the two are compared again and the agreement is reported, not gated:
different tie-breaks fill the tables differently and the searches legitimately drift. Then
`--speed` reports both engines' node rates on the harness openings.
"""

import argparse
import contextlib
import io
import random
import statistics
import time
from collections.abc import Iterator

import chess
import numpy as np

import agent
import fastboard as fb
import fastsearch as fs
from harness.rules import OPENINGS
from tests.test_fastboard import positions

# Positions chosen to reach the branches random playouts do not: bare kings being mopped up,
# a loose mop-up with a defender still on the board, the drawish no-pawn halving, the dead
# draw of a lone minor, kings with and without shields, passed and doubled and isolated pawns,
# rooks on open and half-open files, and the bishop pair. Each is checked from both sides to
# move where the fen allows.
HANDMADE: tuple[str, ...] = (
    "8/8/8/3k4/8/8/8/R3K3 w - - 0 1",
    "8/8/8/3k4/8/8/8/R3K3 b - - 0 1",
    "7k/8/8/8/8/8/8/3QK3 w - - 0 1",
    "k7/8/8/8/8/8/8/3QK3 b - - 0 1",
    "8/8/8/8/8/2k5/8/K1q5 w - - 0 1",
    "8/8/3k4/8/8/8/8/2BNK3 w - - 0 1",
    "8/8/3k4/8/8/8/8/2B1K3 w - - 0 1",
    "8/8/3k4/8/8/8/8/2N1K3 b - - 0 1",
    "8/8/3k4/8/8/8/8/R1b1K3 w - - 0 1",
    "8/8/3k4/8/8/8/8/R1b1K3 b - - 0 1",
    "8/8/3k4/8/8/8/8/RR2q1K1 w - - 0 1",
    "8/8/2bk4/8/8/8/8/R3K3 w - - 0 1",
    "8/8/2nk4/8/8/8/8/RQ2K3 w - - 0 1",
    "8/1p1k4/8/8/8/8/1P1K4/8 w - - 0 1",
    "8/1p1k4/8/8/8/8/1P1K4/8 b - - 0 1",
    "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
    "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",
    "r5k1/8/8/8/8/8/5PPP/6K1 w - - 0 1",
    "4k3/8/8/8/8/8/PPP5/4K3 w - - 0 1",
    "4k3/pp6/8/8/8/8/5PPP/4K3 w - - 0 1",
    "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/3P4/3P4/8/8/4K3 w - - 0 1",
    "4k3/8/8/2pP4/8/8/8/4K3 w - - 0 1",
    "4k3/2p5/8/3P4/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/3P4/8/8/3p4/4K3 b - - 0 1",
    "4k3/8/8/8/8/8/P1P1P1P1/4K3 w - - 0 1",
    "4k3/1p1p1p1p/8/8/8/8/8/4K3 b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/1ppppppp/8/8/8/8/PPPPPPP1/R3K2R w KQkq - 0 1",
    "2b1kb2/8/8/8/8/8/8/2B1KB2 w - - 0 1",
    "2b1kb2/pppppppp/8/8/8/8/PPPPPPPP/2B1K3 w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "5rk1/5ppp/8/8/8/8/8/1Q4K1 w - - 0 1",
    "5rk1/5ppp/8/8/8/8/8/1Q4K1 b - - 0 1",
    "8/8/8/8/8/1k6/8/K6q w - - 0 1",
    "8/8/8/8/8/8/2k5/K7 w - - 0 1",
    "8/8/8/8/8/8/2kp4/K7 w - - 0 1",
    "8/8/8/8/8/8/2kq4/K7 w - - 0 1",
    "8/8/8/8/8/5k2/8/3K1B2 w - - 0 1",
    "8/8/8/8/8/5k2/8/3K1BB1 w - - 0 1",
    "8/8/8/8/8/5k2/8/3K1BN1 w - - 0 1",
    "8/8/8/8/8/5k2/8/3K1NN1 w - - 0 1",
)


class Failure(Exception):
    """A verification step disagreed with the Python engine."""


def tables() -> tuple[np.ndarray, np.ndarray]:
    """The evaluation's numbers, packed from `agent.py` exactly as `agent.py` packs them."""
    pst = fs.pack_tables(agent.MIDDLEGAME_TABLES, agent.ENDGAME_TABLES, agent.PIECE_VALUES)
    weights = fs.pack_weights(
        piece_values=agent.PIECE_VALUES,
        passed_mg=agent.PASSED_MG,
        passed_eg=agent.PASSED_EG,
        isolated_mg=agent.ISOLATED_MG,
        isolated_eg=agent.ISOLATED_EG,
        doubled_mg=agent.DOUBLED_MG,
        doubled_eg=agent.DOUBLED_EG,
        rook_open_mg=agent.ROOK_OPEN_MG,
        rook_open_eg=agent.ROOK_OPEN_EG,
        rook_semi_open_mg=agent.ROOK_SEMI_OPEN_MG,
        rook_semi_open_eg=agent.ROOK_SEMI_OPEN_EG,
        bishop_pair_mg=agent.BISHOP_PAIR_MG,
        bishop_pair_eg=agent.BISHOP_PAIR_EG,
        shield_penalty=agent.SHIELD_PENALTY,
        shield_max_cover=agent.SHIELD_MAX_COVER,
        mop_up_cmd=agent.MOP_UP_CMD,
        mop_up_close=agent.MOP_UP_CLOSE,
        mop_up_loose_cmd=agent.MOP_UP_LOOSE_CMD,
        mop_up_loose_close=agent.MOP_UP_LOOSE_CLOSE,
        mop_up_min_advantage=agent.MOP_UP_MIN_ADVANTAGE,
        mop_up_max_weak_pieces=agent.MOP_UP_MAX_WEAK_PIECES,
        mop_up_bare_pieces=agent.MOP_UP_BARE_PIECES,
        drawish_margin=agent.DRAWISH_MARGIN,
        phase_rook=agent.PHASE_ROOK,
        phase_queen=agent.PHASE_QUEEN,
        phase_max=agent.PHASE_MAX,
    )
    return pst, weights


def check_evaluation(fens: list[str], pst: np.ndarray, weights: np.ndarray) -> int:
    """Every fen through both evaluations. Returns how many were compared."""
    compared = 0
    for fen in fens:
        reference = agent.evaluate(chess.Board(fen))
        board, st, _ = fb.from_fen(fen)
        got = int(fs.evaluate(board, st, pst, weights))
        if got != reference:
            raise Failure(f"evaluate({fen}) = {got}, agent.evaluate = {reference}")
        compared += 1
    return compared


class _NoTable(dict[agent._Key, agent._Entry]):
    """A transposition table that forgets everything it is told, to switch the Python one off."""

    def __setitem__(self, key: agent._Key, value: agent._Entry) -> None:
        return None


@contextlib.contextmanager
def python_engine(board: chess.Board, memory: bool) -> Iterator[None]:
    """`agent.py` set up as `_think` would set it up for this root, with or without its memory.

    Without: an empty table that stays empty, no killers and no history, so ordering is
    MVV-LVA alone, exactly what the compiled side does with `C_USE_TABLE` and `C_USE_KILLERS`
    off. Either way the root is in `seen`, as `_observe` puts it there in a game.
    """
    saved = (agent._MEMORY.table, agent._MEMORY.seen, agent._MEMORY.history, agent._remember_cutoff)
    agent._MEMORY.table = {} if memory else _NoTable()
    agent._MEMORY.seen = {agent._key(board)}
    agent._MEMORY.history = [0] * len(agent._MEMORY.history)
    if not memory:
        agent._remember_cutoff = lambda board, move, depth, ply, search: None
    try:
        yield
    finally:
        agent._MEMORY.table, agent._MEMORY.seen, agent._MEMORY.history, agent._remember_cutoff = (
            saved
        )


def check_constants() -> None:
    for name in (
        "MATE",
        "INFINITY",
        "MATE_FOUND",
        "QUIESCENCE_MAX_PLY",
        "STALEMATE_PIECE_LIMIT",
        "FIFTY_MOVE_PLIES",
        "EXACT",
        "LOWER",
        "UPPER",
        "TABLE_BONUS",
        "CAPTURE_BONUS",
        "PROMOTION_BONUS",
        "KILLER_BONUS",
        "HISTORY_CAP",
    ):
        if getattr(agent, name) != getattr(fs, name):
            raise Failure(f"{name}: agent {getattr(agent, name)} != fastsearch {getattr(fs, name)}")
    print("  the search constants agree")


def check_search(
    fens: list[str], depth: int, state: fs.SearchState, memory: bool
) -> tuple[int, int, int, int]:
    """Both searches at one fixed depth. Returns (positions, same move, ties, score mismatches).

    Without memory a score mismatch is a failure. With memory it is counted and reported.
    """
    compared = same = ties = mismatches = 0
    state.ctl[fs.C_USE_TABLE] = int(memory)
    state.ctl[fs.C_USE_KILLERS] = int(memory)
    for fen in fens:
        board = chess.Board(fen)
        if board.is_game_over() or not board.is_valid():
            continue
        contempt = agent._contempt(agent.evaluate(board))
        moves = list(board.legal_moves)
        agent._order(board, moves)
        first = moves[0]
        with python_engine(board, memory):
            search = agent._Search(deadline=float("inf"), contempt=contempt)
            with contextlib.redirect_stdout(io.StringIO()):
                py_move, py_score = agent._root(board, depth, first, search)
            state.set_position(fen)
            state.new_game()
            state.remember(state.key())
            state.begin_move(contempt)
            first_c = fb.uci_to_move(state.board, state.st, state.undo, first.uci())
            c_move, c_score = fs.root(state, depth, first_c)
            compared += 1
            if c_score != py_score:
                if not memory:
                    raise Failure(
                        f"depth {depth} at {fen}: compiled {fb.move_to_uci(c_move)} {c_score}, "
                        f"python {py_move.uci()} {py_score}"
                    )
                mismatches += 1
                continue
            if fb.move_to_uci(c_move) == py_move.uci():
                same += 1
                continue
            if memory:
                ties += 1
                continue
            # Different moves at the same score is a tie only if the Python search agrees
            # the compiled move is worth that score.
            board.push(chess.Move.from_uci(fb.move_to_uci(c_move)))
            check = agent._Search(deadline=float("inf"), contempt=contempt)
            alternative = -agent._negamax(
                board, depth - 1, 1, -agent.INFINITY, agent.INFINITY, check
            )
            board.pop()
            if alternative != py_score:
                raise Failure(
                    f"depth {depth} at {fen}: compiled {fb.move_to_uci(c_move)} scores "
                    f"{alternative} by the Python search, not {py_score} ({py_move.uci()})"
                )
            ties += 1
    return compared, same, ties, mismatches


def check_speed(state: fs.SearchState, depth: int, python_depth: int) -> None:
    """Node rates on the harness openings, everything on, fresh table per position."""
    compiled_rates, python_rates = [], []
    state.ctl[fs.C_USE_TABLE] = 1
    state.ctl[fs.C_USE_KILLERS] = 1
    print(f"  {'opening':<22} {'compiled d' + str(depth):>16} {'python d' + str(python_depth):>16}")
    for name, fen in OPENINGS:
        board = chess.Board(fen)
        contempt = agent._contempt(agent.evaluate(board))
        state.set_position(fen)
        state.new_game()
        state.remember(state.key())
        state.begin_move(contempt)
        moves = fb.legal_moves(state.board, state.st, state.undo)
        started = time.perf_counter()
        first = moves[0]
        for d in range(1, depth + 1):
            first, _ = fs.root(state, d, first)
        elapsed = time.perf_counter() - started
        compiled_rate = state.nodes / elapsed
        compiled_rates.append(compiled_rate)

        py_moves = list(board.legal_moves)
        agent._order(board, py_moves)
        py_first = py_moves[0]
        with python_engine(board, True):
            search = agent._Search(deadline=float("inf"), contempt=contempt)
            started = time.perf_counter()
            for d in range(1, python_depth + 1):
                with contextlib.redirect_stdout(io.StringIO()):
                    py_first, _ = agent._root(board, d, py_first, search)
            elapsed = time.perf_counter() - started
        python_rate = search.nodes / elapsed
        python_rates.append(python_rate)
        print(
            f"  {name:<22} {compiled_rate / 1e3:10.0f} k nps {python_rate / 1e3:10.0f} k nps  "
            f"({state.nodes:,} vs {search.nodes:,} nodes)"
        )
    ratio = statistics.median(compiled_rates) / statistics.median(python_rates)
    print(
        f"  median compiled {statistics.median(compiled_rates) / 1e3:.0f} k nps, "
        f"python {statistics.median(python_rates) / 1e3:.0f} k nps, ratio {ratio:.1f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--search-positions", type=int, default=300)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--deep-positions", type=int, default=40, help="also at depth + 1")
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--speed-depth", type=int, default=6)
    parser.add_argument("--python-depth", type=int, default=4)
    arguments = parser.parse_args()

    pst, weights = tables()
    print("evaluation")
    handmade = check_evaluation(list(HANDMADE), pst, weights)
    print(f"  {handmade} hand-built endings and structures agree")

    rng = random.Random(20260908)
    fens = positions(rng, arguments.positions)
    started = time.perf_counter()
    compared = check_evaluation(fens, pst, weights)
    print(f"  {compared:,} generated positions agree, {time.perf_counter() - started:.1f} s")

    print("\nsearch")
    check_constants()
    state = fs.SearchState(pst, weights)
    started = time.perf_counter()
    nodes = fs.warm(state)
    print(
        f"  warm-up search from the start position: {nodes:,} nodes, "
        f"{time.perf_counter() - started:.2f} s"
    )

    sample = list(HANDMADE) + fens[: arguments.search_positions]
    for depth, subset in (
        (arguments.search_depth, sample),
        (arguments.search_depth + 1, sample[: arguments.deep_positions]),
    ):
        if not subset:
            continue
        started = time.perf_counter()
        compared, same, ties, _ = check_search(subset, depth, state, memory=False)
        print(
            f"  depth {depth}, table and killers off: {compared} positions, same score on all, "
            f"same move on {same}, {ties} ties, {time.perf_counter() - started:.0f} s"
        )
        started = time.perf_counter()
        compared, same, ties, mismatches = check_search(subset, depth, state, memory=True)
        print(
            f"  depth {depth}, everything on:         {compared} positions, same move on {same}, "
            f"{ties} same score other move, {mismatches} different scores "
            f"(reported, not gated), {time.perf_counter() - started:.0f} s"
        )

    if arguments.speed:
        print("\nspeed")
        check_speed(state, arguments.speed_depth, arguments.python_depth)

    print("\nEverything matches agent.py.")


if __name__ == "__main__":
    main()
