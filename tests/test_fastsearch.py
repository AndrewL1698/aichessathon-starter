"""Prove `fastsearch` against the Python engine in `agent.py`.

    uv run python -m tests.test_fastsearch [--positions 10000]

The compiled evaluation has to return exactly the integer `agent.evaluate` returns, on
thousands of random positions generated the way `tests.test_fastboard` generates them, and on
hand-built endings that random play rarely reaches: mop-up, the no-pawn draw rules, bare and
covered kings. Any mismatch is a failure, because the whole point of the port is that v3.0 is
v2.4's judgement at a higher node rate.
"""

import argparse
import random
import time

import chess
import numpy as np

import agent
import fastboard as fb
import fastsearch as fs
from tests.test_fastboard import positions

# Positions chosen to reach the branches random playouts do not: bare kings being mopped up,
# a loose mop-up with a defender still on the board, the drawish no-pawn halving, the dead
# draw of a lone minor, kings with and without shields, passed and doubled and isolated pawns,
# rooks on open and half-open files, and the bishop pair. Each is checked from both sides to
# move where the fen allows.
HANDMADE: tuple[str, ...] = (
    "8/8/8/3k4/8/8/8/R3K3 w - - 0 1",
    "8/8/8/3k4/8/8/8/R3K3 b - - 0 1",
    "7k/8/8/8/8/8/8/Q3K3 w - - 0 1",
    "k7/8/8/8/8/8/8/4K2Q b - - 0 1",
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
    "8/8/8/8/8/8/1k6/K7 w - - 0 1",
    "8/8/8/8/8/8/1kp5/K7 w - - 0 1",
    "8/8/8/8/8/8/1kq5/K7 w - - 0 1",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=10_000)
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

    board, st, _ = fb.from_fen(fens[0])
    rounds = 200_000
    started = time.perf_counter()
    for _ in range(rounds):
        fs.evaluate(board, st, pst, weights)
    elapsed = time.perf_counter() - started
    print(f"  compiled evaluate from python: {rounds / elapsed / 1e6:5.2f} M calls/s")
    reference = chess.Board(fens[0])
    rounds = 20_000
    started = time.perf_counter()
    for _ in range(rounds):
        agent.evaluate(reference)
    elapsed = time.perf_counter() - started
    print(f"  agent.evaluate:                {rounds / elapsed / 1e3:5.1f} k calls/s")
    print("\nEverything matches agent.py.")


if __name__ == "__main__":
    main()
