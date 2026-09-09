"""Prove `fasteval` equals the evaluation it was ported from, exactly.

`uv run python -m tests.test_fasteval [--full]`.

`fasteval.evaluate` is a rewrite of `agent.py`'s `evaluate` over a different board
representation, and the only useful standard for a rewrite like that is identity: for the same
position the two must return the same integer, not a similar one. A one-centipawn difference is
not a tuning choice, it is a mask or a sign that did not survive the move to mailbox indices,
and it will show up as a lost endgame long before it shows up as a smaller number.

So this compares them on thousands of positions drawn the same way `test_fastboard` draws them,
plus endgames, promotions and bare kings, which is where the terms that only fire near the end
of the game live. It also asserts that every table and constant in `fasteval` is equal to the
one in `agent.py`, so the "copied verbatim" in that file's header is a checked claim rather
than a comment.

The reference is imported here and nowhere else. `fasteval` never imports python-chess.
"""

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import chess

import fastboard as fb
import fasteval as fe
from tests.test_fastboard import STRESS_SEEDS, positions

# Endgames are where mop-up, the drawish scaling and the bare-minor zero are the whole score,
# and random play from the openings almost never reaches one. These are seeded in directly.
ENDGAME_SEEDS: tuple[str, ...] = (
    "8/8/8/4k3/8/8/8/R3K3 w - - 0 1",  # KRvK, the position mop-up exists for
    "8/8/8/3k4/8/8/8/3KQ3 w - - 0 1",  # KQvK
    "8/8/8/8/8/5k2/8/5KB1 w - - 0 1",  # KBvK, the dead draw that must score zero
    "8/8/8/8/8/5k2/8/N4K2 w - - 0 1",  # KNvK
    "8/8/8/8/8/4bk2/8/N4K2 w - - 0 1",  # KNvKB, drawish scaling with no pawns
    "8/8/8/8/8/3rk3/8/3BK3 w - - 0 1",  # rook against a bishop, a book draw
    "8/8/8/8/8/2rrk3/8/3QK3 w - - 0 1",  # two rooks against a queen
    "8/4k3/8/8/8/8/4P3/4K3 w - - 0 1",  # KPvK
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",  # promotions on both sides
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",  # bare kings
    "8/1p1p1p2/8/8/8/8/1P1P1P2/4K1k1 w - - 0 1",  # doubled and isolated pawn structure
    "6k1/5ppp/8/8/8/8/PPP5/6K1 w - - 0 1",  # king shields, one covered side each
    "6k1/8/8/8/8/8/PPP5/6K1 w - - 0 1",  # one king covered, one bare: the shield sign test
    "3rk3/8/8/8/8/8/8/3RK3 w - - 0 1",  # rooks on an open file, both sides
    "r3k3/pppppppp/8/8/8/8/PPPPPPPP/R3K3 w Qq - 0 1",  # rooks on a closed file
    "8/2k5/8/8/8/8/2K5/8 w - - 0 1",
    "8/8/1k6/8/8/6K1/8/8 w - - 0 1",
)


class Failure(Exception):
    """The port disagreed with the evaluation it was ported from."""


def load_reference() -> ModuleType:
    """Import the `evaluate` this file is a port of, from wherever it currently lives.

    Once the evaluation PR has merged it is in this worktree's own `agent.py`. Until then it is
    only in the worktree that PR is being written in, so that is the fallback. Loading the
    wrong one, the material-only `evaluate` that came before it, would make this whole file
    pass against the wrong reference, so the module is checked for a tapered evaluation before
    it is accepted and the failure says which paths were tried.
    """
    root = Path(__file__).resolve().parent.parent
    candidates = [root / "agent.py", root.parent / "phase0-eval" / "agent.py"]
    tried = []
    for path in candidates:
        tried.append(str(path))
        if not path.is_file() or "PHASE_MAX" not in path.read_text():
            continue
        spec = importlib.util.spec_from_file_location("_eval_reference", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        # Not registered in sys.modules under "agent": the reference is a second copy of a
        # module this repo also ships, and shadowing the real one would be a trap later.
        spec.loader.exec_module(module)
        return module
    raise Failure(
        "no tapered evaluation to compare against; looked for PHASE_MAX in " + ", ".join(tried)
    )


def check_constants(reference: ModuleType) -> int:
    """Every weight in `fasteval` has to be the number `agent.py` defines, not a near one."""
    checked = 0
    tables = (
        "PAWN_MG", "PAWN_EG", "KNIGHT_MG", "KNIGHT_EG", "BISHOP_MG", "BISHOP_EG",
        "ROOK_MG", "ROOK_EG", "QUEEN_MG", "QUEEN_EG", "KING_MG", "KING_EG",
    )
    scalars = (
        "PHASE_ROOK", "PHASE_QUEEN", "PHASE_MAX", "PASSED_MG", "PASSED_EG",
        "ISOLATED_MG", "ISOLATED_EG", "DOUBLED_MG", "DOUBLED_EG",
        "ROOK_OPEN_MG", "ROOK_OPEN_EG", "ROOK_SEMI_OPEN_MG", "ROOK_SEMI_OPEN_EG",
        "BISHOP_PAIR_MG", "BISHOP_PAIR_EG", "SHIELD_PENALTY", "SHIELD_MAX_COVER",
        "MOP_UP_CMD", "MOP_UP_CLOSE", "MOP_UP_LOOSE_CMD", "MOP_UP_LOOSE_CLOSE",
        "MOP_UP_MIN_ADVANTAGE", "MOP_UP_MAX_WEAK_PIECES", "MOP_UP_BARE_PIECES",
        "DRAWISH_MARGIN", "STALEMATE_PIECE_LIMIT",
    )
    for name in tables + scalars:
        theirs = getattr(reference, name)
        ours = getattr(fe, name)
        if ours != theirs:
            raise Failure(f"{name} differs: fasteval has {ours!r}, agent.py has {theirs!r}")
        checked += 1
    # PIECE_VALUES is a dict keyed on python-chess piece types there and a tuple here.
    for kind in range(1, 7):
        if fe.PIECE_VALUES[kind - 1] != reference.PIECE_VALUES[kind]:
            raise Failure(f"PIECE_VALUES[{kind}] differs")
        checked += 1
    return checked


def check_positions(fens: list[str], reference: ModuleType) -> dict[str, int]:
    """Compare the two evaluations position by position, reporting what the sample covered."""
    tally = {"positions": 0, "endgames": 0, "no pawns": 0, "bare king": 0, "promotions": 0}
    for fen in fens:
        board, st, _ = fb.from_fen(fen)
        got = int(fe.evaluate(board, st))
        reference_board = chess.Board(fen)
        want = int(reference.evaluate(reference_board))
        if got != want:
            raise Failure(f"evaluate({fen!r}) = {got}, agent.py says {want}")
        tally["positions"] += 1
        if reference.PHASE_MAX > 0 and reference._phase(reference_board) <= 6:
            tally["endgames"] += 1
        if not reference_board.pawns:
            tally["no pawns"] += 1
        if min(
            chess.popcount(reference_board.occupied_co[chess.WHITE]),
            chess.popcount(reference_board.occupied_co[chess.BLACK]),
        ) <= 3:
            tally["bare king"] += 1
        if any(move.promotion for move in reference_board.legal_moves):
            tally["promotions"] += 1
    return tally


def check_mirror(fens: list[str]) -> int:
    """Mirroring a position vertically and swapping colours must negate nothing.

    Both sides are scored from the mover's own point of view, so the mirror of a position has
    the *same* score, not the opposite one. This catches an asymmetric table or a term that
    reads one colour's mask for the other, which the reference comparison would also catch
    only if the sample happened to contain the position; here every position is such a test.
    """
    checked = 0
    for fen in fens:
        board, st, _ = fb.from_fen(fen)
        mirrored = chess.Board(fen).mirror()
        # The mirror's move counters and castling rights follow; only the halfmove clock,
        # which the evaluation never reads, is arbitrary.
        board_m, st_m, _ = fb.from_fen(mirrored.fen())
        ours = int(fe.evaluate(board, st))
        theirs = int(fe.evaluate(board_m, st_m))
        if ours != theirs:
            raise Failure(f"mirror of {fen!r} scores {theirs}, original scores {ours}")
        checked += 1
    return checked


def check_seeds() -> None:
    """A seed with the wrong side in check lets python-chess capture a king.

    That silently produces positions with one king on the board, which `fastboard.from_fen`
    rejects and which nothing here is meant to evaluate, so a bad seed has to fail here rather
    than a thousand positions later as a confusing parse error.
    """
    for fen in ENDGAME_SEEDS + STRESS_SEEDS:
        board = chess.Board(fen)
        if not board.is_valid():
            raise Failure(f"seed {fen!r} is not a legal position: {board.status()!r}")


def endgame_positions(rng: random.Random, wanted: int) -> list[str]:
    """Random playouts from the endgame seeds, so the late terms get a real sample."""
    fens: list[str] = []
    while len(fens) < wanted:
        board = chess.Board(rng.choice(ENDGAME_SEEDS + STRESS_SEEDS))
        for _ in range(rng.randint(1, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            fens.append(board.fen())
            board.push(rng.choice(moves))
    rng.shuffle(fens)
    return fens[:wanted]


def check_speed() -> None:
    """A leaf evaluation has to be cheap enough that a million of them a second is possible."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board, st, _ = fb.from_fen(fen)
    fe.evaluate(board, st)
    rounds = 300_000
    start = time.perf_counter()
    for _ in range(rounds):
        fe.evaluate(board, st)
    elapsed = time.perf_counter() - start
    # The loop is Python calling into a jitted function, so the per-call overhead here is
    # mostly the call itself; inside the search there is none. This is a floor, not a rate.
    print(f"  {rounds / elapsed / 1e6:.2f}M evaluations/s from Python (a floor: call overhead)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="10,000 positions instead of 2,000")
    arguments = parser.parse_args()

    check_seeds()
    reference = load_reference()
    print(f"reference evaluation loaded from {reference.__file__}")
    print(f"constants: {check_constants(reference)} tables and weights match agent.py")

    rng = random.Random(0xE7A1)
    wanted = 10_000 if arguments.full else 2_000
    played = positions(rng, wanted // 2)
    endings = endgame_positions(rng, wanted - len(played))
    fens = played + endings
    rng.shuffle(fens)

    print(f"\nexact match against agent.py, {len(fens):,} positions")
    start = time.perf_counter()
    tally = check_positions(fens, reference)
    print(f"  {tally['positions']:,} positions in {time.perf_counter() - start:.1f} s")
    for name in ("endgames", "no pawns", "bare king", "promotions"):
        print(f"    {name:<12} {tally[name]:,}")

    print(f"\nmirror symmetry on {check_mirror(fens[:1000]):,} positions")

    print("\nspeed")
    check_speed()
    print("\nfasteval is identical to the evaluation in agent.py.")


if __name__ == "__main__":
    try:
        main()
    except Failure as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
