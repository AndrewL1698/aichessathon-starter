"""Prove `fastboard` against python-chess. `uv run python -m tests.test_fastboard [--full]`.

Three things are checked, because a movegen bug that survives perft is a bug that loses games:
perft against the published counts, every legal move of thousands of random positions against
python-chess move for move and fen for fen, and the incremental Zobrist key against a
from-scratch recomputation after every make and every unmake.
"""

import argparse
import random
import time

import chess
import numpy as np
from numba import njit
from numba import types as nbt

import fastboard as fb
from harness.rules import OPENINGS

PERFT_SUITE: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("start", fb.START_FEN, (20, 400, 8902, 197281, 4865609)),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        (48, 2039, 97862, 4085603),
    ),
    ("position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", (14, 191, 2812, 43238, 674624)),
    (
        "position 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        (6, 264, 9467, 422333),
    ),
    (
        "position 4 mirrored",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        (6, 264, 9467, 422333),
    ),
    (
        "position 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        (44, 1486, 62379, 2103487),
    ),
    (
        "position 6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        (46, 2079, 89890, 3894594),
    ),
)

DEEP_SUITE: tuple[tuple[str, str, int, int], ...] = (
    ("start", fb.START_FEN, 6, 119060324),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        5,
        193690690,
    ),
)


STRESS_SEEDS: tuple[str, ...] = (
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/pp3ppp/8/8/8/8/PPP2PPP/4K3 w - - 0 1",
    "8/1p1k1p1p/8/8/8/8/P1PK1P1P/8 w - - 0 1",
)


class Failure(Exception):
    """A verification step disagreed with python-chess."""


def check_perft(deep: bool) -> None:
    print("perft")
    for name, fen, counts in PERFT_SUITE:
        for depth, want in enumerate(counts, 1):
            start = time.perf_counter()
            got = fb.run_perft(fen, depth)
            elapsed = time.perf_counter() - start
            if got != want:
                raise Failure(f"perft({name}, {depth}) = {got}, want {want}")
            rate = got / elapsed / 1e6 if elapsed > 0 else 0.0
            print(
                f"  {name:<20} depth {depth}  {got:>10,}  "
                f"{elapsed * 1000:7.1f} ms  {rate:5.1f} Mnps"
            )
    if not deep:
        return
    for name, fen, depth, want in DEEP_SUITE:
        start = time.perf_counter()
        got = fb.run_perft(fen, depth)
        elapsed = time.perf_counter() - start
        if got != want:
            raise Failure(f"perft({name}, {depth}) = {got}, want {want}")
        print(
            f"  {name:<20} depth {depth}  {got:>10,}  "
            f"{elapsed:7.2f} s   {got / elapsed / 1e6:5.1f} Mnps"
        )


def positions(rng: random.Random, wanted: int) -> list[str]:
    """Random playouts from the start position and the eight harness openings.

    Two deliberate biases, because uniform random play barely reaches the rules that movegens
    get wrong: a quarter of plies prefer a double pawn push when one is available, and every
    position with a live en passant, a check, or a promotion on offer is kept and mixed back in
    at a third of the sample. Unbiased, a 3,000 position run held ten en passant positions.
    `STRESS_SEEDS` add promotion, castling and pawn-ending material to the eight openings.
    """
    seeds = [fb.START_FEN, *(fen for _, fen in OPENINGS)]
    pool = seeds + list(STRESS_SEEDS)
    plain: list[str] = []
    special: list[str] = []
    quota = wanted // 3
    while len(plain) < wanted or len(special) < quota:
        board = chess.Board(rng.choice(pool))
        for _ in range(rng.randint(1, 90)):
            moves = list(board.legal_moves)
            if not moves:
                break
            fen = board.fen()
            interesting = (
                board.has_legal_en_passant()
                or board.is_check()
                or any(move.promotion for move in moves)
            )
            (special if interesting else plain).append(fen)
            pushes = [move for move in moves if _is_double_push(board, move)]
            board.push(rng.choice(pushes) if pushes and rng.random() < 0.25 else rng.choice(moves))
    rng.shuffle(plain)
    rng.shuffle(special)
    return plain[: wanted - quota] + special[:quota]


def _is_double_push(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    return abs(move.to_square - move.from_square) == 16


def check_round_trip(board: np.ndarray, st: np.ndarray, whence: str) -> None:
    """Re-parsing our own fen must land on the same state, key included.

    This is the assertion that catches a position holding two Zobrist keys. The harness hands
    the agent a fen every move, so the state a search reached by playing d2d4 and the state
    parsed back from that fen have to be the same state, or the transposition table and
    repetition detection index one position twice and neither works.

    `st[8]`, the undo pointer, is deliberately excluded: it is the search's own bookkeeping and
    counts the moves still on the stack, not anything about the position.
    """
    parsed_board, parsed_st, _ = fb.from_fen(fb.to_fen(board, st))
    if not np.array_equal(parsed_board, board):
        raise Failure(f"round trip changed the board {whence}")
    if not np.array_equal(parsed_st[:8], st[:8]):
        raise Failure(
            f"round trip changed the state {whence}: {list(st[:8])} became {list(parsed_st[:8])}"
        )
    if int(parsed_st[7]) != int(st[7]):
        raise Failure(f"round trip changed the key {whence}")


PINNED_EP = "8/8/8/8/k1p4R/8/3P4/4K3 w - - 0 1"
UNPINNED_EP = "8/8/8/8/2p4R/k7/3P4/4K3 w - - 0 1"


def check_pinned_ep() -> None:
    """A double push whose only en passant answer is illegal must set no ep square.

    White plays d2d4 beside a black pawn on c4. In `PINNED_EP` the reply c4xd3 would clear both
    pawns off the fourth rank and expose the black king on a4 to the rook on h4, so there is no
    legal en passant and python-chess prints none; in `UNPINNED_EP` the king stands on a3 and
    the capture is legal. Storing the square on mere adjacency gave the first position one key
    inside the search and a different one after the fen went out to the harness and came back.
    """
    for fen, expected in ((PINNED_EP, False), (UNPINNED_EP, True)):
        board, st, undo = fb.from_fen(fen)
        fb.make_move(board, st, undo, fb.uci_to_move(board, st, undo, "d2d4"))
        reference = chess.Board(fen)
        reference.push(chess.Move.from_uci("d2d4"))
        if bool(st[2]) is not expected:
            raise Failure(f"ep square {int(st[2])} after d2d4 from {fen}, expected {expected}")
        if bool(st[2]) is not reference.has_legal_en_passant():
            raise Failure(f"ep square disagrees with python-chess after d2d4 from {fen}")
        if fb.to_fen(board, st) != reference.fen():
            raise Failure(f"{fb.to_fen(board, st)!r} != {reference.fen()!r}")
        check_round_trip(board, st, f"after d2d4 from {fen}")
    print("  a pinned en passant sets no ep square and no ep key; a legal one sets both")


def check_guards() -> None:
    """The malformed input and overflow guards fail loud rather than corrupting the board."""
    for fen, reason in (
        ("ppppppppp/8/8/8/8/8/8/4K3 w - - 0 1", "nine entries in a rank"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", "only three fields"),
        ("8/8/8/8/8/8/8/4K3 w - - 0 1", "no black king"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", "no side to move"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq j9 0 1", "a bogus ep square"),
    ):
        try:
            fb.from_fen(fen)
        except ValueError:
            continue
        raise Failure(f"from_fen accepted {reason}: {fen}")

    phantom = "r3k2r/8/8/8/8/8/8/4K2R w KQkq - 0 1"
    board, st, _ = fb.from_fen(phantom)
    if fb.to_fen(board, st) != chess.Board(phantom).fen():
        raise Failure(f"phantom rights: {fb.to_fen(board, st)!r} != {chess.Board(phantom).fen()!r}")

    board, st, undo = fb.from_fen(fb.START_FEN)
    st[8] = undo.shape[0]
    try:
        fb.make_move(board, st, undo, fb.legal_moves(board, st, undo)[0])
    except IndexError:
        pass
    else:
        raise Failure("a full undo stack did not raise")

    board, st, undo = fb.from_fen(fb.START_FEN)
    try:
        fb.gen_moves(board, st, np.zeros(20, dtype=np.int32))
    except IndexError:
        pass
    else:
        raise Failure("an undersized move buffer did not raise")
    print("  malformed fens, a full undo stack and a short move buffer all raise")


def check_move_ceiling() -> None:
    """The widest position anyone has constructed, against MAX_MOVES."""
    widest = "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1"
    board, st, undo = fb.from_fen(widest)
    buffer = fb.move_buffer()
    pseudo = fb.gen_moves(board, st, buffer)
    legal = fb.gen_legal(board, st, undo, buffer)
    reference = len(list(chess.Board(widest).legal_moves))
    if legal != reference:
        raise Failure(f"{legal} legal moves, python-chess says {reference}")
    print(f"  widest known position: {pseudo} pseudo-legal, {legal} legal, buffer {fb.MAX_MOVES}")


def check_positions(fens: list[str]) -> dict[str, int]:
    """Every legal move, fen, and Zobrist key of each position, against python-chess."""
    tally = {
        "positions": 0,
        "moves": 0,
        "ep": 0,
        "ep captures": 0,
        "castling": 0,
        "castles": 0,
        "promotions": 0,
        "checks": 0,
        "widest": 0,
    }
    scratch = fb.move_buffer()
    for fen in fens:
        reference = chess.Board(fen)
        board, st, undo = fb.from_fen(fen)

        mine = {fb.move_to_uci(move) for move in fb.legal_moves(board, st, undo)}
        theirs = {move.uci() for move in reference.legal_moves}
        if mine != theirs:
            raise Failure(
                f"{fen}\n  only mine: {sorted(mine - theirs)}"
                f"\n  only theirs: {sorted(theirs - mine)}"
            )

        if fb.to_fen(board, st) != reference.fen():
            raise Failure(f"fen {fb.to_fen(board, st)!r} != {reference.fen()!r}")
        if int(st[7]) != int(fb.compute_key(board, st)):
            raise Failure(f"key out of step at {fen}")
        check_round_trip(board, st, f"at {fen}")

        tally["widest"] = max(tally["widest"], fb.gen_moves(board, st, scratch))
        tally["positions"] += 1
        tally["ep"] += reference.ep_square is not None
        tally["castling"] += bool(reference.castling_rights)
        tally["checks"] += reference.is_check()

        before_board, before_st = board.copy(), st.copy()
        for move in fb.legal_moves(board, st, undo):
            uci = fb.move_to_uci(move)
            tally["moves"] += 1
            tally["promotions"] += len(uci) == 5
            tally["ep captures"] += bool(move & fb.FLAG_EP)
            tally["castles"] += bool(move & fb.FLAG_CASTLE)

            fb.make_move(board, st, undo, move)
            if int(st[7]) != int(fb.compute_key(board, st)):
                raise Failure(f"incremental key wrong after {uci} from {fen}")
            reference.push(chess.Move.from_uci(uci))
            if fb.to_fen(board, st) != reference.fen():
                raise Failure(
                    f"after {uci} from {fen}: "
                    f"{fb.to_fen(board, st)!r} != {reference.fen()!r}"
                )
            reference.pop()
            check_round_trip(board, st, f"after {uci} from {fen}")
            fb.unmake_move(board, st, undo, move)

            if not np.array_equal(board, before_board) or not np.array_equal(st, before_st):
                raise Failure(f"unmake of {uci} from {fen} did not restore the state")
            if int(st[7]) != int(fb.compute_key(board, st)):
                raise Failure(f"incremental key wrong after unmaking {uci} from {fen}")
    return tally


def check_attacks(fens: list[str], squares_for: int) -> int:
    """`is_square_attacked` against python-chess, on kings and then on every square.

    The king square alone only ever exercised the rays that happen to reach a king, so the
    first few hundred positions are checked exhaustively: all 64 squares, both colours.
    """
    for index, fen in enumerate(fens):
        reference = chess.Board(fen)
        board, st, _ = fb.from_fen(fen)
        side = int(st[0])
        if bool(fb.is_square_attacked(board, int(st[5 + side]), 1 - side)) != reference.is_check():
            raise Failure(f"attack detection disagrees at {fen}")
        if bool(fb.in_check(board, st)) != reference.is_check():
            raise Failure(f"in_check disagrees at {fen}")
        if index >= squares_for:
            continue
        for square in chess.SQUARES:
            mailbox = fb.square_index(chess.square_name(square))
            for colour in (0, 1):
                mine = bool(fb.is_square_attacked(board, mailbox, colour))
                theirs = reference.is_attacked_by(colour == 0, square)
                if mine != theirs:
                    raise Failure(
                        f"{chess.square_name(square)} attacked by "
                        f"{'white' if colour == 0 else 'black'}: {mine} != {theirs} at {fen}"
                    )
    return len(fens)


def check_speed() -> None:
    kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    start = time.perf_counter()
    nodes = fb.run_perft(kiwipete, 4)
    elapsed = time.perf_counter() - start
    print(f"  perft(kiwipete, 4)           {nodes / elapsed / 1e6:6.2f} M nodes/s")

    board, st, undo = fb.from_fen(kiwipete)
    buffer = fb.move_buffer()
    _churn(board, st, undo, buffer, 1)
    rounds = 2_000_000
    start = time.perf_counter()
    generated = _churn(board, st, undo, buffer, rounds)
    elapsed = time.perf_counter() - start
    print(
        f"  gen_legal + make/unmake      {rounds / elapsed / 1e6:6.2f} M nodes/s, "
        f"{generated / elapsed / 1e6:6.2f} M legal moves/s (jitted)"
    )

    rounds = 200_000
    start = time.perf_counter()
    for _ in range(rounds):
        fb.gen_legal(board, st, undo, buffer)
    elapsed = time.perf_counter() - start
    print(f"  gen_legal from python         {rounds / elapsed / 1e6:6.2f} M calls/s (root only)")


@njit(
    nbt.int64(nbt.int8[::1], nbt.int64[::1], nbt.int64[:, ::1], nbt.int32[::1], nbt.int64),
    cache=False,
)
def _churn(
    board: np.ndarray, st: np.ndarray, undo: np.ndarray, buffer: np.ndarray, rounds: int
) -> int:
    """What a search's inner loop costs: generate legal moves, play one, take it back."""
    total = 0
    for _ in range(rounds):
        count = fb.gen_legal(board, st, undo, buffer)
        total += count
        move = buffer[0]
        fb.make_move(board, st, undo, move)
        fb.unmake_move(board, st, undo, move)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="deep perfts and 4x the positions")
    arguments = parser.parse_args()

    check_perft(arguments.full)
    print("\nregressions")
    check_pinned_ep()
    check_guards()
    check_move_ceiling()

    rng = random.Random(20260908)
    wanted = 12_000 if arguments.full else 3_000
    fens = positions(rng, wanted)
    print(f"\ndifferential against python-chess, {len(fens):,} positions")
    start = time.perf_counter()
    tally = check_positions(fens)
    print(f"  {tally['positions']:,} positions, {tally['moves']:,} moves checked, "
          f"{time.perf_counter() - start:.1f} s")
    print(f"    widest      {tally['widest']} pseudo-legal moves, buffer holds {fb.MAX_MOVES}")
    for name in ("ep", "ep captures", "castling", "castles", "promotions", "checks"):
        print(f"    {name:<12} {tally[name]:,}")

    squares_for = 500
    checked = check_attacks(fens[:2000], squares_for)
    print(
        f"  attack detection agrees on {checked:,} king squares and "
        f"every square of the first {squares_for:,} positions, both colours"
    )

    print("\nspeed")
    check_speed()
    print("\nEverything matches python-chess.")


if __name__ == "__main__":
    main()
