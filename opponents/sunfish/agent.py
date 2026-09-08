"""Sunfish as a local benchmarking opponent, wrapped in the platform's agent contract.

Sunfish (github.com/thomasahle/sunfish, GPL-3) is not vendored here. Run
`opponents/fetch_sunfish.sh` to drop `sunfish.py` beside this file; the harness puts the agent
directory first on sys.path, so `import sunfish` then finds it.

Sunfish keeps a 120 character padded board that is always seen from the side to move, so a
position with Black to move is the board turned 180 degrees and case swapped. Everything
below is the translation between that and a FEN.
"""

import time
from pathlib import Path

import chess

try:
    import sunfish
except ImportError as error:  # pragma: no cover - a missing download, not a code path
    raise ImportError(
        f"sunfish.py is not in {Path(__file__).parent}. Run opponents/fetch_sunfish.sh"
    ) from error

# Sunfish squares: a1 is 91 and rank 8 is ten lower per rank, inside a 12x10 padded board.
A1 = sunfish.A1
BORDER = " " * 9 + "\n"

# What one move may spend. The bonus has to shrink with the clock: a flat one never falls under
# the increment, so every move costs more than it earns and the clock ratchets down to a flag.
# Shrinking it means the budget crosses the increment somewhere and the clock parks there
# instead, at about 1.9 s under a 100 ms increment and 9.4 s under the platform's 500 ms.
BUDGET_DIVISOR = 30
BONUS_DIVISOR = 50
BONUS_CAP_MS = 300
RESERVE_MS = 1000
MIN_BUDGET_MS = 20

# Sunfish looks at its deadline every 2048 nodes, so it always runs a little past. Hand it a
# deadline that much earlier than the budget. The overrun is bounded by what 2048 nodes cost,
# so on a short budget it is a fraction of the budget rather than the full allowance.
OVERSHOOT_MS = 100
OVERSHOOT_SHARE = 4


def square_index(square: int) -> int:
    """Return the Sunfish board index of a python-chess square, seen from White."""
    return A1 + chess.square_file(square) - 10 * chess.square_rank(square)


def to_position(board: chess.Board) -> "sunfish.Position":
    """Build the Sunfish position for `board`, rotated when Black is to move."""
    rows = [BORDER, BORDER]
    for rank in range(7, -1, -1):
        squares = (board.piece_at(chess.square(file, rank)) for file in range(8))
        rows.append(" " + "".join(piece.symbol() if piece else "." for piece in squares) + "\n")
    rows += [BORDER, BORDER]
    squares_string = "".join(rows)

    # search() picks the king table from the queens on the board, and the score it starts from
    # has to be the one that table gives, so choose it here too rather than let them disagree.
    sunfish.pst["K"] = (
        sunfish.K_MID if "Q" in squares_string and "q" in squares_string else sunfish.K_END
    )
    score = 0
    for index, symbol in enumerate(squares_string):
        if symbol in "PNBRQK":
            score += sunfish.pst[symbol][index]
        elif symbol in "pnbrqk":
            score -= sunfish.pst[symbol.upper()][119 - index]

    # A pair is (the rook on that side's A1 corner, the rook on its H1 corner) as that side sees
    # its own board. rotate() turns the board 180 degrees rather than mirroring it, so Black's
    # A1 corner is h8 and its king side rook is the one that comes first. Getting this backwards
    # is invisible while both sides still hold both rights.
    white = (
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.WHITE),
    )
    black = (
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    ep = square_index(board.ep_square) if board.ep_square is not None else 0

    # kp is only the square a king just castled through, so a position read from a FEN has none.
    position = sunfish.Position(squares_string, score, white, black, ep, 0)
    return position.rotate() if board.turn == chess.BLACK else position


def budget_ms(time_left_ms: int) -> float:
    """Return what this move may cost end to end, always leaving the clock a reserve."""
    bonus = min(BONUS_CAP_MS, time_left_ms / BONUS_DIVISOR)
    wanted = time_left_ms / BUDGET_DIVISOR + bonus
    ceiling = max(time_left_ms - RESERVE_MS, MIN_BUDGET_MS)
    return max(min(wanted, ceiling), MIN_BUDGET_MS)


def think_s(time_left_ms: int) -> float:
    """Return how long Sunfish may search, which is the budget less its own overrun."""
    budget = budget_ms(time_left_ms)
    return (budget - min(OVERSHOOT_MS, budget / OVERSHOOT_SHARE)) / 1000.0


def best_move(position: "sunfish.Position", deadline: float) -> "sunfish.Move | None":
    """Run iterative deepening to `deadline` and return the last completed depth's move."""
    searcher = sunfish.Searcher()
    # One deadline for both limits: a depth that has finished is what we play, and the search
    # is cut off mid-depth rather than allowed to start one it cannot pay for.
    searcher.deadline = searcher.soft = deadline
    completed: sunfish.Move | None = None
    candidate: sunfish.Move | None = None
    depth_started = 1
    try:
        for depth, gamma, score, move in searcher.search([position]):
            if depth > depth_started:
                completed, depth_started = candidate or completed, depth
            if score >= gamma:
                if move is None:
                    break
                candidate = move
    except sunfish.Stop:
        candidate = completed or candidate
    return candidate or completed


def to_uci(move: "sunfish.Move", black_to_move: bool) -> str:
    """Turn a Sunfish move back into UCI, undoing the rotation when Black is to move."""
    start, end = move.i, move.j
    if black_to_move:
        start, end = 119 - start, 119 - end
    return sunfish.render(start) + sunfish.render(end) + move.prom.lower()


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    deadline = time.time() + think_s(time_left_ms)
    move = best_move(to_position(board), deadline)
    if move is not None:
        uci = to_uci(move, board.turn == chess.BLACK)
        if chess.Move.from_uci(uci) in board.legal_moves:
            return uci
        print(f"sunfish returned {uci}, which is not legal in {fen}", flush=True)
    else:
        print(f"sunfish found no move in {fen}", flush=True)
    return next(iter(board.legal_moves)).uci()
