"""A chess agent: iterative deepening alpha-beta negamax with quiescence over a material and
piece-square evaluation."""

import time
import traceback
from dataclasses import dataclass

import chess

# Scores are centipawns: a pawn is worth 100 points. MATE is deliberately much larger than
# every possible material and positional score, and INFINITY sits one above it so that a
# forced mate still fits inside an alpha-beta window.
MATE = 1_000_000
INFINITY = MATE + 1
# Any score this large can only be a mate score, so the iteration loop can stop.
MATE_FOUND = MATE - 1_000

MAX_DEPTH = 64
# Quiescence is bounded so a long capture chain, or a run of checks, cannot explode.
QUIESCENCE_MAX_PLY = 8
# The clock is read once every 1024 nodes; reading it per node costs more than it saves.
NODE_CHECK_MASK = 1023

# Budgets in milliseconds, all derived from the clock we were handed, never from a constant.
SOFT_DIVISOR = 25
SOFT_BONUS_MS = 400
HARD_MULTIPLIER = 3
HARD_DIVISOR = 8
# The referee times us from when it sends the request, so process overhead is on our clock.
SAFETY_MARGIN_MS = 300
MINIMUM_BUDGET_MS = 10.0
# Below this the clock is nearly gone: search one ply plus quiescence and reply immediately.
# It is deliberately low. The hard budget already caps a move at an eighth of the clock, so
# this only has to cover the last seconds; set at 5 s it fires for most of a 10 s game, and a
# depth-1 engine cannot see a stalemate at all, because quiescence never generates quiet moves.
PANIC_MS = 1_000
# An iteration that starts this late into the soft budget will not finish, so do not start it.
START_FRACTION = 0.5

# Ordering keys. Captures outrank promotions, which outrank quiet moves.
CAPTURE_BONUS = 1_000_000
PROMOTION_BONUS = 500_000

PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

type PieceSquareTable = tuple[tuple[int, ...], ...]

# Each row is one rank, starting at White's home rank. A Black piece looks up the vertically
# mirrored square. The bonuses refine material without being large enough to overpower it.
PAWN_TABLE: PieceSquareTable = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (5, 10, 10, -20, -20, 10, 10, 5),
    (5, -5, -10, 0, 0, -10, -5, 5),
    (0, 0, 0, 20, 20, 0, 0, 0),
    (5, 5, 10, 25, 25, 10, 5, 5),
    (10, 10, 20, 30, 30, 20, 10, 10),
    (50, 50, 50, 50, 50, 50, 50, 50),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

KNIGHT_TABLE: PieceSquareTable = (
    (-50, -40, -30, -30, -30, -30, -40, -50),
    (-40, -20, 0, 5, 5, 0, -20, -40),
    (-30, 5, 10, 15, 15, 10, 5, -30),
    (-30, 0, 15, 20, 20, 15, 0, -30),
    (-30, 5, 15, 20, 20, 15, 5, -30),
    (-30, 0, 10, 15, 15, 10, 0, -30),
    (-40, -20, 0, 0, 0, 0, -20, -40),
    (-50, -40, -30, -30, -30, -30, -40, -50),
)

BISHOP_TABLE: PieceSquareTable = (
    (-20, -10, -10, -10, -10, -10, -10, -20),
    (-10, 5, 0, 0, 0, 0, 5, -10),
    (-10, 10, 10, 10, 10, 10, 10, -10),
    (-10, 0, 10, 10, 10, 10, 0, -10),
    (-10, 5, 5, 10, 10, 5, 5, -10),
    (-10, 0, 5, 10, 10, 5, 0, -10),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-20, -10, -10, -10, -10, -10, -10, -20),
)

ROOK_TABLE: PieceSquareTable = (
    (0, 0, 0, 5, 5, 0, 0, 0),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (5, 10, 10, 10, 10, 10, 10, 5),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

QUEEN_TABLE: PieceSquareTable = (
    (-20, -10, -10, -5, -5, -10, -10, -20),
    (-10, 0, 5, 0, 0, 0, 0, -10),
    (-10, 5, 5, 5, 5, 5, 0, -10),
    (0, 0, 5, 5, 5, 5, 0, -5),
    (-5, 0, 5, 5, 5, 5, 0, -5),
    (-10, 0, 5, 5, 5, 5, 0, -10),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-20, -10, -10, -5, -5, -10, -10, -20),
)

KING_TABLE: PieceSquareTable = (
    (20, 30, 10, 0, 0, 10, 30, 20),
    (20, 20, 0, 0, 0, 0, 20, 20),
    (-10, -20, -20, -20, -20, -20, -20, -10),
    (-20, -30, -30, -40, -40, -30, -30, -20),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
)

PIECE_SQUARE_TABLES: dict[int, PieceSquareTable] = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


def evaluate(board: chess.Board) -> int:
    """Return a material-and-position score from the side-to-move's perspective."""
    white_score = 0
    for square, piece in board.piece_map().items():
        table_square = square if piece.color == chess.WHITE else chess.square_mirror(square)
        table = PIECE_SQUARE_TABLES[piece.piece_type]
        value = PIECE_VALUES[piece.piece_type]
        value += table[chess.square_rank(table_square)][chess.square_file(table_square)]
        white_score += value if piece.color == chess.WHITE else -value
    return white_score if board.turn == chess.WHITE else -white_score


class _Timeout(Exception):
    """Raised inside the search once the hard deadline has passed."""


@dataclass(slots=True)
class _Search:
    """The little state one search needs. Nothing here survives the move."""

    deadline: float
    nodes: int = 0
    # The best root move of the iteration in progress, or None while its first move is running.
    root_best: chess.Move | None = None


def _budgets(time_left_ms: int) -> tuple[float, float]:
    """Return the soft and hard budgets in milliseconds for a move with this much clock."""
    soft = time_left_ms / SOFT_DIVISOR + SOFT_BONUS_MS
    hard = min(HARD_MULTIPLIER * soft, time_left_ms / HARD_DIVISOR)
    # Never plan to use the last of the clock: the reply still has to travel back.
    hard = max(min(hard, time_left_ms - SAFETY_MARGIN_MS), MINIMUM_BUDGET_MS)
    return min(soft, hard), hard


def _move_score(board: chess.Board, move: chess.Move) -> int:
    """Rank a move for ordering: MVV-LVA captures first, then promotions, then quiet moves."""
    score = 0
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        # An en passant capture lands on an empty square; its victim is always a pawn.
        victim_value = PIECE_VALUES[chess.PAWN] if victim is None else PIECE_VALUES[victim]
        attacker = board.piece_type_at(move.from_square)
        attacker_value = 0 if attacker is None else PIECE_VALUES[attacker]
        # Victim value dominates; the attacker only breaks ties, cheapest attacker first.
        score += CAPTURE_BONUS + victim_value * 100 - attacker_value
    if move.promotion is not None:
        score += PROMOTION_BONUS + PIECE_VALUES[move.promotion]
    return score


def _ordered(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Sort moves so that alpha-beta meets the ones most likely to cut off first."""
    moves.sort(key=lambda move: _move_score(board, move), reverse=True)
    return moves


def _quiescence(
    board: chess.Board, alpha: int, beta: int, ply: int, remaining: int, search: _Search
) -> int:
    """Search the noisy continuations so the evaluation is never read mid-exchange."""
    search.nodes += 1
    if not search.nodes & NODE_CHECK_MASK and time.perf_counter() > search.deadline:
        raise _Timeout

    if board.is_check():
        # In check there is no standing pat, and every evasion has to be looked at. That is
        # also the only place a checkmate can hide down here, and it makes it exact.
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply
        if remaining == 0:
            return evaluate(board)
        best = -INFINITY
    else:
        best = evaluate(board)
        if best >= beta or remaining == 0:
            return best
        alpha = max(alpha, best)
        # Captures, plus the quiet promotions, which change material as much as a capture does.
        # A stalemate here scores as the stand-pat instead of 0; proving it costs a full move
        # generation at every quiet leaf, which is far more than the rare error is worth.
        moves = list(board.generate_legal_captures())
        moves += board.generate_legal_moves(board.pawns, chess.BB_BACKRANKS & ~board.occupied)

    for move in _ordered(board, moves):
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, ply + 1, remaining - 1, search)
        board.pop()
        if score > best:
            best = score
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    return best


def _negamax(
    board: chess.Board, depth: int, ply: int, alpha: int, beta: int, search: _Search
) -> int:
    """Fail-soft alpha-beta by observing that both sides' scores are exact opposites."""
    search.nodes += 1
    if not search.nodes & NODE_CHECK_MASK and time.perf_counter() > search.deadline:
        raise _Timeout
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, QUIESCENCE_MAX_PLY, search)

    # A cheap gate on the expensive call: material is only ever insufficient with no pawn,
    # rook or queen anywhere on the board.
    if not board.pawns | board.rooks | board.queens and board.is_insufficient_material():
        return 0

    moves = list(board.legal_moves)
    if not moves:
        # Mate is scored by ply so that a shorter mate outranks a longer one and we convert.
        return -MATE + ply if board.is_check() else 0

    best = -INFINITY
    for move in _ordered(board, moves):
        board.push(move)
        score = -_negamax(board, depth - 1, ply + 1, -beta, -alpha, search)
        board.pop()
        if score > best:
            best = score
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    return best


def _root(
    board: chess.Board, depth: int, first: chess.Move, search: _Search
) -> tuple[chess.Move, int]:
    """Search every root move at one depth, trying the previous iteration's best move first."""
    search.root_best = None
    moves = _ordered(board, list(board.legal_moves))
    if first in moves:
        moves.remove(first)
        moves.insert(0, first)

    best_move, best_score = moves[0], -INFINITY
    for move in moves:
        board.push(move)
        score = -_negamax(board, depth - 1, 1, -INFINITY, -best_score, search)
        board.pop()
        if score > best_score:
            best_move, best_score = move, score
            # Published for the abort path. Because `first` is searched first, anything that
            # replaces it here has already outscored it at this depth, so a partial iteration
            # only ever hands back a move it has proven better.
            search.root_best = move
    return best_move, best_score


def _think(fen: str, time_left_ms: int) -> str:
    """Deepen until the budget is spent, keeping the best move we have proven so far."""
    started = time.perf_counter()
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"

    soft_ms, hard_ms = _budgets(time_left_ms)
    search = _Search(deadline=started + hard_ms / 1000.0)
    best = _ordered(board, moves)[0]
    best_score = 0
    reached = 0

    for depth in range(1, 2 if time_left_ms < PANIC_MS else MAX_DEPTH + 1):
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if depth > 1 and elapsed_ms > soft_ms * START_FRACTION:
            break
        try:
            best, best_score = _root(board, depth, best, search)
        except _Timeout:
            if search.root_best is not None:
                best = search.root_best
            break
        reached = depth
        if best_score >= MATE_FOUND:
            break  # A forced mate is in hand; searching deeper cannot shorten it.

    spent_ms = (time.perf_counter() - started) * 1000.0
    nps = search.nodes / max(spent_ms, 1.0) * 1000.0
    print(
        f"d{reached} score {best_score:+d} move {best.uci()} nodes {search.nodes} "
        f"nps {nps:.0f} {spent_ms:.0f}ms soft {soft_ms:.0f} hard {hard_ms:.0f} "
        f"clock {time_left_ms}",
        flush=True,
    )
    return best.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move. Nothing raises out of here: a crash loses the game."""
    try:
        return _think(fen, time_left_ms)
    except Exception:
        traceback.print_exc()
    try:
        return next(iter(chess.Board(fen).legal_moves)).uci()
    except Exception:
        traceback.print_exc()
        return "0000"
