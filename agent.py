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
# A budget of a few hundred milliseconds is only a handful of those slices, and there
# overshooting one of them is a flag, so a small budget reads the clock eight times as often.
# The reads that adds cost nothing next to the game they save.
FINE_CHECK_MASK = 127
FINE_CHECK_BELOW_MS = 300

# Budgets in milliseconds, all derived from the clock we were handed, never from a constant.
SOFT_DIVISOR = 25
SOFT_BONUS_MS = 400
HARD_DIVISOR = 8
# The referee times us from when it sends the request, so process overhead is on our clock.
SAFETY_MARGIN_MS = 300
# Below this the clock is nearly gone: search one ply plus quiescence and reply immediately.
# One second, not the five this started at: the hard budget already caps a move at an eighth
# of the clock, so panic only has to cover the last moves of a spent clock. At 5 s the arena's
# 10 s control sat across the threshold, about 70% of moves came back at depth 1, and that
# scored 79.7% against minimax where 1 s scores 95.3%.
PANIC_MS = 1_000
# What the next iteration is expected to cost, as a multiple of the last one. With ordering
# this good the ratio is nearer 2 while one move keeps failing high and nearer 8 when the
# window reopens, so the observed ratio is clamped to that range; before two iterations have
# run there is no ratio to observe and we assume the middle of it.
GROWTH_MIN = 2.0
GROWTH_MAX = 8.0
GROWTH_UNKNOWN = 5.0
# Under this many milliseconds the elapsed time is mostly measurement noise, and a rate
# divided out of it says more about the clock than about the search, so we do not print one.
NPS_FLOOR_MS = 5

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
    # Nodes between clock reads. A small budget sets this finer so the abort lands on time.
    check_mask: int = NODE_CHECK_MASK
    nodes: int = 0
    # The best root move of the iteration in progress, or None while its first move is running.
    root_best: chess.Move | None = None


def _budgets(time_left_ms: int) -> tuple[float, float]:
    """Return the soft and hard budgets in milliseconds for a move with this much clock."""
    soft = time_left_ms / SOFT_DIVISOR + SOFT_BONUS_MS
    # Never plan to use the last of the clock: the reply still has to travel back. The margin
    # is taken off last so that it always wins. Under it there is no time to think at all, the
    # budget is zero, and the search aborts at its first clock check, which is the only safe
    # thing to do on a clock that short.
    hard = max(min(time_left_ms / HARD_DIVISOR, time_left_ms - SAFETY_MARGIN_MS), 0.0)
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


def _order(board: chess.Board, moves: list[chess.Move]) -> None:
    """Sort moves in place so alpha-beta meets the ones most likely to cut off first."""
    moves.sort(key=lambda move: _move_score(board, move), reverse=True)


def _quiescence(
    board: chess.Board, alpha: int, beta: int, ply: int, remaining: int, search: _Search
) -> int:
    """Search the noisy continuations so the evaluation is never read mid-exchange."""
    search.nodes += 1
    if not search.nodes & search.check_mask and time.perf_counter() > search.deadline:
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
        # Only the queen: an underpromotion is a way to avoid a stalemate or to fork, and
        # neither is something a search of the noisy moves alone can see.
        moves += [
            move
            for move in board.generate_legal_moves(
                board.pawns, chess.BB_BACKRANKS & ~board.occupied
            )
            if move.promotion == chess.QUEEN
        ]

    _order(board, moves)
    for move in moves:
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
    if not search.nodes & search.check_mask and time.perf_counter() > search.deadline:
        raise _Timeout

    # A cheap gate on the expensive call: material is only ever insufficient with no pawn,
    # rook or queen anywhere on the board. It comes before the depth check because quiescence
    # would score a dead draw from the tables instead, and at depth 1 that is every leaf.
    if not board.pawns | board.rooks | board.queens and board.is_insufficient_material():
        return 0
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, QUIESCENCE_MAX_PLY, search)

    moves = list(board.legal_moves)
    if not moves:
        # Mate is scored by ply so that a shorter mate outranks a longer one and we convert.
        return -MATE + ply if board.is_check() else 0

    best = -INFINITY
    _order(board, moves)
    for move in moves:
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
    moves = list(board.legal_moves)
    _order(board, moves)
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


def _projected(last_ms: float, previous_ms: float) -> float:
    """Estimate what the next iteration costs from the last one and how fast cost is growing."""
    growth = last_ms / previous_ms if previous_ms > 0.0 else GROWTH_UNKNOWN
    return last_ms * min(max(growth, GROWTH_MIN), GROWTH_MAX)


def _think(fen: str, time_left_ms: int) -> str:
    """Deepen until the budget is spent, keeping the best move we have proven so far."""
    started = time.perf_counter()
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"

    soft_ms, hard_ms = _budgets(time_left_ms)
    mask = NODE_CHECK_MASK if hard_ms >= FINE_CHECK_BELOW_MS else FINE_CHECK_MASK
    search = _Search(deadline=started + hard_ms / 1000.0, check_mask=mask)
    _order(board, moves)
    best = moves[0]
    best_score = 0
    reached = 0
    partial = False
    last_ms, previous_ms = 0.0, 0.0

    # A zero budget means the clock is under the safety margin, and then even the first
    # hundred nodes are time we do not have: the ordered first move is the whole reply.
    deepest = 0 if hard_ms <= 0.0 else 1 if time_left_ms < PANIC_MS else MAX_DEPTH
    for depth in range(1, deepest + 1):
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Start an iteration only when the soft budget is projected to fall inside it, not
        # after it, so the average move lands near the budget: an iteration costs several
        # times the one before, so refusing every one that could end past the budget leaves
        # most of the budget unspent. Gating on the time already spent, as this used to,
        # admits an iteration with the whole rest of the hard budget ahead of it, and one
        # that runs to the deadline is wasted entirely.
        if depth > 1 and elapsed_ms + _projected(last_ms, previous_ms) / 2.0 > soft_ms:
            break
        iteration_started = time.perf_counter()
        try:
            best, best_score = _root(board, depth, best, search)
        except _Timeout:
            # The unwind skips every board.pop() of the line being searched, so `board` is
            # left with that line still on it. Nothing below touches the board, and the next
            # move builds a new one from its fen.
            if search.root_best is not None:
                best = search.root_best
                partial = True
            break
        previous_ms = last_ms
        last_ms = (time.perf_counter() - iteration_started) * 1000.0
        reached = depth
        if best_score >= MATE_FOUND:
            break  # A forced mate is in hand; searching deeper cannot shorten it.

    spent_ms = (time.perf_counter() - started) * 1000.0
    # `best_score` and `reached` are the last completed iteration's. After an abort the move
    # is not: it is one the unfinished iteration had already proven better, so say where it
    # came from rather than reading as the score's move.
    depth_text = f"d{reached} score {best_score:+d}" if reached else "d0"
    move_text = f"move {best.uci()}"
    if partial:
        move_text += f" from partial d{reached + 1}"
    rate_text = ""
    if spent_ms >= NPS_FLOOR_MS:
        rate_text = f"nps {search.nodes / spent_ms * 1000.0:.0f} "
    print(
        f"{depth_text} {move_text} nodes {search.nodes} "
        f"{rate_text}{spent_ms:.0f}ms soft {soft_ms:.0f} hard {hard_ms:.0f} "
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
