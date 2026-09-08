"""A chess agent: iterative deepening alpha-beta negamax with quiescence over a material and
piece-square evaluation, remembering what it searched and where the game has been."""

import resource
import sys
import time
import traceback
from collections.abc import Hashable
from dataclasses import dataclass, field

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
# Futility pruning at the frontier: one ply from the leaves, a quiet move is not searched when
# the static evaluation plus this margin cannot reach alpha. A quiet move rarely gains more
# than this, and quiescence would only have looked at the captures after it anyway.
FUTILITY_MARGIN = 150
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

# Ordering keys, highest first: the table's move, then captures, then promotions, then the
# killers, then the quiet moves by how often they have cut off before.
TABLE_BONUS = 4_000_000
CAPTURE_BONUS = 1_000_000
PROMOTION_BONUS = 500_000
KILLER_BONUS = 400_000
# History counts are clamped below both killers so no amount of them ever ties one.
HISTORY_CAP = KILLER_BONUS - 3
HISTORY_SIDE = 64 * 64

# What a stored score says about the true one: it is the score, or a bound on it.
EXACT, LOWER, UPPER = 0, 1, 2
# What python-chess's transposition key is: a tuple, and what we key everything on. See _key.
type _Key = Hashable
# depth, bound, score relative to the node's ply, and the move that was best there.
type _Entry = tuple[int, int, int, chess.Move | None]
# An entry costs about 500 bytes, so half a million of them is 250 MB of the container's two
# gigabytes, which leaves room for an evaluation heavier than this one. The table is cleared
# rather than evicted when it fills: a clear costs one search of refilling and happens a
# handful of times in a long game, while any eviction policy costs something on every store.
TABLE_MAX_ENTRIES = 500_000

# A draw is not worth zero. Above this much advantage a draw is a loss of half a point we had
# in hand, and below minus this much it is half a point rescued.
CONTEMPT = 50
CONTEMPT_THRESHOLD = 150

# The fifty move rule is claimed at a hundred half-moves without a capture or a pawn move.
FIFTY_MOVE_PLIES = 100

# getrusage reports the peak resident set in bytes on macOS and in kilobytes on Linux, which
# is where this actually runs.
RSS_DIVISOR = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0

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
class _Memory:
    """What survives from one of our moves to the next. The process lives for one game."""

    # The transposition table, keyed on the position; see _key.
    table: dict[_Key, _Entry] = field(default_factory=dict)
    # Quiet cutoff counts, indexed by side to move, from-square and to-square.
    history: list[int] = field(default_factory=lambda: [0] * (2 * HISTORY_SIDE))
    # Every position this game has stood in, ours and the opponent's, from the first fen on.
    # Playing Black we never see the opening position itself, because the first fen we are
    # handed is already one ply in; the referee still claims at the third occurrence, so that
    # is one position we are blind to rather than a position we can repeat into a lost draw.
    seen: set[_Key] = field(default_factory=set)
    # The position we handed back last move. The next fen has to be one legal move from it.
    expected: chess.Board | None = None


_MEMORY = _Memory()


@dataclass(slots=True)
class _Search:
    """The little state one search needs. Nothing here survives the move."""

    deadline: float
    # Nodes between clock reads. A small budget sets this finer so the abort lands on time.
    check_mask: int = NODE_CHECK_MASK
    # What a draw is worth to the side to move at the root, so to us. See _draw_score.
    contempt: int = 0
    nodes: int = 0
    cutoffs: int = 0
    # The best root move of the iteration in progress, or None while its first move is running.
    root_best: chess.Move | None = None
    # The keys on the path from the root down to the node being searched. A node that meets a
    # key already on the path returns before pushing it again, so a key is never in here twice.
    path: set[_Key] = field(default_factory=set)
    killers: list[list[chess.Move | None]] = field(
        default_factory=lambda: [[None, None] for _ in range(MAX_DEPTH + 1)]
    )
    # Counts the draw scores handed back by repetition and by the fifty move rule. Those depend
    # on how the position was reached, not only on the position, so a node whose count moved
    # while its children ran is not written to the table.
    draws: int = 0


def _key(board: chess.Board) -> _Key:
    """The position as python-chess describes it, without the move counters.

    The tuple is the key, not its hash. CPython reduces an integer's hash modulo 2**61 - 1, so
    hashing the tuple would fold the top three bits of every bitboard onto the bottom three:
    a rook on h8 would collide with a rook on c1. A dict keyed on the tuples costs about two
    and a half times the memory and 24 ns more per lookup, against the 540 ns a node already
    spends building the key, and it cannot hand back another position's score.
    """
    return board._transposition_key()


def _to_table(score: int, ply: int) -> int:
    """Rewrite a mate score to be counted from this node, not from the root.

    A mate is scored MATE minus the ply it lands on. Stored that way, a hit from a different
    ply would claim the wrong distance, so the depth from the root is added back out here and
    taken off again on the way in.
    """
    if score > MATE_FOUND:
        return score + ply
    if score < -MATE_FOUND:
        return score - ply
    return score


def _from_table(score: int, ply: int) -> int:
    """Undo _to_table for a node at this ply."""
    if score > MATE_FOUND:
        return score - ply
    if score < -MATE_FOUND:
        return score + ply
    return score


def _store(
    key: _Key, depth: int, bound: int, score: int, move: chess.Move | None, ply: int
) -> None:
    """Write what this node proved, keeping the table inside its memory budget."""
    table = _MEMORY.table
    if len(table) >= TABLE_MAX_ENTRIES:
        table.clear()
    table[key] = (depth, bound, _to_table(score, ply), move)


def _contempt(root_score: int) -> int:
    """What a draw is worth to us, given how the root position stands from our side.

    Negative when we are winning, so the search prefers nearly anything to a repetition, and
    positive when we are losing, so it steers into one.
    """
    if root_score > CONTEMPT_THRESHOLD:
        return -CONTEMPT
    if root_score < -CONTEMPT_THRESHOLD:
        return CONTEMPT
    return 0


def _draw_score(ply: int, search: _Search) -> int:
    """Score a draw from the point of view of the side to move at this ply.

    `search.contempt` is the value of a draw to the side that moves at the root, which is us.
    Negamax scores every node from its own mover's side, and the mover at an odd ply is the
    opponent, so the sign flips there: a draw we dislike at -50 is one they like at +50.
    """
    return search.contempt if ply % 2 == 0 else -search.contempt


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


def _order_fully(
    board: chess.Board,
    moves: list[chess.Move],
    table_move: chess.Move | None,
    killers: list[chess.Move | None],
) -> None:
    """Sort a full move list in place, using everything earlier searches learned about it."""
    history = _MEMORY.history
    side = int(board.turn) * HISTORY_SIDE

    def rank(move: chess.Move) -> int:
        if move == table_move:
            return TABLE_BONUS
        score = _move_score(board, move)
        if score:  # non-zero means a capture or a promotion
            return score
        if move == killers[0]:
            return KILLER_BONUS
        if move == killers[1]:
            return KILLER_BONUS - 1
        return min(history[side + move.from_square * 64 + move.to_square], HISTORY_CAP)

    moves.sort(key=rank, reverse=True)


def _remember_cutoff(
    board: chess.Board, move: chess.Move, depth: int, ply: int, search: _Search
) -> None:
    """Record a quiet move that caused a cutoff so its siblings elsewhere are tried after it.

    Captures already order themselves by what they win, so only the quiet moves need this.
    """
    if board.is_capture(move) or move.promotion is not None:
        return
    killers = search.killers[ply]
    if move != killers[0]:
        killers[1] = killers[0]
        killers[0] = move
    # Squared, because a cutoff found deep in the tree stood up to far more refutations than
    # one found at a leaf, and is that much better evidence about the move.
    _MEMORY.history[int(board.turn) * HISTORY_SIDE + move.from_square * 64 + move.to_square] += (
        depth * depth
    )


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

    key = _key(board)
    # A position the game has already stood in, or one already on this path, is a draw: the
    # referee claims the third occurrence, and the second is the move that offers it. Counting
    # two rather than three is the standard simplification, and it is not neutral. Winning, it
    # makes us shy of positions that are not yet drawn, which is what we want. Losing, it lets
    # us bank a half point the opponent has not agreed to give, so we may steer at a draw that
    # is still one move from being refused. A repeated position is never checkmate, since the
    # game would have ended the first time it appeared.
    if key in _MEMORY.seen or key in search.path:
        search.draws += 1
        return _draw_score(ply, search)
    # The fifty move rule does not rescue a side that is being mated: mate ends the game first,
    # so a position with no escape from check is scored below, not here.
    if board.halfmove_clock >= FIFTY_MOVE_PLIES and (
        not board.is_check() or any(board.legal_moves)
    ):
        search.draws += 1
        return _draw_score(ply, search)

    # A cheap gate on the expensive call: material is only ever insufficient with no pawn,
    # rook or queen anywhere on the board. It comes before the depth check because quiescence
    # would score a dead draw from the tables instead, and at depth 1 that is every leaf.
    if not board.pawns | board.rooks | board.queens and board.is_insufficient_material():
        return _draw_score(ply, search)
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, QUIESCENCE_MAX_PLY, search)

    table_move: chess.Move | None = None
    entry = _MEMORY.table.get(key)
    if entry is not None:
        stored_depth, bound, stored_score, table_move = entry
        if stored_depth >= depth:
            score = _from_table(stored_score, ply)
            # An exact score settles the node. A bound only settles it when it already falls
            # outside the window we were asked about.
            if (
                bound == EXACT
                or (bound == LOWER and score >= beta)
                or (bound == UPPER and score <= alpha)
            ):
                return score

    moves = list(board.legal_moves)
    if not moves:
        # Mate is scored by ply so that a shorter mate outranks a longer one and we convert.
        return -MATE + ply if board.is_check() else _draw_score(ply, search)

    draws_before = search.draws
    window_alpha = alpha
    best = -INFINITY
    best_move = moves[0]
    # Only at the frontier, never in check and never around a mate score. At a futile node
    # the best score starts at what the pruning assumed, so a node whose every move was
    # pruned returns that bound and stores nothing tighter than it.
    futile = False
    if depth == 1 and abs(alpha) < MATE_FOUND and not board.is_check():
        futile_score = evaluate(board) + FUTILITY_MARGIN
        if futile_score <= alpha:
            futile = True
            best = futile_score
    search.path.add(key)
    _order_fully(board, moves, table_move, search.killers[ply])
    for move in moves:
        if (
            futile
            and not board.is_capture(move)
            and move.promotion is None
            and not board.gives_check(move)
        ):
            continue
        board.push(move)
        score = -_negamax(board, depth - 1, ply + 1, -beta, -alpha, search)
        board.pop()
        if score > best:
            best, best_move = score, move
            alpha = max(alpha, best)
            if alpha >= beta:
                search.cutoffs += 1
                _remember_cutoff(board, move, depth, ply, search)
                break
    search.path.discard(key)

    # A score that came out of a repetition belongs to the path, not to the position, so it is
    # not written. A stalemate or an insufficient-material node returns before this and is never
    # written either; what is written is an ancestor whose score came up through one. Contempt
    # colours those, and it is re-derived every move, so an entry stored while we were winning
    # and read back while we were losing is out by 2 * CONTEMPT, a whole pawn. Those are
    # properties of the position rather than of the path, and a pawn of error on a drawn line
    # is worth less than the searches the entries save.
    if search.draws == draws_before:
        bound = UPPER if best <= window_alpha else LOWER if best >= beta else EXACT
        _store(key, depth, bound, best, best_move, ply)
    return best


def _root(
    board: chess.Board, depth: int, first: chess.Move, search: _Search
) -> tuple[chess.Move, int]:
    """Search every root move at one depth, trying the previous iteration's best move first."""
    search.root_best = None
    search.path.clear()
    moves = list(board.legal_moves)
    # The root's own killers are always empty: beta is infinity here, so no root move ever cuts
    # off and nothing is ever recorded at ply 0. Passing them keeps the one ordering function.
    _order_fully(board, moves, first, search.killers[0])

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


def _reachable(previous: chess.Board, board: chess.Board) -> bool:
    """Is this position one legal move on from the one we handed back last time?"""
    target = _key(board)
    for move in previous.legal_moves:
        previous.push(move)
        found = _key(previous) == target
        previous.pop()
        if found:
            return True
    return False


def _observe(board: chess.Board) -> None:
    """Fold the position we were handed into the game history, forgetting a game we are not in.

    The first fen we are given is where the game starts for repetition and fifty move purposes,
    so there is nothing to remember before it. A process is meant to live for exactly one game;
    if what arrives is not one legal move on from what we handed back, we are somewhere else
    and everything we remember is about another game.
    """
    previous = _MEMORY.expected
    if previous is None or not _reachable(previous, board):
        _MEMORY.table.clear()
        _MEMORY.seen.clear()
        _MEMORY.history = [0] * len(_MEMORY.history)
    else:
        # Halved rather than kept or cleared: which quiet moves cut off is still largely true
        # two plies later, but it decays, and halving stops a long game's counts running away.
        _MEMORY.history = [count // 2 for count in _MEMORY.history]
    _MEMORY.seen.add(_key(board))


def _think(fen: str, time_left_ms: int) -> str:
    """Deepen until the budget is spent, keeping the best move we have proven so far."""
    started = time.perf_counter()
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"

    _observe(board)
    soft_ms, hard_ms = _budgets(time_left_ms)
    mask = NODE_CHECK_MASK if hard_ms >= FINE_CHECK_BELOW_MS else FINE_CHECK_MASK
    # Contempt is set once, from the static evaluation, and left alone. Reading it off the
    # previous iteration's score would be sharper but it feeds back: a root score that is
    # itself a contempt-flavoured draw drags contempt to zero, which makes the same draw
    # acceptable on the next iteration, which is exactly the mistake this is here to stop.
    search = _Search(
        deadline=started + hard_ms / 1000.0, check_mask=mask, contempt=_contempt(evaluate(board))
    )
    _order(board, moves)
    best = moves[0]
    # What the table already knows about this position, most likely from the search two plies
    # ago, is better ordering than anything else we have before the first iteration runs.
    entry = _MEMORY.table.get(_key(board))
    if entry is not None:
        stored_move = entry[3]
        if stored_move is not None and stored_move in moves:
            best = stored_move
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
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_DIVISOR
    print(
        f"{depth_text} {move_text} nodes {search.nodes} "
        f"{rate_text}{spent_ms:.0f}ms soft {soft_ms:.0f} hard {hard_ms:.0f} "
        f"clock {time_left_ms} tt {len(_MEMORY.table)} cut {search.cutoffs} "
        f"contempt {search.contempt:+d} peakrss {rss_mb:.0f}MB",
        flush=True,
    )

    # We are handed the position after the opponent's reply next, so both this position and the
    # one we are about to make are part of the game's history. The abort above leaves `board`
    # mid-line, as it says, so this starts again from the fen.
    after = chess.Board(fen)
    after.push(best)
    _MEMORY.seen.add(_key(after))
    _MEMORY.expected = after
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
