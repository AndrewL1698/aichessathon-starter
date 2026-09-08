"""A deterministic two-ply chess agent using negamax and positional evaluation."""

import chess

# Scores are centipawns: a pawn is worth 100 points. MATE is deliberately much larger than
# every possible material and positional score.
MATE = 1_000_000
SEARCH_DEPTH = 2

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


def negamax(board: chess.Board, depth: int, ply: int) -> int:
    """Search both sides by observing that their scores are exact opposites."""
    if board.is_checkmate():
        return -MATE + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        return evaluate(board)

    best_score = -MATE
    for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci()):
        board.push(move)
        score = -negamax(board, depth - 1, ply + 1)
        board.pop()
        best_score = max(best_score, score)
    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    """Return the best legal UCI move after searching our move and the opponent's reply."""
    del time_left_ms  # Fixed-depth search does not need clock management yet.
    board = chess.Board(fen)
    legal_moves = sorted(board.legal_moves, key=lambda move: move.uci())
    if not legal_moves:
        raise ValueError("get_move was called on a position with no legal moves")

    best_move = legal_moves[0]
    best_score = -MATE
    for move in legal_moves:
        board.push(move)
        score = -negamax(board, SEARCH_DEPTH - 1, 1)
        board.pop()
        if score > best_score:
            best_move = move
            best_score = score

    return best_move.uci()
