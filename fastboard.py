"""Numba mailbox chess core: board, pseudo-legal movegen, make/unmake, attacks, Zobrist.

No search and no evaluation live here; this is the substrate they sit on. Everything hot is
`@njit` with an eager signature, so every function compiles at import (inside the platform's
90 second init budget) and none of it ever compiles on the clock. `cache=False` on purpose:
the platform wipes `/tmp` between games, so a cache buys nothing and costs a stat call.

State is three numpy arrays, passed explicitly to every jitted function:

- `board`  int8[120]   10x12 mailbox. 0 empty, 1..6 white P N B R Q K, 7..12 black, 13 off board.
                       a8 is 21 and h1 is 98, so a white pawn steps by -10.
- `st`     int64[9]    0 side to move (0 white, 1 black), 1 castling mask (1 WK, 2 WQ, 4 BK,
                       8 BQ), 2 en passant square in mailbox coordinates or 0, 3 halfmove
                       clock, 4 fullmove number, 5 white king square, 6 black king square,
                       7 Zobrist key, 8 undo stack pointer. `st[5 + side]` is the side's king
                       and `st[7]` is the key a transposition table indexes on.
- `undo`   int64[N, 6] the unmake stack: captured piece, ep, castling, halfmove, key, fullmove.

A move is one int32: bits 0-6 from, 7-13 to, 14-16 promotion piece type, 17 en passant
capture, 18 castling, 19 double pawn push. Move lists are preallocated int32 arrays plus a
count; nothing here allocates per node.

The en passant square is stored only when an enemy pawn actually sits beside the pushed pawn,
which keeps the Zobrist key free of noise that would split identical positions in the table.
`to_fen` then prints the square only when the capture is genuinely legal, which is the rule
python-chess follows, so the two agree character for character.
"""

import numpy as np
from numba import njit
from numba import types as nbt

EMPTY = 0
OFF = 13
MAX_MOVES = 256
MAX_PLY = 128
UNDO_SIZE = 1024

FLAG_EP = 1 << 17
FLAG_CASTLE = 1 << 18
FLAG_DOUBLE = 1 << 19

PIECE_CHARS = ".PNBRQKpnbrqk"
PROMO_CHARS = {2: "n", 3: "b", 4: "r", 5: "q"}

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

KNIGHT_D = np.array([-21, -19, -12, -8, 8, 12, 19, 21], dtype=np.int64)
KING_D = np.array([-11, -10, -9, -1, 1, 9, 10, 11], dtype=np.int64)
ALL_D = np.array([-11, -9, 9, 11, -10, -1, 1, 10], dtype=np.int64)

_BORDER = np.zeros(120, dtype=np.int8)
for _i in range(120):
    if _i < 21 or _i > 98 or _i % 10 == 0 or _i % 10 == 9:
        _BORDER[_i] = OFF

CASTLE_MASK = np.full(120, 15, dtype=np.int64)
CASTLE_MASK[91] = 15 & ~2
CASTLE_MASK[98] = 15 & ~1
CASTLE_MASK[95] = 15 & ~3
CASTLE_MASK[21] = 15 & ~8
CASTLE_MASK[28] = 15 & ~4
CASTLE_MASK[25] = 15 & ~12

_RNG = np.random.default_rng(0x9E3779B97F4A7C15)
_KEYS = _RNG.integers(0, 1 << 64, size=13 * 120 + 16 + 8 + 1, dtype=np.uint64).view(np.int64)
ZOB_PIECE = np.ascontiguousarray(_KEYS[: 13 * 120].reshape(13, 120))
ZOB_CASTLE = np.ascontiguousarray(_KEYS[13 * 120 : 13 * 120 + 16])
ZOB_EP = np.ascontiguousarray(_KEYS[13 * 120 + 16 : 13 * 120 + 24])
ZOB_SIDE = int(_KEYS[-1])

_BOARD_T = nbt.int8[::1]
_ST_T = nbt.int64[::1]
_UNDO_T = nbt.int64[:, ::1]
_MOVES_T = nbt.int32[::1]
_BUFS_T = nbt.int32[:, ::1]


@njit(nbt.boolean(_BOARD_T, nbt.int64, nbt.int64), cache=False)
def is_square_attacked(board: np.ndarray, sq: int, by_side: int) -> bool:
    """Is `sq` attacked by any piece of `by_side` (0 white, 1 black)?"""
    base = 6 * by_side
    if by_side == 0:
        if board[sq + 9] == 1 or board[sq + 11] == 1:
            return True
    elif board[sq - 9] == 7 or board[sq - 11] == 7:
        return True
    knight = base + 2
    king = base + 6
    for k in range(8):
        if board[sq + KNIGHT_D[k]] == knight:
            return True
        if board[sq + KING_D[k]] == king:
            return True
    bishop = base + 3
    rook = base + 4
    queen = base + 5
    for k in range(8):
        step = ALL_D[k]
        slider = bishop if k < 4 else rook
        to = sq + step
        piece = board[to]
        while piece == EMPTY:
            to += step
            piece = board[to]
        if piece in (queen, slider):
            return True
    return False


@njit(nbt.boolean(_BOARD_T, _ST_T), cache=False)
def in_check(board: np.ndarray, st: np.ndarray) -> bool:
    """Is the side to move in check?"""
    side = st[0]
    return is_square_attacked(board, st[5 + side], 1 - side)


@njit(nbt.int64(_BOARD_T, _ST_T, _MOVES_T), cache=False)
def gen_moves(board: np.ndarray, st: np.ndarray, out: np.ndarray) -> int:
    """Write the pseudo-legal moves into `out` and return how many there are.

    Castling is filtered here (the king may not be in check, cross an attacked square, or land
    on one) because the make-and-look-at-the-king test cannot see the squares it passed over.
    Everything else can leave the king hanging and is rejected by the caller.
    """
    side = st[0]
    ep = st[2]
    my_lo = 1 + 6 * side
    my_hi = my_lo + 5
    opp_lo = 1 + 6 * (1 - side)
    opp_hi = opp_lo + 5
    n = 0
    for frm in range(21, 99):
        piece = board[frm]
        if piece < my_lo or piece > my_hi:
            continue
        kind = piece - 6 * side
        if kind == 1:
            if side == 0:
                fwd = -10
                start_lo = 81
                promo_lo = 21
            else:
                fwd = 10
                start_lo = 31
                promo_lo = 91
            start_hi = start_lo + 7
            promo_hi = promo_lo + 7
            to = frm + fwd
            if board[to] == EMPTY:
                if promo_lo <= to <= promo_hi:
                    for kind_p in range(2, 6):
                        out[n] = frm | (to << 7) | (kind_p << 14)
                        n += 1
                else:
                    out[n] = frm | (to << 7)
                    n += 1
                    if start_lo <= frm <= start_hi and board[frm + 2 * fwd] == EMPTY:
                        out[n] = frm | ((frm + 2 * fwd) << 7) | FLAG_DOUBLE
                        n += 1
            for side_step in range(-1, 2, 2):
                to = frm + fwd + side_step
                target = board[to]
                if opp_lo <= target <= opp_hi:
                    if promo_lo <= to <= promo_hi:
                        for kind_p in range(2, 6):
                            out[n] = frm | (to << 7) | (kind_p << 14)
                            n += 1
                    else:
                        out[n] = frm | (to << 7)
                        n += 1
                elif ep != 0 and to == ep:
                    out[n] = frm | (to << 7) | FLAG_EP
                    n += 1
        elif kind == 2 or kind == 6:
            for k in range(8):
                to = frm + (KNIGHT_D[k] if kind == 2 else KING_D[k])
                target = board[to]
                if target == EMPTY or opp_lo <= target <= opp_hi:
                    out[n] = frm | (to << 7)
                    n += 1
        else:
            if kind == 3:
                first = 0
                last = 4
            elif kind == 4:
                first = 4
                last = 8
            else:
                first = 0
                last = 8
            for k in range(first, last):
                step = ALL_D[k]
                to = frm + step
                target = board[to]
                while target == EMPTY:
                    out[n] = frm | (to << 7)
                    n += 1
                    to += step
                    target = board[to]
                if opp_lo <= target <= opp_hi:
                    out[n] = frm | (to << 7)
                    n += 1
    rights = st[1]
    if side == 0:
        king_sq = 95
        short_right = 1
        long_right = 2
        rook_short = 98
        rook_long = 91
        rook_piece = 4
    else:
        king_sq = 25
        short_right = 4
        long_right = 8
        rook_short = 28
        rook_long = 21
        rook_piece = 10
    if (rights & (short_right | long_right)) != 0 and board[king_sq] == 6 + 6 * side:
        other = 1 - side
        if not is_square_attacked(board, king_sq, other):
            if (
                (rights & short_right) != 0
                and board[rook_short] == rook_piece
                and board[king_sq + 1] == EMPTY
                and board[king_sq + 2] == EMPTY
                and not is_square_attacked(board, king_sq + 1, other)
                and not is_square_attacked(board, king_sq + 2, other)
            ):
                out[n] = king_sq | ((king_sq + 2) << 7) | FLAG_CASTLE
                n += 1
            if (
                (rights & long_right) != 0
                and board[rook_long] == rook_piece
                and board[king_sq - 1] == EMPTY
                and board[king_sq - 2] == EMPTY
                and board[king_sq - 3] == EMPTY
                and not is_square_attacked(board, king_sq - 1, other)
                and not is_square_attacked(board, king_sq - 2, other)
            ):
                out[n] = king_sq | ((king_sq - 2) << 7) | FLAG_CASTLE
                n += 1
    return n


@njit(nbt.void(_BOARD_T, _ST_T, _UNDO_T, nbt.int32), cache=False)
def make_move(board: np.ndarray, st: np.ndarray, undo: np.ndarray, move: int) -> None:
    """Play `move`, updating the Zobrist key incrementally and pushing an undo record."""
    frm = move & 127
    to = (move >> 7) & 127
    promo = (move >> 14) & 7
    side = st[0]
    piece = board[frm]
    captured = board[to]
    key = st[7]

    sp = st[8]
    undo[sp, 0] = captured
    undo[sp, 1] = st[2]
    undo[sp, 2] = st[1]
    undo[sp, 3] = st[3]
    undo[sp, 4] = key
    undo[sp, 5] = st[4]
    st[8] = sp + 1

    if st[2] != 0:
        key ^= ZOB_EP[(st[2] - 21) % 10]
    key ^= ZOB_CASTLE[st[1]]
    key ^= ZOB_SIDE

    board[frm] = EMPTY
    key ^= ZOB_PIECE[piece, frm]

    reset_clock = piece == 1 + 6 * side
    if (move & FLAG_EP) != 0:
        cap_sq = to + 10 if side == 0 else to - 10
        cap_piece = board[cap_sq]
        undo[sp, 0] = cap_piece
        board[cap_sq] = EMPTY
        key ^= ZOB_PIECE[cap_piece, cap_sq]
    elif captured != EMPTY:
        key ^= ZOB_PIECE[captured, to]
        reset_clock = True

    landed = promo + 6 * side if promo != 0 else piece
    board[to] = landed
    key ^= ZOB_PIECE[landed, to]

    if (move & FLAG_CASTLE) != 0:
        if to > frm:
            rook_from = frm + 3
            rook_to = frm + 1
        else:
            rook_from = frm - 4
            rook_to = frm - 1
        rook = board[rook_from]
        board[rook_from] = EMPTY
        board[rook_to] = rook
        key ^= ZOB_PIECE[rook, rook_from] ^ ZOB_PIECE[rook, rook_to]

    rights = st[1] & CASTLE_MASK[frm] & CASTLE_MASK[to]
    st[1] = rights
    key ^= ZOB_CASTLE[rights]

    ep = 0
    if (move & FLAG_DOUBLE) != 0:
        enemy_pawn = 1 + 6 * (1 - side)
        if board[to - 1] == enemy_pawn or board[to + 1] == enemy_pawn:
            ep = (frm + to) // 2
            key ^= ZOB_EP[(ep - 21) % 10]
    st[2] = ep

    st[3] = 0 if reset_clock else st[3] + 1
    if side == 1:
        st[4] += 1
    if piece == 6 + 6 * side:
        st[5 + side] = to
    st[0] = 1 - side
    st[7] = key


@njit(nbt.void(_BOARD_T, _ST_T, _UNDO_T, nbt.int32), cache=False)
def unmake_move(board: np.ndarray, st: np.ndarray, undo: np.ndarray, move: int) -> None:
    """Undo the move that `make_move` just played, restoring the state exactly."""
    frm = move & 127
    to = (move >> 7) & 127
    promo = (move >> 14) & 7
    side = 1 - st[0]

    sp = st[8] - 1
    st[8] = sp
    captured = undo[sp, 0]
    st[2] = undo[sp, 1]
    st[1] = undo[sp, 2]
    st[3] = undo[sp, 3]
    st[7] = undo[sp, 4]
    st[4] = undo[sp, 5]
    st[0] = side

    piece = 1 + 6 * side if promo != 0 else board[to]
    board[frm] = piece
    board[to] = EMPTY

    if (move & FLAG_EP) != 0:
        board[to + 10 if side == 0 else to - 10] = captured
    elif captured != EMPTY:
        board[to] = captured

    if (move & FLAG_CASTLE) != 0:
        if to > frm:
            rook_from = frm + 3
            rook_to = frm + 1
        else:
            rook_from = frm - 4
            rook_to = frm - 1
        board[rook_from] = board[rook_to]
        board[rook_to] = EMPTY

    if piece == 6 + 6 * side:
        st[5 + side] = frm


@njit(nbt.int64(_BOARD_T, _ST_T, _UNDO_T, _MOVES_T), cache=False)
def gen_legal(board: np.ndarray, st: np.ndarray, undo: np.ndarray, out: np.ndarray) -> int:
    """Write the legal moves into `out` and return how many. Pseudo-legal, then make and look."""
    n = gen_moves(board, st, out)
    side = st[0]
    other = 1 - side
    kept = 0
    for i in range(n):
        move = out[i]
        make_move(board, st, undo, move)
        legal = not is_square_attacked(board, st[5 + side], other)
        unmake_move(board, st, undo, move)
        if legal:
            out[kept] = move
            kept += 1
    return kept


@njit(nbt.int64(_BOARD_T, _ST_T), cache=False)
def compute_key(board: np.ndarray, st: np.ndarray) -> int:
    """Recompute the Zobrist key from scratch. The incremental key must always match this."""
    key = 0
    for sq in range(21, 99):
        piece = board[sq]
        if piece != EMPTY and piece != OFF:
            key ^= ZOB_PIECE[piece, sq]
    key ^= ZOB_CASTLE[st[1]]
    if st[2] != 0:
        key ^= ZOB_EP[(st[2] - 21) % 10]
    if st[0] == 1:
        key ^= ZOB_SIDE
    return int(key)


@njit(
    nbt.int64(_BOARD_T, _ST_T, _UNDO_T, _BUFS_T, nbt.int64, nbt.int64), cache=False
)
def perft(
    board: np.ndarray, st: np.ndarray, undo: np.ndarray, bufs: np.ndarray, depth: int, ply: int
) -> int:
    """Count leaf nodes at `depth`, counting legal moves in bulk at the last ply."""
    out = bufs[ply]
    n = gen_moves(board, st, out)
    side = st[0]
    other = 1 - side
    total = 0
    for i in range(n):
        move = out[i]
        make_move(board, st, undo, move)
        if not is_square_attacked(board, st[5 + side], other):
            total += 1 if depth == 1 else perft(board, st, undo, bufs, depth - 1, ply + 1)
        unmake_move(board, st, undo, move)
    return total


def new_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An empty board, its scalar state and a fresh undo stack."""
    return _BORDER.copy(), np.zeros(9, dtype=np.int64), np.zeros((UNDO_SIZE, 6), dtype=np.int64)


def move_buffer() -> np.ndarray:
    """A move list sized for any position."""
    return np.zeros(MAX_MOVES, dtype=np.int32)


def perft_buffers() -> np.ndarray:
    """One move list per ply, for `perft`."""
    return np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)


def square_index(name: str) -> int:
    """`e4` to its mailbox index."""
    return 21 + (7 - (ord(name[1]) - ord("1"))) * 10 + (ord(name[0]) - ord("a"))


def square_name(sq: int) -> str:
    """A mailbox index back to `e4`."""
    return chr(ord("a") + (sq - 21) % 10) + chr(ord("1") + 7 - (sq - 21) // 10)


def from_fen(fen: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a FEN into `(board, st, undo)`."""
    board, st, undo = new_state()
    fields = fen.split()
    placement, turn, castling, ep = fields[0], fields[1], fields[2], fields[3]
    sq = 21
    for char in placement:
        if char == "/":
            sq += 2
        elif char.isdigit():
            sq += int(char)
        else:
            board[sq] = PIECE_CHARS.index(char)
            sq += 1
    st[0] = 0 if turn == "w" else 1
    rights = 0
    for char, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)):
        if char in castling:
            rights |= bit
    st[1] = rights
    st[3] = int(fields[4]) if len(fields) > 4 else 0
    st[4] = int(fields[5]) if len(fields) > 5 else 1
    for square in range(21, 99):
        if board[square] == 6:
            st[5] = square
        elif board[square] == 12:
            st[6] = square
    if ep != "-":
        target = square_index(ep)
        pushed = target + 10 if st[0] == 0 else target - 10
        capturer = 1 + 6 * st[0]
        if board[pushed - 1] == capturer or board[pushed + 1] == capturer:
            st[2] = target
    st[7] = compute_key(board, st)
    return board, st, undo


def to_fen(board: np.ndarray, st: np.ndarray, undo: np.ndarray) -> str:
    """Serialise the state.

    The ep square is printed only when the capture is genuinely legal, which is what
    python-chess does, so the two agree character for character.
    """
    rows = []
    for rank_start in range(21, 99, 10):
        row = ""
        run = 0
        for sq in range(rank_start, rank_start + 8):
            piece = board[sq]
            if piece == EMPTY:
                run += 1
                continue
            if run:
                row += str(run)
                run = 0
            row += PIECE_CHARS[piece]
        rows.append(row + (str(run) if run else ""))
    rights = "".join(
        char for char, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)) if st[1] & bit
    )
    ep = "-"
    if st[2] != 0:
        moves = move_buffer()
        count = gen_legal(board, st, undo, moves)
        target = int(st[2])
        if any(int(moves[i] >> 7) & 127 == target and moves[i] & FLAG_EP for i in range(count)):
            ep = square_name(target)
    turn = "w" if st[0] == 0 else "b"
    return f"{'/'.join(rows)} {turn} {rights or '-'} {ep} {st[3]} {st[4]}"


def move_to_uci(move: int) -> str:
    """`e2e4` or `e7e8q`."""
    promo = (move >> 14) & 7
    return square_name(move & 127) + square_name((move >> 7) & 127) + PROMO_CHARS.get(promo, "")


def uci_to_move(board: np.ndarray, st: np.ndarray, undo: np.ndarray, uci: str) -> int:
    """Find the legal move with this UCI string, or raise."""
    moves = move_buffer()
    count = gen_legal(board, st, undo, moves)
    for i in range(count):
        if move_to_uci(int(moves[i])) == uci:
            return int(moves[i])
    raise ValueError(f"{uci} is not legal here")


def legal_moves(board: np.ndarray, st: np.ndarray, undo: np.ndarray) -> list[int]:
    """The legal moves as a Python list. For the root and for tests, never inside a search."""
    moves = move_buffer()
    count = gen_legal(board, st, undo, moves)
    return [int(moves[i]) for i in range(count)]


def run_perft(fen: str, depth: int) -> int:
    """Perft from a FEN, allocating the buffers it needs."""
    board, st, undo = from_fen(fen)
    if depth <= 0:
        return 1
    return int(perft(board, st, undo, perft_buffers(), depth, 0))


def warm() -> None:
    """Run every jitted function once with the types it will really see.

    The eager signatures above already compile at import; this proves the whole graph runs and
    warms the recursive `perft` specialisation.
    """
    board, st, undo = from_fen(START_FEN)
    moves = move_buffer()
    gen_moves(board, st, moves)
    gen_legal(board, st, undo, moves)
    in_check(board, st)
    is_square_attacked(board, 95, 1)
    compute_key(board, st)
    first = int(moves[0])
    make_move(board, st, undo, first)
    unmake_move(board, st, undo, first)
    perft(board, st, undo, perft_buffers(), 2, 0)


warm()
