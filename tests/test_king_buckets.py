"""King buckets, proved rather than assumed.

    uv run python -m tests.test_king_buckets [--full] [--sequences N]

`docs/NNUE_KING_BUCKETS.md` is the spec. This file checks the four implementations of it agree
-- training features, export layout, the python reference and the numba runtime -- and that the
accumulator rules the bucketing forces are exact.

What it does not check, because no test can: whether king conditioning is worth anything. A
warm-started file evaluates every position to the integer the 768 net returned, which is what
`check_warm_start` proves. Strength is a bench question, after fine-tuning.

The checks, in the order they run:

  tables            the runtime's KING_OFFSET is the offline KING_BUCKET, square by square
  index space       every index in range, blocks disjoint, base indices collision-free
  symmetry          colour symmetry and the mirror invariant, offline and in the runtime
  buckets           all four are reachable, selected correctly, and used by the runtime
  warm start        the tiled net returns exactly what the 768 net returns, 10,000 positions
  reference         numba inference equals nnue_ref, over positions in every bucket
  increments        push equals a from-scratch refresh over >= 10,000 randomised plies
  distinct          the same, on a net whose four blocks differ, which a warm start's do not
  rebucket          the scheme-1 -> scheme-2 shard converter against direct extraction
  bounds            the int16 accumulator bound, proved per block and watched empirically
  file              a bucketised weight file loads, evaluates identically and fits the cap
  import            import time and peak resident memory of a real agent process
"""

from __future__ import annotations

import argparse
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chess
import numpy as np

import fastboard as fb
import fastnnue as fn
import fastsearch as fs
from tests.test_fastboard import STRESS_SEEDS
from tests.test_fasteval import ENDGAME_SEEDS
from tests.test_nnue import DELTA_SEEDS, Failure, sample
from tools.nnue import bucketize, features, nnue_ref, rebucket

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "nnue.npz"
# The zip cap the platform enforces, and the room the rest of the agent needs inside it.
ZIP_CAP_BYTES = 50_000_000
# What the platform allows for import before the clock starts.
INIT_BUDGET_S = 90.0
# What the container gives us.
MEMORY_CAP_MB = 2048


# What a full-length randomised walk has to have reached before it is allowed to pass. A walk
# that never castled or never crossed a bucket boundary proves nothing about those paths, and
# silence is exactly how that kind of gap survives.
REQUIRED = ("captures", "en passant", "promotions", "castles", "nulls", "crossed", "same bucket")
# What the shorter distinct-block walk has to have reached. En passant is rare enough that a
# 4,000-ply sample can miss it honestly -- it did, once `tests/test_nnue.py` changed under v4.4
# and moved the shared random stream -- and that walk is not there to cover en passant anyway:
# the full walk above already did, on the same code, with the only difference being the weights.
# What this one is for is the bucket paths on a net whose blocks actually differ.
BUCKET_REQUIRED = ("captures", "crossed", "same bucket")


def mailbox(square: int) -> int:
    """The `fastboard` mailbox index for a python-chess square (a1 = 0, h8 = 63).

    The 10x12 board runs top down: index 21 is a8 and index 91 is a1, which is what
    `fasteval.RANK_OF` says (`7 - offset // 10`). Getting this backwards is the first thing
    `check_tables` catches, and it caught it once already.
    """
    return 21 + (7 - square // 8) * 10 + (square % 8)


def check_tables() -> str:
    """The runtime's bucket table has to be the offline one, on every square and perspective.

    Two copies of a formula in two languages is exactly the shape of bug that survives every
    other test here: the net trains on one bucketing and plays another, both self-consistent.
    """
    checked = 0
    for perspective in range(2):
        for square in range(64):
            flip = 56 if perspective == 1 else 0
            want = int(features.KING_BUCKET[square ^ flip]) * features.BASE_FEATURES
            got = int(fn.KING_OFFSET[perspective, mailbox(square)])
            if got != want:
                raise Failure(
                    f"perspective {perspective}, {chess.square_name(square)}: the runtime says "
                    f"offset {got}, tools/nnue/features.py says {want}"
                )
            checked += 1
    off_board = [
        int(fn.KING_OFFSET[perspective, index])
        for perspective in range(2)
        for index in (0, 20, 99, 119)
    ]
    if any(off_board):
        raise Failure("an off-board mailbox square has a non-zero king offset")
    return f"{checked} square/perspective pairs match the offline table, off-board squares are 0"


def check_index_space() -> str:
    """Every index in range, the four blocks disjoint, and the base indices collision-free."""
    seen: dict[tuple[int, int], tuple[int, int, int]] = {}
    for perspective in range(2):
        for piece in range(1, 13):
            for square in range(64):
                index = int(fn.FEATURE[perspective, piece, mailbox(square)])
                if not 0 <= index < features.BASE_FEATURES:
                    raise Failure(
                        f"base index {index} for piece {piece} on {chess.square_name(square)} "
                        f"is outside [0, {features.BASE_FEATURES})"
                    )
                key = (perspective, index)
                if key in seen and seen[key] != (perspective, piece, square):
                    other = seen[key]
                    raise Failure(
                        f"base index {index} is shared by piece {piece} on square {square} and "
                        f"piece {other[1]} on square {other[2]}, perspective {perspective}"
                    )
                seen[key] = (perspective, piece, square)

    # Every reachable index, base plus any bucket offset, has to address a real row.
    highest = features.BASE_FEATURES * (features.NUM_BUCKETS - 1) + features.BASE_FEATURES - 1
    if highest != features.NUM_FEATURES - 1:
        raise Failure("the bucket blocks do not tile the first layer exactly")
    rows = fn.NET[0].shape[0]
    if rows != features.NUM_FEATURES:
        raise Failure(f"the loaded first layer has {rows} rows, not {features.NUM_FEATURES}")
    # And the blocks must be disjoint: bucket b owns [b * 768, (b + 1) * 768).
    for bucket in range(features.NUM_BUCKETS):
        low = bucket * features.BASE_FEATURES
        if int(fn.KING_OFFSET.max()) < low and bucket:
            raise Failure(f"bucket {bucket} is never selected by any king square")
    return (
        f"{len(seen)} base indices in [0, {features.BASE_FEATURES}), all distinct per "
        f"perspective; {features.NUM_BUCKETS} disjoint blocks tile {features.NUM_FEATURES} rows"
    )


def _accumulator(board_fen: str, net: fn.Net) -> tuple[np.ndarray, int]:
    """Both perspectives of `board_fen` built from scratch, and the side to move."""
    board, st, _ = fb.from_fen(board_fen)
    acc = fn.accumulators(1, net[1].shape[0])
    fn.refresh(board, acc, 0, net)
    return acc, int(st[0])


def check_symmetry(fens: list[str], net: fn.Net) -> str:
    """Colour symmetry offline, and the same thing inside the runtime's accumulators.

    The 768 scheme's invariant is that a position and its colour-swapped vertical mirror give
    the same features. Applying the bucket to the already-flipped king square is what keeps
    that true, so it is checked on the composed index rather than on the bucket alone.
    """
    mirrored = 0
    for fen in fens:
        board = chess.Board(fen)
        flipped = board.mirror()
        if sorted(features.features(board)) != sorted(features.features(flipped)):
            raise Failure(f"{fen!r} and its mirror do not have the same features")
        # And in the runtime: the mirror swaps which perspective is which.
        acc, _ = _accumulator(fen, net)
        macc, _ = _accumulator(flipped.fen(), net)
        if not np.array_equal(acc[0, 0, : net[1].shape[0]], macc[0, 1, : net[1].shape[0]]):
            raise Failure(f"{fen!r}: the white perspective is not the mirror's black one")
        mirrored += 1
    return f"{mirrored} positions equal their colour-swapped mirror, offline and in the runtime"


def _placed(rng: random.Random, white_king: int, black_king: int) -> chess.Board | None:
    """A random legal position with the two kings on the squares asked for, or None."""
    board = chess.Board.empty()
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    kinds = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    for _ in range(rng.randint(2, 10)):
        square = rng.randrange(64)
        if board.piece_at(square) is not None:
            continue
        kind = rng.choice(kinds)
        if kind == chess.PAWN and not 8 <= square < 56:
            continue
        board.set_piece_at(square, chess.Piece(kind, rng.choice([chess.WHITE, chess.BLACK])))
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not board.is_valid():
        return None
    return board


def bucket_positions(rng: random.Random, per_bucket: int) -> dict[int, list[str]]:
    """`per_bucket` random legal positions with the side to move's king in each bucket."""
    wanted = {bucket: per_bucket for bucket in range(features.NUM_BUCKETS)}
    found: dict[int, list[str]] = {bucket: [] for bucket in range(features.NUM_BUCKETS)}
    attempts = 0
    while any(len(found[b]) < wanted[b] for b in wanted) and attempts < per_bucket * 4000:
        attempts += 1
        board = _placed(rng, rng.randrange(64), rng.randrange(64))
        if board is None:
            continue
        king = board.king(board.turn)
        if king is None:
            continue
        flip = 56 if board.turn == chess.BLACK else 0
        bucket = features.king_bucket(king ^ flip)
        if len(found[bucket]) < wanted[bucket]:
            found[bucket].append(board.fen())
    missing = [bucket for bucket in found if len(found[bucket]) < wanted[bucket]]
    if missing:
        raise Failure(f"could not build positions for bucket(s) {missing}")
    return found


def check_buckets(by_bucket: dict[int, list[str]], net: fn.Net) -> str:
    """All four buckets are reachable, and the runtime selects the one the spec names."""
    hidden = net[1].shape[0]
    for bucket, fens in sorted(by_bucket.items()):
        for fen in fens:
            board = chess.Board(fen)
            king = board.king(board.turn)
            assert king is not None
            perspective = 0 if board.turn == chess.WHITE else 1
            offset = int(fn.KING_OFFSET[perspective, mailbox(king)])
            if offset != bucket * features.BASE_FEATURES:
                raise Failure(
                    f"{fen!r}: the runtime puts the king in bucket "
                    f"{offset // features.BASE_FEATURES}, the spec says {bucket}"
                )
            # And the accumulator really is the one that block builds.
            acc, side = _accumulator(fen, net)
            want = nnue_ref.evaluate(nnue_ref.load(WEIGHTS), features.features(board))
            got = fn.infer(acc, 0, side, net)
            if want != got:
                raise Failure(f"{fen!r} in bucket {bucket}: runtime {got} cp, reference {want} cp")
            _ = hidden
    counts = ", ".join(f"bucket {b} {len(f)}" for b, f in sorted(by_bucket.items()))
    return f"all {features.NUM_BUCKETS} buckets selected and evaluated correctly ({counts})"


def check_warm_start(fens: list[str], net: fn.Net) -> str:
    """The warm-started net returns exactly what the 768 net returned. This is the whole claim.

    The 768 evaluation is recomputed here from the file's own untiled first layer and the base
    index (`index % 768`), so it does not depend on the old runtime still being importable.
    """
    with np.load(WEIGHTS) as data:
        if int(data["version"]) != fn.FLAT_SCHEME_VERSION:
            return (
                f"skipped: {WEIGHTS.name} is already scheme {int(data['version'])}, so there is "
                f"no 768 net to compare against"
            )
        flat = nnue_ref.Weights(
            version=fn.FLAT_SCHEME_VERSION,
            hidden=int(data["hidden"]),
            qa=int(data["qa"]),
            qb=int(data["qb"]),
            qc=int(data["qc"]),
            cp_scale=int(data["cp_scale"]),
            l1_weight=data["l1_weight"],
            l1_bias=data["l1_bias"],
            l2_weight=data["l2_weight"],
            l2_bias=data["l2_bias"],
            l3_weight=data["l3_weight"],
            l3_bias=int(data["l3_bias"]),
        )
    buckets_seen: set[int] = set()
    for fen in fens:
        board = chess.Board(fen)
        active = features.features(board)
        buckets_seen.add(int(active[0]) // features.BASE_FEATURES)
        want = nnue_ref.evaluate(flat, active % features.BASE_FEATURES)
        acc, side = _accumulator(fen, net)
        got = fn.infer(acc, 0, side, net)
        if want != got:
            raise Failure(
                f"{fen!r}: the bucketed net says {got} cp, the 768 net it was warm started "
                f"from says {want} cp"
            )
    return (
        f"{len(fens):,} positions evaluate to the integer the 768 net returned, across "
        f"{len(buckets_seen)} of {features.NUM_BUCKETS} buckets"
    )


def check_reference(fens: list[str], net: fn.Net) -> dict[str, int]:
    """numba inference equals `tools/nnue/nnue_ref.py`, the spec, on bucketed features."""
    weights = nnue_ref.load(WEIGHTS)
    tally = {"positions": 0, "black to move": 0, "en passant": 0, "promotions": 0}
    for bucket in range(features.NUM_BUCKETS):
        tally[f"bucket {bucket}"] = 0
    for fen in fens:
        board = chess.Board(fen)
        active = features.features(board)
        want = int(nnue_ref.evaluate(weights, active))
        acc, side = _accumulator(fen, net)
        got = int(fn.infer(acc, 0, side, net))
        if want != got:
            raise Failure(f"{fen!r}: numba {got} cp, reference {want} cp")
        tally["positions"] += 1
        tally["black to move"] += board.turn == chess.BLACK
        tally["en passant"] += board.ep_square is not None
        tally["promotions"] += any(
            move.promotion is not None for move in board.legal_moves
        )
        tally[f"bucket {int(active[0]) // features.BASE_FEATURES}"] += 1
    return tally


def check_increments(
    rng: random.Random, wanted: int, net: fn.Net, require: tuple[str, ...] = REQUIRED
) -> dict[str, int]:
    """`push` has to equal a from-scratch refresh, at every ply, including bucket crossings.

    The same walk `tests/test_nnue.py` does, with the king-bucket cases counted separately and
    asserted non-zero at the end: a run in which no king ever crossed a boundary would pass
    while proving nothing about the one path this branch adds.

    The seeds include positions with kings already in the far half, because a walk that starts
    from opening positions reaches bucket 2 and 3 rarely and the crossing into them is the case
    that matters.
    """
    hidden = net[1].shape[0]
    acc = fn.accumulators(fs.MAX_SEARCH_PLY + 1, hidden)
    scratch = fn.accumulators(1, hidden)
    tally = {
        "plies": 0, "captures": 0, "en passant": 0, "promotions": 0, "castles": 0, "nulls": 0,
        "king moves": 0, "same bucket": 0, "crossed": 0, "castles crossing": 0,
    }
    far = [fens[0] for fens in bucket_positions(rng, 6).values()]
    seeds = [fb.START_FEN, *DELTA_SEEDS, *STRESS_SEEDS, *ENDGAME_SEEDS, *far]
    depth_limit = 8
    per_seed = max(wanted // 40, 40)
    budget = 0

    def descend(board: np.ndarray, st: np.ndarray, undo: np.ndarray, ply: int, passes: int) -> None:
        if tally["plies"] >= budget or ply >= depth_limit:
            return
        moves = fb.legal_moves(board, st, undo)
        if not moves:
            return
        if passes == 0 and rng.random() < 0.10:
            parent = acc[ply].copy()
            fn.push_null(acc, ply, net)
            fs.make_null(st, undo)
            fn.refresh(board, scratch, 0, net)
            if not np.array_equal(acc[ply + 1], scratch[0]):
                raise Failure(f"a null move at ply {ply} left the accumulator wrong")
            tally["plies"] += 1
            tally["nulls"] += 1
            descend(board, st, undo, ply + 1, 1)
            fs.unmake_null(st, undo)
            if not np.array_equal(acc[ply], parent):
                raise Failure(f"unmaking a null move at ply {ply} did not leave acc[{ply}] alone")
            return

        # King moves are rare in a uniform sample and are the whole point here, so they are
        # drawn on purpose as well as at random.
        chosen = rng.sample(moves, min(len(moves), 3))
        kings = [m for m in moves if board[m & 127] in (fn.WHITE_KING, fn.BLACK_KING)]
        if kings:
            chosen.append(rng.choice(kings))
        for move in chosen:
            if tally["plies"] >= budget:
                return
            side = int(st[0])
            frm = move & 127
            to = (move >> 7) & 127
            piece = board[frm]
            is_king = piece in (fn.WHITE_KING, fn.BLACK_KING)
            crossing = is_king and (
                fn.KING_OFFSET[side, frm] != fn.KING_OFFSET[side, to]
            )
            target = board[to]
            parent = acc[ply].copy()
            fn.push(board, side, acc, ply, move, net)
            fb.make_move(board, st, undo, move)
            fn.refresh(board, scratch, 0, net)
            if not np.array_equal(acc[ply + 1], scratch[0]):
                for perspective in range(2):
                    if np.array_equal(acc[ply + 1, perspective], scratch[0, perspective]):
                        continue
                    unit = int(np.argmax(acc[ply + 1, perspective] != scratch[0, perspective]))
                    raise Failure(
                        f"{fb.move_to_uci(move)} at ply {ply} in {fb.to_fen(board, st)!r} "
                        f"({'crossing' if crossing else 'same bucket'}): perspective "
                        f"{perspective} unit {unit} of {hidden} is "
                        f"{acc[ply + 1, perspective, unit]}, from scratch it is "
                        f"{scratch[0, perspective, unit]}"
                    )
            tally["plies"] += 1
            tally["captures"] += target != 0
            tally["en passant"] += (move & fb.FLAG_EP) != 0
            tally["promotions"] += ((move >> 14) & 7) != 0
            tally["castles"] += (move & fb.FLAG_CASTLE) != 0
            tally["king moves"] += is_king
            tally["crossed"] += crossing
            tally["same bucket"] += is_king and not crossing
            tally["castles crossing"] += bool((move & fb.FLAG_CASTLE) != 0 and crossing)
            descend(board, st, undo, ply + 1, 0)
            fb.unmake_move(board, st, undo, move)
            if not np.array_equal(acc[ply], parent):
                raise Failure(
                    f"unmaking {fb.move_to_uci(move)} at ply {ply} did not leave acc[{ply}] alone"
                )

    while tally["plies"] < wanted:
        budget = min(wanted, tally["plies"] + per_seed)
        board, st, undo = fb.from_fen(rng.choice(seeds))
        for _ in range(rng.randint(0, 16)):
            moves = fb.legal_moves(board, st, undo)
            if not moves:
                break
            fb.make_move(board, st, undo, rng.choice(moves))
        fn.refresh(board, acc, 0, net)
        descend(board, st, undo, 0, 0)

    for name in require:
        if not tally[name]:
            raise Failure(
                f"the walk never reached a single {name} in {tally['plies']:,} plies, so it "
                f"proves nothing about it"
            )
    return tally


def _perturbed(net: fn.Net, seed: int) -> fn.Net:
    """The loaded net with each bucket block moved differently, block 0 left alone.

    Small enough not to threaten the int16 bound (at most 16 per weight, 32 men, against seven
    thousand of headroom) and large enough that no position could evaluate the same by accident.
    """
    generator = np.random.default_rng(seed)
    weights = net[0].astype(np.int32)
    noise = generator.integers(-16, 17, size=weights.shape, dtype=np.int32)
    noise[: fn.BASE_FEATURES] = 0
    moved = np.clip(weights + noise, -fn.INT16_MAX, fn.INT16_MAX).astype(np.int16)
    return (np.ascontiguousarray(moved), *net[1:])


def check_distinct_blocks(rng: random.Random, fens: list[str], net: fn.Net) -> str:
    """Everything above ran on a file whose four blocks are identical. This one does not.

    A warm-started file cannot tell a bucketing bug from a correct bucketing: if `push` used the
    wrong block, or `refresh` and `push` disagreed about which block they were in, four copies
    of the same rows would hide it completely. So the checks that matter are repeated here
    against a net whose blocks differ, which is what a fine-tuned file will look like.
    """
    other = _perturbed(net, seed=20260910)
    hidden = other[1].shape[0]
    blocks = other[0].reshape(fn.NUM_BUCKETS, fn.BASE_FEATURES, hidden)
    for bucket in range(1, fn.NUM_BUCKETS):
        if np.array_equal(blocks[bucket], blocks[0]):
            raise Failure(f"block {bucket} was not perturbed, so this proves nothing")

    weights = nnue_ref.Weights(
        version=fn.SCHEME_VERSION,
        hidden=hidden,
        qa=int(other[6]),
        qb=int(other[7]),
        qc=int(other[8]),
        cp_scale=int(other[9]),
        l1_weight=other[0],
        l1_bias=other[1],
        # the runtime holds layer 2 transposed for its inner loop; the reference wants it back
        l2_weight=np.ascontiguousarray(other[2].T),
        l2_bias=other[3],
        l3_weight=other[4],
        l3_bias=int(other[5]),
    )

    differ = 0
    for fen in fens:
        board = chess.Board(fen)
        want = int(nnue_ref.evaluate(weights, features.features(board)))
        acc, side = _accumulator(fen, other)
        got = int(fn.infer(acc, 0, side, other))
        if want != got:
            raise Failure(f"{fen!r} with distinct blocks: numba {got} cp, reference {want} cp")
        plain, _ = _accumulator(fen, net)
        differ += int(fn.infer(plain, 0, side, net)) != got

    walk = check_increments(rng, 4_000, other, require=BUCKET_REQUIRED)
    if not walk["crossed"]:
        raise Failure("the distinct-block walk never crossed a bucket boundary")
    return (
        f"{len(fens):,} positions exact against the reference with four different blocks "
        f"({differ:,} of them score differently from the warm-started net, as they must), and "
        f"{walk['plies']:,} more plies with {walk['crossed']:,} crossings"
    )


def check_rebucket(rng: random.Random, fens: list[str], by_bucket: dict[int, list[str]]) -> str:
    """`tools/nnue/rebucket.py` must agree with building scheme-2 features from the board.

    The converter never sees a board: it recovers the bucket from the row's own plane-5 king
    feature and adds `bucket * 768`. So the thing to check is that its answer equals what
    `features.features` produces from the position -- on real positions, in all four buckets,
    and on mirrors. The mirror is the case that matters most: it is where the flip and the
    bucket have to compose in the right order, and a converter that read the king's square
    unflipped would pass everything else and fail exactly there.

    Scheme-1 rows are synthesised as `features(board) % 768`, which is precisely the index the
    768 scheme assigned, because a scheme-2 index is that number plus a multiple of 768.
    """
    boards = [chess.Board(f) for f in fens]
    boards += [chess.Board(f) for fens_ in by_bucket.values() for f in fens_]
    boards += [board.mirror() for board in list(boards)]
    rng.shuffle(boards)

    wanted = np.full((len(boards), features.MAX_ACTIVE), features.PAD, dtype=np.int16)
    flat = np.full((len(boards), features.MAX_ACTIVE), features.PAD, dtype=np.int16)
    seen: dict[int, int] = dict.fromkeys(range(features.NUM_BUCKETS), 0)
    for row, board in enumerate(boards):
        active = features.features(board)
        wanted[row, : active.size] = active
        flat[row, : active.size] = active % features.BASE_FEATURES
        seen[int(active[0]) // features.BASE_FEATURES] += 1
    if any(count == 0 for count in seen.values()):
        raise Failure(f"the sample does not cover every bucket: {seen}")

    got = rebucket.convert_indices(flat, "test")
    # The property the converter rests on, asserted rather than implied by the comparison
    # below: a scheme-2 index is the scheme-1 index plus a whole number of blocks, so the two
    # agree modulo 768 on every active feature and the offset is constant within a row.
    active_mask = flat != features.PAD
    if not np.array_equal(got[active_mask] % features.BASE_FEATURES, flat[active_mask]):
        raise Failure("a converted index is not the original one modulo 768")
    offsets = got[active_mask] - flat[active_mask]
    per_row = {int(o) for o in offsets}
    if not per_row <= {b * features.BASE_FEATURES for b in range(features.NUM_BUCKETS)}:
        raise Failure(f"a converted row was offset by something that is not a block: {per_row}")
    if not np.array_equal(got, wanted):
        row = int(np.argmax((got != wanted).any(axis=1)))
        raise Failure(
            f"rebucket disagrees with direct extraction on {boards[row].fen()!r}: "
            f"{sorted(int(x) for x in got[row] if x >= 0)} against "
            f"{sorted(int(x) for x in wanted[row] if x >= 0)}"
        )
    if got.dtype != flat.dtype or got.shape != flat.shape:
        raise Failure("rebucket changed the index matrix's dtype or shape")

    # A whole shard, so the on-disk path is covered too: every other array and the row order
    # have to survive, and the marker has to say scheme 2.
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "in" / "shard_0000.npz"
        source.parent.mkdir(parents=True)
        cp = np.array(rng.sample(range(-2000, 2000), len(boards)), dtype=np.int16)
        wdl = np.array([rng.choice([-1, 0, 1]) for _ in boards], dtype=np.int8)
        extra = np.arange(len(boards), dtype=np.int32)
        np.savez_compressed(source, indices=flat, cp=cp, wdl=wdl, hand=extra)
        out = Path(directory) / "out" / "shard_0000.npz"
        rows, spread = rebucket.convert_shard(source, out)
        with np.load(out) as shard:
            if int(shard["scheme"]) != fn.SCHEME_VERSION:
                raise Failure("the converted shard does not declare scheme 2")
            if not np.array_equal(shard["indices"], wanted):
                raise Failure("the converted shard's indices are not the direct extraction")
            if not np.array_equal(shard["cp"], cp):
                raise Failure("the converter changed cp")
            if not np.array_equal(shard["wdl"], wdl):
                raise Failure("the converter changed wdl")
            if not np.array_equal(shard["hand"], extra):
                raise Failure("the converter dropped an array it did not understand")

        # And it has to refuse what it cannot convert, rather than guessing.
        # A real plane-5 index, so the row genuinely has two friendly kings. Taking "some
        # feature from row 0" instead is how the first version of this passed: piece_map order
        # put a pawn there, the row still held exactly one king, and nothing was refused.
        second_king = np.int16(rebucket.KING_LOW + 3)
        broken_cases = {
            "no king": np.where((flat >= rebucket.KING_LOW) & (flat < rebucket.KING_HIGH),
                                np.int16(0), flat),
            "two kings": np.concatenate(
                [flat, np.full((flat.shape[0], 1), second_king, dtype=np.int16)], axis=1
            ),
            "out of range": np.where(flat >= 0, np.int16(features.NUM_FEATURES), flat),
        }
        refused = []
        for name, broken in broken_cases.items():
            try:
                rebucket.convert_indices(np.ascontiguousarray(broken), name)
            except rebucket.Malformed:
                refused.append(name)
        if len(refused) != len(broken_cases):
            raise Failure(
                f"the converter accepted a malformed matrix; it refused only {refused}"
            )
        try:
            rebucket.convert_shard(out, Path(directory) / "again.npz")
        except rebucket.Malformed:
            pass
        else:
            raise Failure("the converter re-converted a shard that was already scheme 2")

    counts = ", ".join(f"bucket {b} {c}" for b, c in sorted(seen.items()))
    return (
        f"{len(boards)} positions, half of them mirrors, convert to exactly what direct "
        f"extraction gives ({counts}); a {rows}-row shard round-trips with cp, wdl, row order "
        f"and an unrecognised array intact, spread {dict(spread)}; missing kings, double kings, "
        f"out-of-range indices and a re-conversion are all refused"
    )


def check_bounds(net: fn.Net) -> str:
    """The int16 accumulator bound, per block, and what the walk above actually reached."""
    l1_weight, l1_bias = net[0], net[1]
    hidden = l1_bias.shape[0]
    magnitudes = np.abs(l1_weight.astype(np.int32))
    blocks = magnitudes.reshape(features.NUM_BUCKETS, features.BASE_FEATURES, hidden)
    per_block = np.sort(blocks, axis=1)[:, -fn.MAX_ACTIVE:, :].sum(axis=1)
    worst = int((per_block.max(axis=0) + np.abs(l1_bias.astype(np.int32))).max())
    if worst > fn.INT16_MAX:
        raise Failure(f"the int16 accumulator can reach {worst}, past {fn.INT16_MAX}")
    across = int(
        (np.sort(magnitudes, axis=0)[-fn.MAX_ACTIVE:].sum(axis=0)
         + np.abs(l1_bias.astype(np.int32))).max()
    )
    return (
        f"worst reachable accumulator {worst:,} of {fn.INT16_MAX:,} "
        f"(headroom {fn.INT16_MAX - worst:,}); the looser across-block bound would be {across:,}"
    )


def check_file(net: fn.Net, fens: list[str]) -> str:
    """A bucketised scheme-2 file loads, evaluates identically, and leaves room under the cap."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "nnue-buckets.npz"
        with np.load(WEIGHTS) as data:
            if int(data["version"]) != fn.FLAT_SCHEME_VERSION:
                return f"skipped: {WEIGHTS.name} is not a 768 file"
        bucketize.bucketize_weights(WEIGHTS, out)
        written = fn.load(out)
        if fn.scheme(out) != fn.SCHEME_VERSION:
            raise Failure("the bucketised file does not declare the bucketed scheme")
        if not np.array_equal(written[0], net[0]):
            raise Failure("the file on disk and the in-memory warm start disagree")
        for fen in fens[:200]:
            acc, side = _accumulator(fen, written)
            other, _ = _accumulator(fen, net)
            if fn.infer(acc, 0, side, written) != fn.infer(other, 0, side, net):
                raise Failure(f"{fen!r} evaluates differently from the file than from memory")
        size = out.stat().st_size
        book = (ROOT / "weights" / "book.bin").stat().st_size
        source = sum(p.stat().st_size for p in ROOT.glob("*.py"))
        total = size + book + source
        if total > ZIP_CAP_BYTES:
            raise Failure(f"the shipped set would be {total:,} bytes, past {ZIP_CAP_BYTES:,}")
        return (
            f"bucketised file {size:,} bytes ({size / 1e6:.2f} MB), "
            f"{size / WEIGHTS.stat().st_size:.1f}x the 768 file; with the book and the source "
            f"that is {total / 1e6:.1f} MB of the {ZIP_CAP_BYTES / 1e6:.0f} MB cap"
        )


def check_import() -> str:
    """Import time and peak resident memory of a real agent process, against the platform's."""
    script = "import resource, sys, time\nstarted = time.perf_counter()\nimport agent\n" \
             "divisor = 1 << 20 if sys.platform == 'darwin' else 1024\n" \
             "print(f'{time.perf_counter() - started:.2f} " \
             "{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor:.0f}')"
    finished = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        raise Failure(f"importing the agent failed: {finished.stderr[-500:]}")
    seconds, megabytes = finished.stdout.strip().splitlines()[-1].split()
    if float(seconds) > INIT_BUDGET_S:
        raise Failure(f"import takes {seconds}s, past the platform's {INIT_BUDGET_S:.0f}s")
    if float(megabytes) > MEMORY_CAP_MB:
        raise Failure(f"peak resident {megabytes} MB, past the container's {MEMORY_CAP_MB} MB")
    return (
        f"{seconds}s with warm-up of the {INIT_BUDGET_S:.0f}s budget, {megabytes} MB peak "
        f"resident of {MEMORY_CAP_MB} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove the king-bucket scheme exact.")
    parser.add_argument("--full", action="store_true", help="more positions and more plies")
    parser.add_argument("--sequences", type=int, default=0, help="randomised plies to walk")
    arguments = parser.parse_args()
    positions = 10_000 if arguments.full else 2_000
    sequences = arguments.sequences or (30_000 if arguments.full else 10_000)

    rng = random.Random(20260910)
    net = fn.NET
    if not fn.LOADED:
        raise Failure(f"no weight file loaded: {fn.STATUS}")
    print(f"engine: {fn.STATUS}")
    print(f"scheme: {fn.NUM_BUCKETS} buckets x {fn.BASE_FEATURES} = {fn.NUM_FEATURES} inputs, "
          f"first layer {fn.NET[0].nbytes / (1 << 20):.2f} MiB\n")

    print(f"tables:      {check_tables()}")
    print(f"index space: {check_index_space()}")

    fens = sample(rng, positions)
    by_bucket = bucket_positions(rng, max(25, positions // 40))
    every_bucket = [fen for fens_ in by_bucket.values() for fen in fens_]

    print(f"symmetry:    {check_symmetry(fens[:200] + every_bucket[:100], net)}")
    print(f"buckets:     {check_buckets(by_bucket, net)}")
    print(f"warm start:  {check_warm_start(fens + every_bucket, net)}")

    started = time.perf_counter()
    tally = check_reference(fens + every_bucket, net)
    print(f"reference:   {tally['positions']:,} positions, 0 mismatches, in "
          f"{time.perf_counter() - started:.1f} s")
    print("             " + ", ".join(
        f"{name} {count:,}" for name, count in tally.items() if name != "positions"
    ))

    started = time.perf_counter()
    walk = check_increments(rng, sequences, net)
    print(f"increments:  {walk['plies']:,} make/unmake sequences, 0 mismatches, in "
          f"{time.perf_counter() - started:.1f} s")
    print("             " + ", ".join(
        f"{name} {count:,}" for name, count in walk.items() if name != "plies"
    ))

    started = time.perf_counter()
    print(f"distinct:    {check_distinct_blocks(rng, fens[:500] + every_bucket, net)}, in "
          f"{time.perf_counter() - started:.1f} s")
    print(f"rebucket:    {check_rebucket(rng, fens[:300], by_bucket)}")
    print(f"bounds:      {check_bounds(net)}")
    print(f"file:        {check_file(net, fens)}")
    print(f"import:      {check_import()}")
    divisor = 1 << 20 if sys.platform == "darwin" else 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    print(f"\nthis process peaked at {peak:.0f} MB")
    print("the king-bucket scheme is exact, and warm starts to the 768 net it came from.")


if __name__ == "__main__":
    main()
