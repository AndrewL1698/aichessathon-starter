"""Add the hand evaluation to every shard, so a net can be trained as a correction on top of it.

    uv run python -m tools.nnue.hand --data tools/nnue/data/lichess-48m \
        --out tools/nnue/data/lichess-48m-hand --engine-dir .

Reads every shard under ``--data``, computes ``fasteval.evaluate`` for each row and writes the
same shard with one more array, ``hand`` (int16, side-to-move centipawns, clipped like ``cp``).
``train.py --target residual`` then fits ``sigmoid((hand + net) / cp_scale)`` to the same target
as before, so the net learns only what the hand evaluation gets wrong and material sanity is
the hand evaluation's by construction.

The board is rebuilt from the feature row rather than from a fen, because the shards hold no
fen. A feature row is the position seen from the side to move, with "our" men on planes 0-5;
placing those as White on the encoded squares, "their" men as Black, and calling the evaluation
with White to move gives exactly the number the real position gets with the real side to move,
because the evaluation is colour-symmetric and reads nothing the row does not carry: no
castling rights, no en passant square, no move counters. That claim is checked, not assumed:
``--verify FILE.jsonl.zst`` encodes real lichess records both ways and demands equality.

``--engine-dir`` is where ``fastboard.py`` and ``fasteval.py`` are imported from, so the
training pipeline never has to live on the same branch as the engine. Neither module is
modified; the commit they came from belongs in the run note.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from tools.nnue.features import MAX_ACTIVE, PAD

CP_CLIP = 2000


def _import_engine(engine_dir: Path) -> tuple[object, object]:
    sys.path.insert(0, str(engine_dir.resolve()))
    import fastboard
    import fasteval

    return fastboard, fasteval


def _build_batch_evaluator(fasteval: object) -> object:
    evaluate = fasteval.evaluate  # type: ignore[attr-defined]

    @njit(cache=False)
    def hand_batch(indices: np.ndarray, out: np.ndarray) -> None:
        board = np.zeros(120, dtype=np.int8)
        for i in range(120):
            if i < 21 or i > 98 or i % 10 == 0 or i % 10 == 9:
                board[i] = 13
        st = np.zeros(9, dtype=np.int64)
        for row in range(indices.shape[0]):
            for i in range(21, 99):
                if board[i] != 13:
                    board[i] = 0
            st[:] = 0
            st[4] = 1
            for j in range(MAX_ACTIVE):
                index = indices[row, j]
                if index < 0:
                    continue
                plane = index // 64
                square = index % 64
                mailbox = 21 + (7 - square // 8) * 10 + square % 8
                piece = plane + 1 if plane < 6 else plane - 6 + 7
                board[mailbox] = piece
                if piece == 6:
                    st[5] = mailbox
                elif piece == 12:
                    st[6] = mailbox
            score = evaluate(board, st)
            out[row] = max(-CP_CLIP, min(CP_CLIP, score))

    return hand_batch


def verify(path: Path, fastboard: object, fasteval: object, hand_batch: object, limit: int) -> None:
    """Encode real records both ways and demand the same hand evaluation."""
    import chess

    from tools.nnue.data import _stream_zst_lines
    from tools.nnue.features import features

    rows = np.full((limit, MAX_ACTIVE), PAD, dtype=np.int16)
    direct = np.zeros(limit, dtype=np.int64)
    count = 0
    for line in _stream_zst_lines(path):
        record = json.loads(line)
        board = chess.Board(str(record["fen"]))
        if not board.is_valid() or board.is_check():
            continue
        active = features(board)
        rows[count, : active.size] = active
        fb_board, fb_st, _ = fastboard.from_fen(board.fen())  # type: ignore[attr-defined]
        direct[count] = fasteval.evaluate(fb_board, fb_st)  # type: ignore[attr-defined]
        count += 1
        if count == limit:
            break
    rebuilt = np.zeros(count, dtype=np.int16)
    hand_batch(rows[:count], rebuilt)
    clipped = np.clip(direct[:count], -CP_CLIP, CP_CLIP)
    mismatches = int((clipped != rebuilt).sum())
    print(f"verify: {count} positions, {mismatches} mismatches between fen and feature-row paths")
    if mismatches:
        raise SystemExit("the hand evaluation reads something the feature row does not carry")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="shard directory to read")
    parser.add_argument("--out", required=True, help="shard directory to write")
    parser.add_argument("--engine-dir", default=".", help="where fastboard.py and fasteval.py are")
    parser.add_argument("--verify", default=None, help="a lichess .jsonl.zst to cross-check on")
    parser.add_argument("--verify-count", type=int, default=20_000)
    arguments = parser.parse_args(argv)

    fastboard, fasteval = _import_engine(Path(arguments.engine_dir))
    hand_batch = _build_batch_evaluator(fasteval)
    if arguments.verify:
        verify(Path(arguments.verify), fastboard, fasteval, hand_batch, arguments.verify_count)

    out_dir = Path(arguments.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(arguments.data).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no .npz shards under {arguments.data}")
    started = time.monotonic()
    total = 0
    for path in paths:
        with np.load(path) as shard:
            indices = shard["indices"]
            cp = shard["cp"]
            wdl = shard["wdl"]
        hand = np.zeros(indices.shape[0], dtype=np.int16)
        hand_batch(indices, hand)
        np.savez_compressed(out_dir / path.name, indices=indices, cp=cp, wdl=wdl, hand=hand)
        total += indices.shape[0]
    seconds = time.monotonic() - started
    print(f"wrote {total:,} rows with hand evaluation to {out_dir} in {seconds:.0f}s")


if __name__ == "__main__":
    main()
