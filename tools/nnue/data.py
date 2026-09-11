"""Build training shards for the learned evaluation. Offline only; nothing here ships.

Two data paths, one shard format.

  * ``lichess``  reads the public lichess Stockfish evaluations database
                 (https://database.lichess.org/#evals). No engine needed locally.
  * ``selfplay`` generates or reads games and labels sampled positions with a local
                 Stockfish over a pool of worker processes, one engine per worker.

Both write ``.npz`` shards holding exactly three arrays:

    indices  int16 [N, 32]   active feature indices, padded with -1 (see features.py)
    cp       int16 [N]       target, side-to-move relative centipawns, clipped to +-2000
    wdl      int8  [N]       game result from the side to move: +1 win, 0 draw/unknown, -1 loss

``wdl`` is zero for every position whose game result is unknown, which is every position on the
lichess path. train.py does not use it today; it is stored because collecting it later would
cost another full labelling pass.

Run from the repo root:

    uv run python -m tools.nnue.data lichess  --input lichess_db_eval.jsonl.zst --positions 5000000
    uv run python -m tools.nnue.data selfplay --positions 1000000 --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from tools.nnue.features import MAX_ACTIVE, PAD, features

# Targets are clipped here. Past a couple of queens the position is simply winning and the exact
# number carries no signal the search needs; clipping also keeps the target inside int16.
CP_CLIP = 2000
# A forced mate becomes the clip value, so mate and "utterly winning" share a target.
MATE_CP = 2000
# Positions per shard. 500k x 32 x 2 bytes is a ~32 MB file, small enough to write often.
SHARD_POSITIONS = 500_000
# Openings are book knowledge, not evaluation knowledge, so the first few plies are skipped.
SKIP_OPENING_PLIES = 6
# Rough piece values, used only to check the sign convention of an ingested eval file.
_MATERIAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


# --------------------------------------------------------------------------------------
# Shard writing
# --------------------------------------------------------------------------------------


class ShardWriter:
    """Accumulate encoded positions and flush a ``.npz`` every ``shard_size`` rows."""

    def __init__(self, out_dir: Path, shard_size: int = SHARD_POSITIONS) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.written = 0
        self.shards = 0
        self._indices = np.full((shard_size, MAX_ACTIVE), PAD, dtype=np.int16)
        self._cp = np.zeros(shard_size, dtype=np.int16)
        self._wdl = np.zeros(shard_size, dtype=np.int8)
        self._fill = 0

    def add(self, board: chess.Board, cp: int, wdl: int = 0) -> None:
        active = features(board)
        self._indices[self._fill] = PAD
        self._indices[self._fill, : active.size] = active
        self._cp[self._fill] = max(-CP_CLIP, min(CP_CLIP, cp))
        self._wdl[self._fill] = wdl
        self._fill += 1
        self.written += 1
        if self._fill == self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self._fill == 0:
            return
        path = self.out_dir / f"shard_{self.shards:04d}.npz"
        np.savez_compressed(
            path,
            indices=self._indices[: self._fill],
            cp=self._cp[: self._fill],
            wdl=self._wdl[: self._fill],
        )
        self.shards += 1
        self._fill = 0

    def close(self) -> None:
        self.flush()
        print(f"wrote {self.written:,} positions in {self.shards} shard(s) to {self.out_dir}")


def _mirror_horizontal(board: chess.Board) -> chess.Board | None:
    """Return the a-h mirror of ``board``, or None when the mirror is not an equal position.

    The vertical flip with a colour swap is already inside the feature scheme, so mirroring that
    way yields a byte-identical feature vector and no new data at all. The augmentation that is
    actually free is the *horizontal* flip: chess is very nearly symmetric about the d/e file
    boundary, so the evaluation carries over. It is not symmetric while either side can still
    castle, because the flip puts the king on d1 with the rooks in the wrong places, so those
    positions are skipped.
    """
    if board.castling_rights:
        return None
    return board.transform(chess.flip_horizontal)


def _emit(writer: ShardWriter, board: chess.Board, cp: int, wdl: int, mirror: bool) -> None:
    """Write one position, and its horizontal mirror when ``mirror`` and the mirror is sound."""
    writer.add(board, cp, wdl)
    if mirror:
        flipped = _mirror_horizontal(board)
        if flipped is not None:
            writer.add(flipped, cp, wdl)


# --------------------------------------------------------------------------------------
# Path A: the lichess evaluations database
# --------------------------------------------------------------------------------------


def _stream_zst_lines(path: Path) -> Iterator[str]:
    """Yield decompressed lines from a .zst file, stopping cleanly on a truncated prefix.

    Uses the ``zstandard`` package when installed (``uv sync`` installs it from the dev group)
    and otherwise shells out to the ``zstd`` CLI (``brew install zstd``).
    """
    try:
        import zstandard
    except ImportError:
        zstandard = None

    if zstandard is not None:
        with path.open("rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            try:
                yield from io.TextIOWrapper(reader, encoding="utf-8")
            except zstandard.ZstdError:
                # Expected when the file is a `curl -r` prefix: the last frame is incomplete.
                print("note: compressed stream ended mid-frame (a truncated prefix is fine)")
        return

    if shutil.which("zstd") is None:
        raise SystemExit(
            "neither the `zstandard` package nor the `zstd` binary is available.\n"
            "  uv sync            installs zstandard from the dev group\n"
            "  brew install zstd  installs the CLI"
        )
    process = subprocess.Popen(
        ["zstd", "-dc", str(path)], stdout=subprocess.PIPE, text=True, bufsize=1 << 20
    )
    if process.stdout is None:
        raise SystemExit("could not read from zstd")
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()


def _deepest_cp(record: dict[str, object]) -> int | None:
    """Return the deepest eval's best-line score in centipawns, from White's point of view.

    The ``evals`` array is ordered by multipv count rather than by depth, so the deepest entry
    has to be picked out. Within one eval, ``pvs[0]`` is the best line. A line with neither a
    ``cp`` nor a ``mate`` is skipped by returning None.
    """
    evals = record.get("evals")
    if not isinstance(evals, list) or not evals:
        return None
    best = max(evals, key=lambda item: item.get("depth", 0))
    pvs = best.get("pvs")
    if not isinstance(pvs, list) or not pvs:
        return None
    pv = pvs[0]
    if "cp" in pv:
        return int(pv["cp"])
    if "mate" in pv:
        return MATE_CP if int(pv["mate"]) > 0 else -MATE_CP
    return None


def _material_stm(board: chess.Board) -> int:
    """A crude side-to-move-relative material count, used only to check a sign convention."""
    total = 0
    for piece in board.piece_map().values():
        value = _MATERIAL[piece.piece_type]
        total += value if piece.color == board.turn else -value
    return total


def _check_sign(samples: list[tuple[bool, int, int]]) -> None:
    """Fail loudly if ingested labels anti-correlate with material, checking each side alone.

    The lichess file stores ``cp`` from White's point of view and this ingester negates it when
    Black is to move. If that convention is wrong, every label in the dataset silently carries
    the wrong sign and the trained net is worse than useless, so it is worth 2000 positions to
    check. The correlation is weak by nature -- material is a poor evaluation -- but it is
    unambiguously positive.

    White-to-move and Black-to-move positions are checked separately, and that separation is the
    whole point: getting the point of view wrong negates only the Black-to-move half, which
    leaves the pooled correlation weakly positive and hides the bug. Measured on a file that is
    genuinely White-relative, pooling reports +0.67 correctly and +0.06 when told the wrong
    convention -- still positive, still passing. Split by side, the wrong convention shows up as
    a clearly negative correlation on the Black-to-move half.
    """
    for turn, name in ((True, "white to move"), (False, "black to move")):
        subset = [(m, c) for is_white, m, c in samples if is_white == turn]
        if len(subset) < 50:
            print(f"sign check ({name}): only {len(subset)} positions, skipped")
            continue
        material = np.array([m for m, _ in subset], dtype=np.float64)
        label = np.array([c for _, c in subset], dtype=np.float64)
        if material.std() == 0 or label.std() == 0:
            print(f"sign check ({name}): no variance in the sample, skipped")
            continue
        correlation = float(np.corrcoef(material, label)[0, 1])
        print(f"sign check ({name}): corr(material, label) = {correlation:+.3f} "
              f"over {len(subset)} positions")
        if correlation <= 0:
            raise SystemExit(
                f"labels anti-correlate with material for {name}, so the point-of-view "
                "convention is wrong.\n"
                "The lichess file is White-relative (--pov white, the default). Use --pov stm "
                "only for a file that already holds side-to-move-relative scores."
            )


def build_lichess(arguments: argparse.Namespace) -> None:
    source = Path(arguments.input)
    if not source.exists():
        raise SystemExit(f"no such file: {source}")
    writer = ShardWriter(Path(arguments.out), arguments.shard_size)
    started = time.monotonic()
    read = skipped = 0
    samples: list[tuple[bool, int, int]] = []
    checked = False

    for raw_line in _stream_zst_lines(source):
        if writer.written >= arguments.positions:
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if read == 0:
                # A range download that starts mid-file begins on a cut line; skip it.
                print("note: skipped a partial first JSON line (expected on a mid-file range)")
                continue
            # The last line of a truncated prefix download is usually cut in half.
            print("note: stopped at a partial JSON line (expected on a prefix download)")
            break
        read += 1
        cp = _deepest_cp(record)
        if cp is None:
            skipped += 1
            continue
        try:
            board = chess.Board(str(record["fen"]))
        except (ValueError, KeyError):
            skipped += 1
            continue
        # The database holds analysis-board positions too, some of them impossible (eight
        # queens a side, pawns on the back rank). More than 32 men overflows the feature row.
        if not board.is_valid() or board.is_check():
            skipped += 1
            continue
        if arguments.pov == "white" and board.turn == chess.BLACK:
            cp = -cp
        if not checked:
            samples.append(
                (board.turn == chess.WHITE, _material_stm(board),
                 max(-CP_CLIP, min(CP_CLIP, cp)))
            )
            if len(samples) == 2000:
                _check_sign(samples)
                checked = True
        _emit(writer, board, cp, 0, arguments.mirror)
        _progress(writer.written, arguments.positions, started)

    if not checked and samples:
        _check_sign(samples)
    writer.close()
    print(f"read {read:,} records, skipped {skipped:,} without a usable score")


# --------------------------------------------------------------------------------------
# Path B: self-generated games labelled by a local engine
# --------------------------------------------------------------------------------------

# Each worker process owns exactly one engine, opened once in the pool initializer.
_ENGINE: chess.engine.SimpleEngine | None = None
_LABEL_LIMIT: chess.engine.Limit | None = None
_PLAY_LIMIT: chess.engine.Limit | None = None
_SMOKE = False


def _init_worker(engine_path: str, nodes: int, depth: int, play_depth: int, smoke: bool) -> None:
    global _ENGINE, _LABEL_LIMIT, _PLAY_LIMIT, _SMOKE
    _SMOKE = smoke
    if smoke:
        return
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _LABEL_LIMIT = chess.engine.Limit(nodes=nodes) if nodes else chess.engine.Limit(depth=depth)
    _PLAY_LIMIT = chess.engine.Limit(depth=play_depth)


def _shutdown_worker() -> None:
    if _ENGINE is not None:
        _ENGINE.quit()


def _label(board: chess.Board) -> int | None:
    """Return a side-to-move-relative centipawn label, or None when the position has no score."""
    if _SMOKE:
        from tools.nnue.smoke_eval import smoke_evaluate

        return smoke_evaluate(board)
    if _ENGINE is None or _LABEL_LIMIT is None:
        raise RuntimeError("worker engine was never initialised")
    info = _ENGINE.analyse(board, _LABEL_LIMIT)
    score = info.get("score")
    if score is None:
        return None
    return int(score.relative.score(mate_score=MATE_CP))


def _shallow_move(board: chess.Board, rng: random.Random) -> chess.Move:
    """Pick a plausible move cheaply: one shallow engine call, or a one-ply greedy scan."""
    if not _SMOKE and _ENGINE is not None and _PLAY_LIMIT is not None:
        result = _ENGINE.play(board, _PLAY_LIMIT)
        if result.move is not None:
            return result.move
    moves = list(board.legal_moves)
    best_score: int | None = None
    best: list[chess.Move] = []
    for move in moves:
        board.push(move)
        # _label is side-to-move relative and it is now the opponent's move, so negate.
        score = _label(board)
        board.pop()
        if score is None:
            continue
        score = -score
        if best_score is None or score > best_score:
            best_score, best = score, [move]
        elif score == best_score:
            best.append(move)
    return rng.choice(best) if best else rng.choice(moves)


def _play_game(rng: random.Random, random_probability: float, max_plies: int) -> chess.Board:
    """Play one fast game and return the finished board, whose move stack is the game."""
    board = chess.Board()
    while not board.is_game_over(claim_draw=False) and board.ply() < max_plies:
        if rng.random() < random_probability:
            board.push(rng.choice(list(board.legal_moves)))
        else:
            board.push(_shallow_move(board, rng))
    return board


def _result_wdl(result: str, turn: chess.Color) -> int:
    if result == "1-0":
        return 1 if turn == chess.WHITE else -1
    if result == "0-1":
        return -1 if turn == chess.WHITE else 1
    return 0


def _sample_positions(
    finished: chess.Board, rng: random.Random, per_game: int
) -> list[tuple[chess.Board, int]]:
    """Replay the game and return up to ``per_game`` (position, wdl) pairs.

    The first few plies and any position in check are skipped: the opening is book knowledge and
    a position in check has a score dominated by one forced reply rather than by the features.
    """
    outcome = finished.outcome(claim_draw=False)
    result = outcome.result() if outcome is not None else "*"
    # Rated and arena games start from an opening fen, not the standard start.
    replay = finished.root()
    candidates: list[chess.Board] = []
    for move in finished.move_stack:
        replay.push(move)
        if replay.ply() <= SKIP_OPENING_PLIES or replay.is_check():
            continue
        if replay.is_game_over(claim_draw=False):
            continue
        candidates.append(replay.copy(stack=False))
    rng.shuffle(candidates)
    return [(board, _result_wdl(result, board.turn)) for board in candidates[:per_game]]


def _worker(job: tuple[int, float, int, int]) -> list[tuple[str, int, int]]:
    """Play one game, sample positions, label each. Returns (fen, cp, wdl) triples."""
    seed, random_probability, max_plies, per_game = job
    rng = random.Random(seed)
    finished = _play_game(rng, random_probability, max_plies)
    out: list[tuple[str, int, int]] = []
    for board, wdl in _sample_positions(finished, rng, per_game):
        cp = _label(board)
        if cp is None:
            continue
        out.append((board.fen(), max(-CP_CLIP, min(CP_CLIP, cp)), wdl))
    return out


def _pgn_positions(
    pgn_dir: Path, rng: random.Random, per_game: int
) -> Iterator[tuple[chess.Board, int]]:
    """Yield sampled (position, wdl) pairs from every game in every .pgn under ``pgn_dir``."""
    paths = sorted(pgn_dir.glob("**/*.pgn"))
    if not paths:
        raise SystemExit(f"no .pgn files under {pgn_dir}")
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                board = game.end().board()
                if board.move_stack:
                    yield from _sample_positions(board, rng, per_game)


def _resolve_engine(arguments: argparse.Namespace, smoke: bool) -> str:
    engine_path = arguments.stockfish or os.environ.get("STOCKFISH") or "stockfish"
    if smoke:
        print("SMOKE PATH: labelling with the repo's own hand-written evaluate(), not an engine.")
        print("            Enough to wire the pipeline up, useless as a real training target.")
        return engine_path
    if shutil.which(engine_path) is None and not Path(engine_path).exists():
        raise SystemExit(
            f"no stockfish at {engine_path!r}.\n"
            "  brew install stockfish   or pass --stockfish /path, or set $STOCKFISH\n"
            "  --labeller repo-eval     labels with the repo's own evaluate(); smoke tests only"
        )
    return engine_path


def build_selfplay(arguments: argparse.Namespace) -> None:
    smoke = arguments.labeller == "repo-eval"
    engine_path = _resolve_engine(arguments, smoke)
    writer = ShardWriter(Path(arguments.out), arguments.shard_size)
    started = time.monotonic()

    if arguments.pgn_dir:
        _init_worker(engine_path, arguments.nodes, arguments.depth, arguments.play_depth, smoke)
        rng = random.Random(arguments.seed)
        for board, wdl in _pgn_positions(Path(arguments.pgn_dir), rng, arguments.per_game):
            if writer.written >= arguments.positions:
                break
            cp = _label(board)
            if cp is None:
                continue
            _emit(writer, board, cp, wdl, arguments.mirror)
            _progress(writer.written, arguments.positions, started)
        _shutdown_worker()
        writer.close()
        return

    # Bound the job stream: the pool's feeder thread drains whatever it is given, so an endless
    # generator would queue millions of tasks before the first result came back.
    per_position = max(arguments.per_game, 1)
    games = int(arguments.positions / per_position * 1.5) + arguments.workers + 8
    jobs = [
        (arguments.seed + index, arguments.random_probability, arguments.max_plies,
         arguments.per_game)
        for index in range(games)
    ]
    context = mp.get_context("spawn")
    with context.Pool(
        arguments.workers,
        initializer=_init_worker,
        initargs=(engine_path, arguments.nodes, arguments.depth, arguments.play_depth, smoke),
    ) as pool:
        for batch in pool.imap_unordered(_worker, jobs, chunksize=1):
            for fen, cp, wdl in batch:
                _emit(writer, chess.Board(fen), cp, wdl, arguments.mirror)
            _progress(writer.written, arguments.positions, started)
            if writer.written >= arguments.positions:
                break
        pool.terminate()
    writer.close()


# --------------------------------------------------------------------------------------

_LAST_REPORT = 0.0


def _progress(done: int, target: int, started: float, interval: float = 5.0) -> None:
    """Print rate and ETA, at most once every ``interval`` seconds."""
    global _LAST_REPORT
    now = time.monotonic()
    if now - _LAST_REPORT < interval:
        return
    _LAST_REPORT = now
    elapsed = max(now - started, 1e-6)
    rate = done / elapsed
    remaining = max(target - done, 0) / rate if rate > 0 else 0.0
    print(
        f"  {done:,}/{target:,} positions  {rate:,.0f} pos/s  "
        f"elapsed {elapsed / 60:.1f}m  eta {remaining / 60:.1f}m",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build NNUE training shards.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("lichess", "selfplay"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--out", default=f"tools/nnue/data/{name}", help="shard output directory")
        sub.add_argument("--positions", type=int, default=1_000_000)
        sub.add_argument("--shard-size", type=int, default=SHARD_POSITIONS)
        sub.add_argument(
            "--mirror",
            action="store_true",
            help="also emit the a-h mirror of each position (2x data; skips castling positions)",
        )

    lichess = subparsers.choices["lichess"]
    lichess.add_argument("--input", required=True, help="lichess_db_eval.jsonl.zst, or a prefix")
    lichess.add_argument(
        "--pov",
        choices=("white", "stm"),
        default="white",
        help="point of view of the file's cp values; the lichess file uses white",
    )

    selfplay = subparsers.choices["selfplay"]
    selfplay.add_argument(
        "--stockfish", default=None, help="engine path, else $STOCKFISH, else PATH"
    )
    selfplay.add_argument("--nodes", type=int, default=20_000, help="fixed node budget per label")
    selfplay.add_argument("--depth", type=int, default=9, help="used when --nodes 0")
    selfplay.add_argument("--play-depth", type=int, default=4, help="depth for generating moves")
    selfplay.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    selfplay.add_argument("--per-game", type=int, default=10, help="positions sampled per game")
    selfplay.add_argument("--max-plies", type=int, default=200)
    selfplay.add_argument("--random-probability", type=float, default=0.25)
    selfplay.add_argument("--seed", type=int, default=1)
    selfplay.add_argument("--pgn-dir", default=None, help="label positions from PGNs instead")
    selfplay.add_argument(
        "--labeller",
        choices=("stockfish", "repo-eval"),
        default="stockfish",
        help="repo-eval is a SMOKE-ONLY stand-in using the repo's own evaluate()",
    )

    arguments = parser.parse_args(argv)
    if arguments.command == "lichess":
        build_lichess(arguments)
    else:
        build_selfplay(arguments)


if __name__ == "__main__":
    sys.exit(main())
