"""Build `weights/book.bin`, a polyglot opening book, from human master games.

    uv run python tools/book/build.py                    # the shipped book
    uv run python tools/book/build.py --files 8 --out /tmp/small.bin   # a quick one

Offline only. Nothing here ships: `harness/package.py` zips root modules, the directories they
import, and `weights/`, and no root module imports `tools`. The agent reads the `.bin` this
writes and never this file.

## Where the moves come from

The brief for this book was the Lichess masters opening explorer
(`https://explorer.lichess.ovh/masters`). That service answers 401 to every request from here,
on both `/masters` and `/lichess`, with and without a token, so it could not be used; see
`docs/BENCH_LOG.md`. The corpus instead is PGN Mentor's opening collections
(`https://www.pgnmentor.com/openings/<name>.zip`), which are over-the-board games between
titled players, one zip per opening variation.

Provenance is the point, not convenience. `AGENTS.md` allows an opening book and forbids "a
table of engine moves or evaluations for the agent to look up while it plays". Every move in
this book is a move a human master played in a real game, counted by how often it was played;
no engine, ours or anyone's, is consulted anywhere in this file.

## The shape of the book

Breadth-first from the standard start position and from each of the eight sample openings in
`harness/rules.py`, keeping at most `--max-moves` moves per position, each of which needs
`--min-games` games and `--min-share` of the most-played move's count. `--min-games` defaults
well below the brief's 100 because that number was calibrated against the explorer's ~2.5M
game database: this corpus is a few hundred thousand games, and 100 there is a dozen here.

Raw downloads are cached under `tools/book/cache/` (gitignored), so a rebuild re-reads the
zips and never re-fetches them.
"""

import argparse
import random
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from pathlib import Path

import chess
import chess.polyglot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from harness.rules import OPENINGS

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = Path(__file__).resolve().parent / "cache"
INDEX_URL = "https://www.pgnmentor.com/files.html"
OPENING_URL = "https://www.pgnmentor.com/{name}"
AGENT = "aichessathon-book-builder/1.0 (one request at a time)"

# Polite fetching: one request at a time, a pause between them, and a doubling wait when the
# server says no. Nothing here is time critical and the site is someone else's.
DELAY_S = 0.4
RETRIES = 5
BACKOFF_S = 5.0
TIMEOUT_S = 120

# Polyglot's entry: an 8 byte key, the move, a weight, and a learn field we leave at zero.
ENTRY = struct.Struct(">QHHI")
MAX_WEIGHT = 65535

# A game's opening is over long before this, and the agent stops using the book at ply 20.
# The extra plies are for the sample openings, which already start as deep as ply 17.
MAX_PLY = 24
MOVE_NUMBER = re.compile(r"^\d+\.+")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def sources(limit: int | None) -> list[str]:
    """The opening collections to build from, newest-listed first, read off the file index."""
    index = _fetch(INDEX_URL, CACHE / "files.html").decode("latin-1")
    names = sorted(set(re.findall(r'href="(openings/[^"/]+\.zip)"', index)))
    return names[:limit] if limit else names


def _fetch(url: str, path: Path) -> bytes:
    """Download `url` to `path` once. A cached file is never re-fetched."""
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    wait = BACKOFF_S
    for attempt in range(RETRIES):
        request = urllib.request.Request(url, headers={"User-Agent": AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                body: bytes = response.read()
            path.write_bytes(body)
            time.sleep(DELAY_S)
            return body
        except (urllib.error.URLError, TimeoutError, OSError) as failure:
            code = getattr(failure, "code", None)
            retryable = code is None or code == 429 or code >= 500
            if not retryable or attempt == RETRIES - 1:
                raise SystemExit(f"{url}: {failure}") from failure
            print(f"  {url}: {failure}, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
            wait *= 2
    raise SystemExit(f"{url}: out of retries")


def games(archive: Path, limit: int) -> list[list[str]]:
    """The last `limit` games in a collection, as lists of SAN tokens.

    The last ones rather than the first: the collections are in date order, so the tail is
    modern master practice and the head is the nineteenth century.
    """
    with zipfile.ZipFile(archive) as bundle:
        name = next(item for item in bundle.namelist() if item.endswith(".pgn"))
        text = bundle.read(name).decode("latin-1").replace("\r\n", "\n")
    found: list[list[str]] = []
    movetext: list[str] = []
    for line in text.split("\n"):
        if line.startswith("["):
            if movetext:
                found.append(_tokens(" ".join(movetext)))
                movetext = []
        elif line.strip():
            movetext.append(line.strip())
    if movetext:
        found.append(_tokens(" ".join(movetext)))
    return [moves for moves in found[-limit:] if moves]


def _tokens(movetext: str) -> list[str]:
    """The first `MAX_PLY` SAN moves of a game, with move numbers and results dropped.

    These collections write `1.e4 c5`, with no space after the number, so the number is
    stripped off the front of a token rather than the token being thrown away.
    """
    moves = []
    for token in movetext.split():
        san = MOVE_NUMBER.sub("", token).rstrip("!?+#")
        if not san or san in RESULTS or san.startswith(("$", "{", "(", "-")):
            continue
        moves.append(san)
        if len(moves) >= MAX_PLY:
            break
    return moves


def encode(board: chess.Board, move: chess.Move) -> int:
    """Polyglot's 16 bit move: to, from, and the promotion piece, castling as king takes rook.

    `chess.polyglot`'s reader turns the king-takes-rook encoding back into `e1g1` using the
    position, so the book has to be written its way round or castling comes back illegal.
    """
    to_square = move.to_square
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_file = 7 if kingside else 0
        to_square = chess.square(rook_file, rank)
    promotion = 0 if move.promotion is None else move.promotion - 1
    return to_square | (move.from_square << 6) | (promotion << 12)


def tally(archives: list[Path], per_file: int) -> dict[tuple[int, int], int]:
    """Count how often each move was played in each position, over every game we have.

    Keyed on the polyglot hash of the position and the polyglot encoding of the move, so a
    position reached by two different move orders is one entry, and so the counts can be read
    back later by asking a board for its legal moves and encoding each one.
    """
    counts: dict[tuple[int, int], int] = {}
    total = 0
    for index, archive in enumerate(archives, 1):
        for moves in games(archive, per_file):
            board = chess.Board()
            try:
                for san in moves:
                    move = board.parse_san(san)
                    key = (chess.polyglot.zobrist_hash(board), encode(board, move))
                    counts[key] = counts.get(key, 0) + 1
                    board.push(move)
            except (chess.IllegalMoveError, chess.AmbiguousMoveError, chess.InvalidMoveError):
                pass  # A malformed game contributes the plies it got right and no more.
            total += 1
        if index % 25 == 0 or index == len(archives):
            print(
                f"  {index}/{len(archives)} collections, {total:,} games, "
                f"{len(counts):,} position-moves",
                flush=True,
            )
    return counts


def roots() -> list[tuple[str, chess.Board]]:
    """Where the book starts: the standard position and every sample opening."""
    found = [("start", chess.Board())]
    found += [(name, chess.Board(fen)) for name, fen in OPENINGS]
    return found


def select(
    counts: dict[tuple[int, int], int], min_games: int, min_share: float, max_moves: int,
    max_depth: int,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, chess.Board]]:
    """Breadth-first from every root, keeping the moves masters actually played.

    Returns the entries, keyed by position, and one board per position, which is what proves
    later that every entry in the file is legal where it is filed.
    """
    entries: dict[int, list[tuple[int, int]]] = {}
    boards: dict[int, chess.Board] = {}
    queue: deque[tuple[chess.Board, int]] = deque()
    for _, board in roots():
        queue.append((board.copy(stack=False), 0))
    while queue:
        board, depth = queue.popleft()
        key = chess.polyglot.zobrist_hash(board)
        if key in entries or depth >= max_depth or board.ply() > MAX_PLY:
            continue
        played = [
            (move, counts[(key, encode(board, move))])
            for move in board.legal_moves
            if (key, encode(board, move)) in counts
        ]
        if not played:
            continue
        best = max(count for _, count in played)
        kept = sorted(
            (
                (move, count)
                for move, count in played
                if count >= min_games and count >= best * min_share
            ),
            key=lambda pair: -pair[1],
        )[:max_moves]
        if not kept:
            continue
        entries[key] = [(encode(board, move), count) for move, count in kept]
        boards[key] = board
        for move, _ in kept:
            board.push(move)
            queue.append((board.copy(stack=False), depth + 1))
            board.pop()
    return entries, boards


def write(entries: dict[int, list[tuple[int, int]]], path: Path) -> int:
    """Write the book: sorted by key, and by weight descending inside a key.

    A count can be larger than the two bytes a weight has, so every count is scaled by the
    same factor and floored at one. Only the ratios inside a position matter to the agent,
    and one factor for the whole book keeps those intact.
    """
    largest = max(count for moves in entries.values() for _, count in moves)
    scale = min(1.0, MAX_WEIGHT / largest)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("wb") as handle:
        for key in sorted(entries):
            for raw, count in sorted(entries[key], key=lambda pair: -pair[1]):
                weight = max(1, min(MAX_WEIGHT, round(count * scale)))
                handle.write(ENTRY.pack(key, raw, weight, 0))
                written += 1
    return written


def verify(path: Path, boards: dict[int, chess.Board]) -> int:
    """Read the book back and prove every entry is a legal move in the position it is filed on.

    Read back through `chess.polyglot`, not through the writer's own idea of the format: this
    is the check that the move encoding, the castling convention and the sort order are the
    ones the agent's reader will see.
    """
    checked = 0
    with chess.polyglot.open_reader(path) as reader:
        previous = -1
        for entry in reader:
            if entry.key < previous:
                raise SystemExit("the book is not sorted by key")
            previous = entry.key
            board = boards.get(entry.key)
            if board is None:
                raise SystemExit(f"entry {entry.key:016x} belongs to no position we built")
            # The reader's own normalisation, which is what turns e1h1 back into e1g1.
            move = board._from_chess960(
                False, entry.move.from_square, entry.move.to_square, entry.move.promotion
            )
            if not board.is_legal(move):
                raise SystemExit(f"{move.uci()} is not legal in {board.fen()}")
            checked += 1
    return checked


def coverage(path: Path) -> list[tuple[str, int]]:
    """How many plies of the most popular line each root has in the book."""
    found = []
    with chess.polyglot.open_reader(path) as reader:
        for name, start in roots():
            board = start.copy(stack=False)
            plies = 0
            while True:
                entry = reader.get(board)
                if entry is None:
                    break
                board.push(entry.move)
                plies += 1
            found.append((name, plies))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "book.bin")
    parser.add_argument("--files", type=int, default=0, help="use only the first N collections")
    parser.add_argument("--games-per-file", type=int, default=1500)
    parser.add_argument("--min-games", type=int, default=12)
    parser.add_argument("--min-share", type=float, default=0.15)
    parser.add_argument("--max-moves", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=16)
    arguments = parser.parse_args()
    random.seed()  # Nothing here is random; this is only so a stray sample cannot be seeded.

    names = sources(arguments.files or None)
    print(f"{len(names)} opening collections from pgnmentor.com, cache {CACHE}")
    archives = [Path(_cache_path(name)) for name in names]
    for name, archive in zip(names, archives, strict=True):
        if not archive.exists():
            print(f"  fetching {name}", flush=True)
        _fetch(OPENING_URL.format(name=name), archive)

    started = time.perf_counter()
    counts = tally(archives, arguments.games_per_file)
    print(f"counted in {time.perf_counter() - started:.0f}s")

    entries, boards = select(
        counts, arguments.min_games, arguments.min_share, arguments.max_moves,
        arguments.max_depth,
    )
    written = write(entries, arguments.out)
    checked = verify(arguments.out, boards)
    size = arguments.out.stat().st_size
    print(
        f"{arguments.out}: {written:,} entries over {len(entries):,} positions, "
        f"{size:,} bytes, {checked:,} verified legal"
    )
    for name, plies in coverage(arguments.out):
        print(f"  {name:<22} {plies} plies of book on the most popular line")


def _cache_path(name: str) -> Path:
    return CACHE / name.split("/")[-1]


if __name__ == "__main__":
    main()
