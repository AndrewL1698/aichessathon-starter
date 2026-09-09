"""Prove the opening book is a book the agent can play out of.

`uv run python -m tests.test_book`.

A book is a file of positions and moves that nothing in the engine checks: the search cannot
tell a good book move from a corrupt one, and an illegal move out of the book loses a game the
same way an illegal move out of the search does. So the load-bearing tests here are about the
file rather than about strength.

The first walks the whole book and proves every entry is a move that is legal in the position
it is filed under, which is the assertion that the polyglot encoding, the castling convention
(`e1h1`, not `e1g1`) and the zobrist keys are all the ones `chess.polyglot` will read back.
The second proves the book actually reaches into each of the sample openings the ladder plays,
because a book that only knows the standard start position would never be used: rated games
start from curated positions. The rest is what a broken book still passes those by doing:
never being consulted, being consulted and then not committed to the game history, refusing to
give the search the position back when it runs out, or printing a line the log reader drops.
"""

import argparse
import contextlib
import io
import itertools
import re
import sys
import time
from collections import deque

import chess
import chess.polyglot

import agent
import fastsearch as fs
from harness import readlog
from harness.rules import OPENINGS

# How deep the book has to run from each position it is built from, and why that is not one
# number. The counts in the comments are master games in the 1.13M game corpus that pass
# through the position, printed by `tools/book/build.py` on every build.
#
# Rated games never start from the standard position, so the openings are what a book is for
# here - and four of the eight the ladder publishes as samples occur in no master game at all.
# They are curated to be close to level, not to be theory, and a book of human games cannot
# answer a position humans never played. That is a fact about the platform's positions, not a
# defect in the builder, so this asserts what the corpus can support and prints the rest.
EXPECTED_PLIES: dict[str, int] = {
    "start": 12,  # 1,133,202
    "English Opening": 0,  # 3
    "French Winawer": 3,  # 127
    "Petroff Defence": 0,  # 0
    "Scotch Game": 0,  # 0
    "Grunfeld Defence": 6,  # 651
    "French Classical": 0,  # 0
    "Sicilian Closed": 4,  # 44
    "Sicilian Sveshnikov": 6,  # 1,373
}
# How far into the book a game from the standard position has to stay, which is the depth the
# runtime path is exercised over.
WANTED_PLIES = 6
# The lookup happens before the clock is consulted, so it has to be free. Five milliseconds is
# a hundredth of the shortest budget the engine ever gives itself.
BUDGET_MS = 5.0
# A middlegame position no master game reaches, which is where the book has to stop and hand
# the position to the search.
OFF_BOOK = "r2q1rk1/1b1nbppp/p2ppn2/1p4B1/3NP3/1BN5/PPP1QPPP/R4RK1 w - - 2 13"


class Failure(Exception):
    """The book, or the agent's use of it, did something a book may not do."""


def roots() -> list[tuple[str, chess.Board]]:
    """Every position the book is built from: the standard start and the sample openings."""
    return [("start", chess.Board())] + [(name, chess.Board(fen)) for name, fen in OPENINGS]


def check_entries() -> str:
    """Walk the book from its roots and prove every entry in the file is legal where it sits.

    Legality is checked twice over, because `find_all` quietly drops an entry whose move is
    illegal in the position rather than raising: once by counting what `find_all` yields for
    the position against the raw entries filed under that key, which catches the dropped ones,
    and once with `is_legal` on what it does yield.

    The walk following the book's own moves is what makes this a claim about the whole file:
    the entries reached this way are counted, and the count has to be every entry the file
    has. An entry no line of book play can reach is a defect in the builder, and this fails on
    it rather than leaving it unchecked.
    """
    with chess.polyglot.open_reader(agent.BOOK_PATH) as reader:
        filed: dict[int, int] = {}
        for entry in reader:
            filed[entry.key] = filed.get(entry.key, 0) + 1

        seen: set[int] = set()
        queue: deque[chess.Board] = deque(board for _, board in roots())
        checked = 0
        deepest = 0
        while queue:
            board = queue.popleft()
            key = chess.polyglot.zobrist_hash(board)
            if key in seen or key not in filed:
                continue
            seen.add(key)
            entries = list(reader.find_all(board))
            if len(entries) != filed[key]:
                raise Failure(
                    f"{filed[key] - len(entries)} of {filed[key]} entries for {board.fen()!r} "
                    "decode to a move that is not legal there"
                )
            for entry in entries:
                if not board.is_legal(entry.move):
                    raise Failure(f"{entry.move.uci()} is not legal in {board.fen()!r}")
                if entry.weight < 1:
                    raise Failure(f"{entry.move.uci()} in {board.fen()!r} has weight 0")
                checked += 1
                board.push(entry.move)
                deepest = max(deepest, board.ply())
                queue.append(board.copy(stack=False))
                board.pop()
        if checked != len(reader):
            raise Failure(
                f"{len(reader) - checked} of {len(reader)} entries are not reachable by "
                "playing the book's own moves, so nothing here checked them"
            )
        return (
            f"{checked:,} entries over {len(seen):,} positions, all legal, "
            f"deepest position at ply {deepest}"
        )


def check_coverage() -> str:
    """Every root reaches at least the depth master practice supports for it.

    Following the most popular move, which is the line the book will actually be played down
    most often, and checking each move is legal on the way: an entry filed under the right key
    with the wrong move would pass the legality walk and still lose a game here.
    """
    depths = []
    with chess.polyglot.open_reader(agent.BOOK_PATH) as reader:
        for name, start in roots():
            board = start.copy(stack=False)
            plies = 0
            while True:
                entry = reader.get(board)
                if entry is None:
                    break
                if not board.is_legal(entry.move):
                    raise Failure(f"{entry.move.uci()} is not legal in {board.fen()!r}")
                board.push(entry.move)
                plies += 1
            depths.append((name, plies))
    thin = [
        f"{name} {plies} of {EXPECTED_PLIES[name]}"
        for name, plies in depths
        if plies < EXPECTED_PLIES[name]
    ]
    if thin:
        raise Failure(f"the book lost depth it had: {', '.join(thin)}")
    listed = ", ".join(f"{name} {plies}" for name, plies in depths)
    empty = sum(1 for _, plies in depths if plies == 0)
    return f"{listed} ({empty} of the sample openings are in no master game)"


def _played(fen: str, time_left_ms: int) -> tuple[str, str]:
    """One move from `agent.get_move`, with what it printed."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        move = agent.get_move(fen, time_left_ms)
    return move, output.getvalue()


def _fresh() -> None:
    """Both engines back to knowing nothing, the way a new game starts them."""
    fs.reset()
    agent._MEMORY.table.clear()
    agent._MEMORY.seen.clear()
    agent._MEMORY.expected = None
    agent._MEMORY.history = [0] * len(agent._MEMORY.history)


def check_game() -> str:
    """Play a game through `get_move` and prove the book moves are real, committed moves.

    Both sides come from `get_move`, so this is the sequence the platform would see, and every
    move has to be legal and has to reach the fast engine's game history. That history is the
    point: a book move that never gets to `fastsearch.remember_played` leaves the engine
    describing a game that did not happen, and the first searched move of the game throws away
    the table and every position the game stood in. It grows by one per move, so its final
    value is the ply count plus the position the game started from.
    """
    _fresh()
    board = chess.Board()
    counts = [int(fs.STATS[fs.GAME_COUNT])]
    book_plies = 0
    for _ in range(WANTED_PLIES + 2):
        move, printed = _played(board.fen(), 60_000)
        if move not in {candidate.uci() for candidate in board.legal_moves}:
            raise Failure(f"get_move returned {move} in {board.fen()!r}, which is not legal")
        if " book " in printed:
            book_plies += 1
        counts.append(int(fs.STATS[fs.GAME_COUNT]))
        board.push(chess.Move.from_uci(move))
    if book_plies < WANTED_PLIES:
        raise Failure(f"only {book_plies} of the first {len(counts) - 1} moves came from book")
    for earlier, later in itertools.pairwise(counts):
        if later <= earlier:
            raise Failure(f"the game history did not grow across a move: {counts}")
    if counts[-1] != board.ply() + 1:
        raise Failure(
            f"the fast engine remembers {counts[-1]} positions of a {board.ply()} ply game"
        )
    if fs._EXPECTED is None:
        raise Failure("nothing set the fast engine's expected position after a book move")
    return (
        f"{book_plies} of the first {len(counts) - 1} moves out of book, "
        f"history {counts[0]} -> {counts[-1]} over {board.ply()} plies"
    )


def check_off_book() -> str:
    """Off book, the position goes to the search and the book is not mentioned."""
    _fresh()
    if agent._book_move(OFF_BOOK, 5_000) is not None:
        raise Failure(f"the book claims to know {OFF_BOOK!r}")
    move, printed = _played(OFF_BOOK, 3_000)
    if move not in {candidate.uci() for candidate in chess.Board(OFF_BOOK).legal_moves}:
        raise Failure(f"get_move returned {move} in {OFF_BOOK!r}, which is not legal")
    if " book " in printed:
        raise Failure(f"a book move was printed for an off-book position: {printed.strip()!r}")
    last = printed.strip().splitlines()[-1]
    depth = re.match(r"d(\d+) score ", last)
    if depth is None or int(depth[1]) < 1:
        raise Failure(f"nothing here searched the position: {last!r}")
    # Not `readlog.OUTPUT_LINE` on purpose. The numba search's line has not matched it since
    # v3.0: it prints `tt 31%` where the reader wants an integer and an extra `null 0` field,
    # so `harness.readlog` files every searched move under "other output" and reports a total
    # log gap. That is a drift in `harness/`, which this branch may not edit, and it is why
    # `check_log_line` asserts the reader on the book's line, which does match.
    return f"searched to depth {depth[1]} and played {move}"


def check_ply_cap() -> str:
    """Past `BOOK_MAX_PLY` the book is not consulted, even where it still has moves.

    The cap only means something in a position the book could answer and refuses to, and the
    standard position's lines stop at ply 16, inside the cap. The sample openings start as
    deep as ply 17, so their lines run past it: this finds such a position by following the
    book down from each root, and fails rather than passing quietly if there is none, because
    then nothing here tested the cap.
    """
    past: chess.Board | None = None
    with chess.polyglot.open_reader(agent.BOOK_PATH) as reader:
        for _, start in roots():
            board = start.copy(stack=False)
            while (entry := reader.get(board)) is not None:
                if board.ply() > agent.BOOK_MAX_PLY:
                    past = board
                    break
                board.push(entry.move)
            if past is not None:
                break
    if past is None:
        raise Failure(
            f"no line in the book reaches past ply {agent.BOOK_MAX_PLY}, so the cap is untested"
        )
    known = len(list(agent._BOOK.find_all(past))) if agent._BOOK is not None else 0
    if agent._book_move(past.fen(), 60_000) is not None:
        raise Failure(f"the book answered at ply {past.ply()}, past the cap")
    return f"silent at ply {past.ply()}, a position it holds {known} entries for"


def check_log_line() -> str:
    """The line a book move prints has to survive `harness.readlog`, like a searched move.

    Parsed through the real log reader, not the regex alone: what the platform hands back is a
    log with sections, and a line the reader files under "other output" is a move that has
    dropped out of every per-move figure the reader prints.
    """
    _fresh()
    move, printed = _played(chess.Board().fen(), 90_000)
    line = printed.strip().splitlines()[-1]
    if " book " not in line:
        raise Failure(f"the first move of the game did not come from the book: {line!r}")
    log = readlog.parse(f"MOVES\n\nOUTPUT\n{line}\n")
    if not log.printed:
        raise Failure(f"readlog did not parse the book line: {line!r}")
    if log.other_lines:
        raise Failure(f"readlog filed the book line as other output: {log.other_lines}")
    read = log.printed[0]
    if read.move != move or read.depth != 0 or read.score is not None:
        raise Failure(f"readlog read {read} out of {line!r}")
    return f"{line!r} parses as move {read.move} at depth {read.depth}"


def check_speed(repeats: int) -> str:
    """A lookup has to cost nothing next to a search, on every root and off book too."""
    positions = [board.fen() for _, board in roots()] + [OFF_BOOK]
    worst, worst_fen = 0.0, ""
    for fen in positions:
        started = time.perf_counter()
        for _ in range(repeats):
            with contextlib.redirect_stdout(io.StringIO()):
                agent._book_move(fen, 60_000)
        spent = (time.perf_counter() - started) * 1000.0 / repeats
        if spent > worst:
            worst, worst_fen = spent, fen
    if worst > BUDGET_MS:
        raise Failure(f"a lookup cost {worst:.2f} ms in {worst_fen!r}, over {BUDGET_MS} ms")
    return f"worst lookup {worst:.3f} ms over {len(positions)} positions, budget {BUDGET_MS} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the opening book.")
    parser.add_argument("--repeats", type=int, default=200)
    arguments = parser.parse_args()

    if agent._BOOK is None:
        raise Failure(f"there is no book at {agent.BOOK_PATH}")
    size = agent.BOOK_PATH.stat().st_size
    print(f"{agent.BOOK_PATH.name}: {len(agent._BOOK):,} entries, {size:,} bytes")

    print(f"\nlegality: {check_entries()}")
    print(f"coverage: {check_coverage()}")
    print(f"speed: {check_speed(arguments.repeats)}")
    print(f"ply cap: {check_ply_cap()}")
    print(f"log line: {check_log_line()}")
    print(f"game: {check_game()}")
    print(f"off book: {check_off_book()}")

    print("\nThe book is legal everywhere, deep enough to be used, and free to read.")


if __name__ == "__main__":
    try:
        main()
    except Failure as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
