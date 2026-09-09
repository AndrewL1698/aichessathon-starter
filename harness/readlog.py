"""Read a competition log, and the PGN beside it, and say what the game tells us.

    uv run python -m harness.readlog logs/73-castling.log [logs/73-castling.pgn]

The platform's log has a MOVES table (its own timing of every move) and an OUTPUT section
with what we printed: one line per move from `agent.py`. Only the first and last 4 KB of our
output survive, so a long game loses its middle. Nothing here infers anything about the gap:
the lines that are missing are reported as missing.
"""

import argparse
import contextlib
import io
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
LOCAL_AGENT = ROOT / "local-opponents" / "v2.4"
# The score change that counts as a swing, in centipawns, and the score that means a mate.
SWING_CP = 150
MATE_CP = 900_000
# How the shipped engine derives its budgets, so a log's move times can be judged against them.
HARD_DIVISOR = 8
SAFETY_MARGIN_MS = 300

MOVE_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+([\d.]+) s\s+([\d.]+) s\s*$")
OUTPUT_LINE = re.compile(
    r"^d(?P<depth>\d+)(?: score (?P<score>[+-]\d+))? move (?P<move>\S+)"
    r"(?P<partial> from partial d\d+)? nodes (?P<nodes>\d+)"
    r"(?: nps (?P<nps>\d+))? (?P<ms>\d+)ms soft (?P<soft>\d+) hard (?P<hard>\d+)"
    r" clock (?P<clock>\d+) tt (?P<tt>\d+) cut \d+ contempt [+-]\d+ peakrss (?P<rss>\d+)MB"
)
DROPPED = re.compile(r"\[(\d[\d,]*) bytes dropped\]")


@dataclass(frozen=True)
class Printed:
    """One line of our own output, as parsed."""

    depth: int
    score: int | None
    move: str
    partial: bool
    nodes: int
    nps: int | None
    ms: int
    soft: int
    hard: int
    clock: int
    table: int
    rss_mb: int


@dataclass(frozen=True)
class Timed:
    """One row of the platform's MOVES table."""

    number: int
    san: str
    seconds: float
    clock_left: float


@dataclass
class Log:
    fields: dict[str, str]
    moves: list[Timed]
    printed: list[Printed]
    tracebacks: int
    dropped_bytes: int
    other_lines: list[str]


def parse(text: str) -> Log:
    fields: dict[str, str] = {}
    moves: list[Timed] = []
    printed: list[Printed] = []
    other: list[str] = []
    tracebacks = 0
    dropped = 0
    section = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line and not line.startswith(" ") and line.isupper():
            section = line
            continue
        stripped = line.strip()
        if not stripped or set(stripped) == {"="}:
            continue
        if section == "MOVES":
            row = MOVE_ROW.match(line)
            if row:
                moves.append(Timed(int(row[1]), row[2], float(row[3]), float(row[4])))
            continue
        if section == "OUTPUT":
            found = OUTPUT_LINE.match(stripped)
            if found:
                printed.append(
                    Printed(
                        depth=int(found["depth"]),
                        score=int(found["score"]) if found["score"] else None,
                        move=found["move"],
                        partial=found["partial"] is not None,
                        nodes=int(found["nodes"]),
                        nps=int(found["nps"]) if found["nps"] else None,
                        ms=int(found["ms"]),
                        soft=int(found["soft"]),
                        hard=int(found["hard"]),
                        clock=int(found["clock"]),
                        table=int(found["tt"]),
                        rss_mb=int(found["rss"]),
                    )
                )
                continue
            if stripped.startswith("Traceback"):
                tracebacks += 1
            gap = DROPPED.search(stripped)
            if gap:
                dropped += int(gap[1].replace(",", ""))
            if not stripped.startswith("Email "):
                other.append(stripped)
            continue
        if section == "RESULT":
            fields["Result"] = (fields.get("Result", "") + " " + stripped).strip()
            continue
        key, _, value = stripped.partition("  ")
        if value:
            fields[key.strip()] = value.strip()
    return Log(fields, moves, printed, tracebacks, dropped, other)


def _our_colour(log: Log) -> chess.Color:
    return chess.WHITE if log.fields.get("Colour", "").lower().startswith("w") else chess.BLACK


def _our_positions(pgn_path: Path | None, colour: chess.Color) -> list[tuple[str, chess.Move]]:
    """The FEN before each of our moves, and the move, in game order, from the PGN."""
    if pgn_path is None or not pgn_path.exists():
        return []
    game = chess.pgn.read_game(io.StringIO(pgn_path.read_text()))
    if game is None:
        return []
    board = game.board()
    positions: list[tuple[str, chess.Move]] = []
    for node in game.mainline():
        if board.turn == colour:
            positions.append((board.fen(), node.move))
        board.push(node.move)
    return positions


def _align(
    log: Log, positions: list[tuple[str, chess.Move]]
) -> list[tuple[Printed, tuple[str, chess.Move] | None]]:
    """Pair each printed line with the PGN position it was printed for.

    With nothing dropped the lines and our moves line up one to one from the start. With a
    gap in the middle, the surviving lines are the first few and the last few moves, so the
    head is aligned from the front and the tail from the back, and each pairing is checked
    against the move the line names.
    """
    total = len(log.moves) or len(positions)
    printed = log.printed
    if not positions or not total:
        return [(line, None) for line in printed]
    pairs: list[tuple[Printed, tuple[str, chess.Move] | None]] = []
    if len(printed) >= total or log.dropped_bytes == 0:
        indices = list(range(len(printed)))
    else:
        head = next(
            (i for i, line in enumerate(printed) if line.move != positions[i][1].uci()),
            len(printed),
        )
        tail = len(printed) - head
        indices = list(range(head)) + list(range(total - tail, total))
    for line, index in zip(printed, indices, strict=True):
        if index < len(positions) and positions[index][1].uci() == line.move:
            pairs.append((line, positions[index]))
        else:
            pairs.append((line, None))
    return pairs


def _measure_local_nps(agent_dir: Path, fens: list[str]) -> float | None:
    """Run the local build on a few of the game's positions and read its nps back."""
    if not (agent_dir / "agent.py").exists() or not fens:
        return None
    script = (
        "import sys, contextlib, io\n"
        f"sys.path.insert(0, {str(agent_dir)!r})\n"
        "import agent\n"
        f"for fen in {fens!r}:\n"
        "    out = io.StringIO()\n"
        "    with contextlib.redirect_stdout(out):\n"
        "        agent.get_move(fen, 30000)\n"
        "    print(out.getvalue().strip().splitlines()[-1])\n"
    )
    try:
        run = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    rates = []
    for line in run.stdout.splitlines():
        found = OUTPUT_LINE.match(line.strip())
        if found and found["nps"]:
            rates.append(int(found["nps"]))
    return statistics.median(rates) if rates else None


def _hard_budget_ms(clock_ms: float) -> float:
    return max(min(clock_ms / HARD_DIVISOR, clock_ms - SAFETY_MARGIN_MS), 0.0)


def report(log: Log, pgn_path: Path | None, agent_dir: Path | None, swing_cp: int) -> None:
    fields = log.fields
    print(
        f"Round {fields.get('Round', '?')}, {fields.get('Team', '?')} as "
        f"{fields.get('Colour', '?')} vs {fields.get('Opponent', '?')}, "
        f"{fields.get('Opening', '?')}: {fields.get('Result', '?')}"
    )

    # The gap. Nothing below is inferred across it.
    expected = len(log.moves)
    have = len(log.printed)
    if log.dropped_bytes or (expected and have < expected):
        print(
            f"\nLOG GAP: {have} of {expected} move lines survive"
            + (f", {log.dropped_bytes:,} bytes dropped" if log.dropped_bytes else "")
            + ". Per-move figures below cover only the surviving lines."
        )
    else:
        print(f"\nAll {have} move lines survive; nothing was dropped.")

    # Tracebacks: get_move fell through to its fallback and played a near-random move.
    if log.tracebacks:
        print(f"\nTRACEBACKS: {log.tracebacks}. Each one is a move played by the fallback.")
    else:
        print("\nTracebacks: none.")
    if expected and have < expected and not log.dropped_bytes:
        print(
            f"  {expected - have} moves have no printed line and no gap explains them: "
            "a fallback move prints only its traceback, so check the raw log."
        )

    colour = _our_colour(log)
    positions = _our_positions(pgn_path, colour)
    pairs = _align(log, positions)

    # Speed.
    rates = [line.nps for line in log.printed if line.nps is not None and line.ms >= 500]
    if rates:
        comp = statistics.median(rates)
        print(
            f"\nCompetition nps: median {comp:,.0f}, range {min(rates):,} to {max(rates):,} "
            f"(moves of 500 ms or more, {len(rates)} of them)"
        )
        if agent_dir is not None:
            sample = [fen for (_, pos) in pairs[:4] if pos is not None for fen in [pos[0]]]
            local = _measure_local_nps(agent_dir, sample)
            if local:
                print(
                    f"Local nps on the same opening positions: median {local:,.0f}; "
                    f"competition / local = {comp / local:.2f}"
                )
            else:
                print("Local nps: could not measure (no agent.py or no positions)")

    # Depth and its trend.
    depths = [line.depth for line in log.printed]
    if depths:
        thirds = [depths[i * len(depths) // 3 : (i + 1) * len(depths) // 3] for i in range(3)]
        means = [statistics.mean(part) if part else 0.0 for part in thirds]
        partial = sum(1 for line in log.printed if line.partial)
        print(
            f"\nDepth: min {min(depths)}, max {max(depths)}, mean {statistics.mean(depths):.1f}; "
            f"by thirds of the surviving lines {means[0]:.1f} / {means[1]:.1f} / {means[2]:.1f}; "
            f"{partial} moves came from an aborted iteration"
        )
        print("  " + " ".join(str(depth) for depth in depths))

    # Time against the hard budget, from our own timing and from the platform's.
    if log.printed:
        worst = max(log.printed, key=lambda line: line.ms)
        tightest = max(log.printed, key=lambda line: line.ms / line.hard if line.hard else 0.0)
        soft_share = statistics.mean(line.ms / line.soft for line in log.printed if line.soft)
        print(
            f"\nSlowest move (our timing): {worst.ms} ms for {worst.move} against a hard budget "
            f"of {worst.hard} ms and a soft budget of {worst.soft} ms"
        )
        print(
            f"Tightest against hard: {tightest.move} at {tightest.ms / tightest.hard:.0%} "
            f"of {tightest.hard} ms"
        )
        print(f"Average spend as a share of the soft budget: {soft_share:.0%}")
    if log.moves:
        slowest = max(log.moves, key=lambda row: row.seconds)
        before = slowest.clock_left + slowest.seconds
        for row in log.moves:
            if row.number == slowest.number - 1:
                before = row.clock_left
        print(
            f"Slowest move (platform timing): {slowest.seconds:.1f} s on move {slowest.number} "
            f"({slowest.san}) with {before:.1f} s on the clock, hard budget "
            f"{_hard_budget_ms(before * 1000.0) / 1000.0:.1f} s"
        )

    # The clock at the end: unspent time is strength not used.
    left = fields.get("Left at end")
    used = fields.get("Time used")
    base = fields.get("Base", "")
    if left and used and log.moves:
        base_s = float(base.split()[0]) if base else 120.0
        inc_s = 0.5
        if "plus" in base:
            inc_s = float(base.split("plus")[1].split()[0])
        available = base_s + inc_s * len(log.moves)
        print(
            f"\nClock: used {used} of {available:.1f} s available over {len(log.moves)} moves, "
            f"{left} left at the end ({float(left.split()[0]) / available:.0%} unspent)"
        )

    # Memory.
    if log.printed:
        peak = max(line.rss_mb for line in log.printed)
        table = max(line.table for line in log.printed)
        print(f"\nPeak RSS {peak} MB, transposition table high-water mark {table:,} entries")

    # Swings in our own score from one move to the next, with the position before the move.
    print(f"\nScore swings of {swing_cp} cp or more between consecutive moves (our view):")
    previous: Printed | None = None
    found_any = False
    for line, position in pairs:
        if (
            previous is not None
            and line.score is not None
            and previous.score is not None
            and abs(previous.score) < MATE_CP
            and abs(line.score) < MATE_CP
            and abs(line.score - previous.score) >= swing_cp
        ):
            found_any = True
            where = position[0] if position else "(no PGN position aligned)"
            print(
                f"  {previous.score:+d} -> {line.score:+d} ({line.score - previous.score:+d}) "
                f"at our move {line.move}, depth {line.depth}\n    before: {where}"
            )
        previous = line
    if not found_any:
        print("  none")
    if log.other_lines:
        print("\nOther output lines:")
        for text in log.other_lines[:20]:
            print("  " + text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a competition log and its PGN.")
    parser.add_argument("log", type=Path)
    parser.add_argument("pgn", type=Path, nargs="?")
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=LOCAL_AGENT,
        help="local build to measure nps against; pass an empty string to skip",
    )
    parser.add_argument("--swing-cp", type=int, default=SWING_CP)
    arguments = parser.parse_args()

    pgn = arguments.pgn
    if pgn is None:
        candidate = arguments.log.with_suffix(".pgn")
        pgn = candidate if candidate.exists() else None
    agent_dir = arguments.agent_dir if str(arguments.agent_dir) else None
    log = parse(arguments.log.read_text())
    with contextlib.suppress(BrokenPipeError):
        report(log, pgn, agent_dir, arguments.swing_cp)


if __name__ == "__main__":
    main()
