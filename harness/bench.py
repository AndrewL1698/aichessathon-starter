"""A gauntlet: score one candidate agent against several opponents in one run.

`harness.arena` plays one opponent. This plays the gauntlet, and reuses arena's pieces rather
than rewriting them: the pairing (every opening twice, colours swapped, so colour luck cancels),
the score-to-Elo conversion and the 95% interval. What it adds is what one match does not
report: which side failed and how, the tracebacks the candidate printed while swallowing an
exception, and how long the candidate's slowest move took, recovered from the PGN clocks.

    uv run python -m harness.bench --candidate ../cand-null-move --jobs 4

Any of illegal moves, exceptions, timeouts or over-budget moves disqualifies a candidate no
matter what its Elo says. A move is over budget when it spent more than a quarter of the clock
it had: double the engine's own hard budget, so the tens of milliseconds a clock-check slice
overshoots by are not flagged, and a genuine time-management break is.
"""

import argparse
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from harness.arena import CONFIDENCE, FAST_BASE_MS, FAST_INCREMENT_MS, Game, _elo
from harness.referee import play_match
from harness.rules import OPENINGS, PLY_CAP
from harness.sandbox import local

ROOT = Path(__file__).resolve().parent.parent
OVER_BUDGET_SHARE = 0.25
TRACEBACK = "Traceback (most recent call last)"

# The gauntlet, most games against the frozen baseline because that is the real signal.
GAUNTLET: tuple[tuple[str, Path, int], ...] = (
    ("baseline", ROOT / "local-opponents" / "baseline", 32),
    ("sunfish", ROOT / "local-opponents" / "sunfish", 16),
    ("minimax", ROOT / "baselines" / "minimax", 16),
)

# Mainstream lines beyond the harness's eight, because two deterministic engines replay the
# same game from the same opening, and eight openings is sixteen distinct games at most.
# Each is checked to be legal when this module loads.
BENCH_LINES: tuple[tuple[str, str], ...] = (
    ("Ruy Lopez Closed", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6"),
    ("Giuoco Piano", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O"),
    ("Sicilian Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6"),
    ("Sicilian Dragon", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6"),
    ("French Advance", "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Bd7 Be2 Nge7"),
    ("Caro-Kann Classical", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7"),
    ("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7 Rc1 c6"),
    ("Slav", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4"),
    ("Nimzo-Indian", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6"),
    ("King's Indian Classical", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6"),
    ("Queen's Indian", "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O Nc3 Ne4"),
    ("Catalan", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6"),
    ("English Symmetrical", "c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 Nf6 O-O O-O d4 cxd4 Nxd4"),
    ("Scandinavian", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6"),
    ("Pirc", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6"),
    ("London System", "d4 d5 Nf3 Nf6 Bf4 e6 e3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O"),
    ("Alekhine", "e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6 O-O Be7 c4 Nb6"),
    ("Modern Benoni", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Nf3 Bg7"),
    ("Dutch Leningrad", "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6 Nc3 Qe8"),
    ("Vienna", "e4 e5 Nc3 Nf6 Bc4 Nxe4 Qh5 Nd6 Bb3 Be7 Nf3 Nc6 Nxe5 g6"),
)


def _fen(line: str) -> str:
    board = chess.Board()
    for san in line.split():
        board.push_san(san)
    return board.fen()


BENCH_OPENINGS: tuple[tuple[str, str], ...] = OPENINGS + tuple(
    (name, _fen(line)) for name, line in BENCH_LINES
)


@dataclass(frozen=True)
class Played:
    """One game, seen from the candidate's side."""

    opponent: str
    game: Game
    illegal: int
    exceptions: int
    timeouts: int
    over_budget: int
    worst_ms: float
    worst_clock_ms: float
    seconds: float

    @property
    def points(self) -> float | None:
        result = self.game.outcome.result
        if result == "void":
            return None
        if result == "draw":
            return 0.5
        return 1.0 if self.game.won else 0.0


@dataclass(frozen=True)
class Tally:
    """Wins, draws and losses turned into a score, an interval and an Elo, as arena does it."""

    wins: int
    draws: int
    losses: int

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + self.draws / 2) / self.played if self.played else 0.5

    @property
    def margin(self) -> float:
        played, score = self.played, self.score
        if played < 2:
            return 0.5
        spread = (
            self.wins * (1 - score) ** 2 + self.draws * (0.5 - score) ** 2 + self.losses * score**2
        )
        return CONFIDENCE * math.sqrt(spread / (played - 1) / played)

    def text(self) -> str:
        low, high = self.score - self.margin, self.score + self.margin
        return (
            f"+{self.wins} ={self.draws} -{self.losses} ({self.played}), "
            f"score {self.score:.1%} +- {self.margin:.1%}, "
            f"Elo {_elo_text(self.score)} [{_elo_text(low)}, {_elo_text(high)}]"
        )


def _elo_text(score: float) -> str:
    if score <= 0.0:
        return "-inf"
    if score >= 1.0:
        return "+inf"
    return f"{_elo(score):+.0f}"


def _tally(played: list[Played]) -> Tally:
    points = [game.points for game in played]
    wins = sum(1 for point in points if point == 1.0)
    draws = sum(1 for point in points if point == 0.5)
    losses = sum(1 for point in points if point == 0.0)
    return Tally(wins, draws, losses)


def _assess(
    opponent: str, game: Game, log: str, base_ms: int, increment_ms: int, seconds: float
) -> Played:
    """Attribute a failure to a side and recover the candidate's move times from the clocks."""
    colour = chess.WHITE if game.plays_white else chess.BLACK
    pgn = chess.pgn.read_game(io.StringIO(game.outcome.pgn))
    if pgn is None:
        raise RuntimeError("The referee wrote a PGN python-chess cannot read")
    termination = game.outcome.termination

    # The referee stops before pushing a move that failed, so the side to move at the end is
    # the side that failed. A failed init has no moves: the loser is the side that failed.
    ours = False
    if termination in ("illegal", "flag", "crash"):
        ours = pgn.end().board().turn == colour
    elif termination == "init":
        ours = game.outcome.result != ("white" if game.plays_white else "black")
    elif termination == "both_failed":
        ours = True
    illegal = int(ours and termination == "illegal")
    timeouts = int(ours and termination == "flag")
    crashes = int(ours and termination in ("crash", "init", "both_failed"))
    # The agent catches exceptions inside get_move and prints the traceback, so a search
    # that blew up shows in the log and nowhere else. The log keeps its first and last 4 KB.
    exceptions = crashes + log.count(TRACEBACK)

    remaining = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    worst_ms, worst_clock_ms, over_budget = 0.0, 0.0, 0
    board = pgn.board()
    for node in pgn.mainline():
        mover = board.turn
        clock = node.clock()
        if clock is None:
            raise RuntimeError("The referee wrote a move without a clock")
        after = clock * 1000.0
        before = remaining[mover]
        spent = before - (after - increment_ms)
        remaining[mover] = after
        if mover == colour:
            if spent > worst_ms:
                worst_ms, worst_clock_ms = spent, before
            if spent > before * OVER_BUDGET_SHARE:
                over_budget += 1
        board.push(node.move)

    return Played(
        opponent,
        game,
        illegal,
        exceptions,
        timeouts,
        over_budget,
        worst_ms,
        worst_clock_ms,
        seconds,
    )


def _play(
    opponent: str,
    opponent_dir: Path,
    index: int,
    candidate: Path,
    arguments: argparse.Namespace,
) -> Played:
    opening, fen = BENCH_OPENINGS[(index // 2) % len(BENCH_OPENINGS)]
    plays_white = index % 2 == 0
    ours, theirs = local(candidate, index), local(opponent_dir, index)
    white, black = (ours, theirs) if plays_white else (theirs, ours)
    started = time.monotonic()
    outcome = play_match(
        white,
        black,
        arguments.base_ms,
        arguments.increment_ms,
        ply_cap=arguments.ply_cap,
        start_fen=fen,
    )
    seconds = time.monotonic() - started
    game = Game(index, opening, plays_white, outcome)
    return _assess(
        opponent, game, ours.stderr_log, arguments.base_ms, arguments.increment_ms, seconds
    )


def _disqualifiers(played: list[Played]) -> dict[str, int]:
    return {
        "illegal": sum(game.illegal for game in played),
        "exceptions": sum(game.exceptions for game in played),
        "timeouts": sum(game.timeouts for game in played),
        "over_budget": sum(game.over_budget for game in played),
    }


def _worst(played: list[Played]) -> Played | None:
    return max(played, key=lambda game: game.worst_ms, default=None)


def _summary(name: str, played: list[Played]) -> str:
    tally = _tally(played)
    counts = _disqualifiers(played)
    worst = _worst(played)
    worst_text = (
        f"{worst.worst_ms / 1000.0:.2f}s at clock {worst.worst_clock_ms / 1000.0:.1f}s"
        if worst is not None
        else "n/a"
    )
    counts_text = ", ".join(f"{key} {value}" for key, value in counts.items())
    verdict = "  DISQUALIFIED" if any(counts.values()) else ""
    return f"{name:9} {tally.text()}\n{'':9} {counts_text}, worst move {worst_text}{verdict}"


def _markdown(label: str, name: str, played: list[Played], arguments: argparse.Namespace) -> str:
    tally = _tally(played)
    counts = _disqualifiers(played)
    worst = _worst(played)
    worst_text = f"{worst.worst_ms / 1000.0:.2f}s" if worst is not None else "n/a"
    control = f"{arguments.base_ms / 1000.0:g}s+{arguments.increment_ms / 1000.0:g}s"
    low, high = tally.score - tally.margin, tally.score + tally.margin
    return (
        f"| {label} | {name} | {control} | {tally.played} | "
        f"+{tally.wins} ={tally.draws} -{tally.losses} | {tally.score:.1%} | "
        f"{_elo_text(tally.score)} | {_elo_text(low)} to {_elo_text(high)} | "
        f"{counts['illegal']} / {counts['exceptions']} / {counts['timeouts']} / "
        f"{counts['over_budget']} | {worst_text} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a candidate against the gauntlet.")
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--label", default=None, help="a name for the run in the output")
    parser.add_argument("--baseline-dir", type=Path, default=GAUNTLET[0][1])
    parser.add_argument("--baseline-games", type=int, default=GAUNTLET[0][2])
    parser.add_argument("--sunfish-games", type=int, default=GAUNTLET[1][2])
    parser.add_argument("--minimax-games", type=int, default=GAUNTLET[2][2])
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--jobs", type=int, default=1, help="games played at once")
    parser.add_argument("--pgn-dir", type=Path)
    arguments = parser.parse_args()

    candidate = arguments.candidate.resolve()
    label = arguments.label or candidate.name
    if arguments.pgn_dir:
        arguments.pgn_dir.mkdir(parents=True, exist_ok=True)

    schedule: list[tuple[str, Path, int]] = []
    for (name, directory, _), games in zip(
        GAUNTLET,
        (arguments.baseline_games, arguments.sunfish_games, arguments.minimax_games),
        strict=True,
    ):
        if name == "baseline":
            directory = arguments.baseline_dir.resolve()
        games = games - games % 2
        schedule.extend((name, directory, index) for index in range(games))

    total = len(schedule)
    print(
        f"{label}: {total} games, {arguments.base_ms} ms + {arguments.increment_ms} ms, "
        f"{arguments.jobs} at a time"
    )
    started = time.monotonic()
    played: dict[str, list[Played]] = {name: [] for name, _, _ in GAUNTLET}
    with ThreadPoolExecutor(max_workers=max(1, arguments.jobs)) as pool:
        futures = {
            pool.submit(_play, name, directory, index, candidate, arguments): (name, index)
            for name, directory, index in schedule
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            played[result.opponent].append(result)
            game = result.game
            print(
                f"[{done}/{total}] {result.opponent} {game.index + 1}, {game.opening} as "
                f"{game.colour}: {game.outcome.result} by {game.outcome.termination}, "
                f"worst {result.worst_ms / 1000.0:.2f}s, {result.seconds:.0f}s"
                + (
                    "  <-- FAILURE"
                    if result.illegal or result.exceptions or result.timeouts or result.over_budget
                    else ""
                ),
                flush=True,
            )
            if arguments.pgn_dir:
                destination = arguments.pgn_dir / f"{result.opponent}-{game.index + 1:03d}.pgn"
                destination.write_text(game.outcome.pgn + "\n")

    elapsed = time.monotonic() - started
    everything = [game for games in played.values() for game in games]
    print(f"\n{label} vs the gauntlet, {elapsed / 60.0:.1f} minutes")
    for name, _, _ in GAUNTLET:
        if played[name]:
            print(_summary(name, sorted(played[name], key=lambda game: game.game.index)))
    print(_summary("overall", everything))

    print(
        "\n| run | opponent | control | games | +=- | score | Elo | 95% "
        "| ill/exc/tmo/over | worst |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, _, _ in GAUNTLET:
        if played[name]:
            print(_markdown(label, name, played[name], arguments))
    print(_markdown(label, "overall", everything, arguments))

    if any(_disqualifiers(everything).values()):
        raise SystemExit(f"{label} is disqualified: see the counts above")


if __name__ == "__main__":
    main()
