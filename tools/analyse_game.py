"""Run a local Stockfish over a competition PGN and list where our evaluation dropped.

    uv run python tools/analyse_game.py logs/73-castling.pgn [--team "2 Pawns and a Queen"]

OFFLINE ANALYSIS ONLY. Stockfish is never imported, invoked, referenced or shipped by
agent.py or anything in submission.zip. This file lives in tools/, which harness/package.py
cannot reach: it zips root-level modules and the directories they import by name, and no
root module imports `tools`.

For every move of ours it reports the drop in Stockfish's evaluation from before the move to
after it, the position before, the move we played and the move Stockfish preferred. A drop
of `--blunder` centipawns or more is a blunder. A blunder is COSMETIC when the position after
it is still decisively won for us (at least `--decisive` centipawns); it cost nothing that
matters. Everything else is REAL. Real blunders are the ones that go into tests/positions.
"""

import argparse
import io
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ENGINE = "stockfish"
TEAM = "2 Pawns and a Queen"
BLUNDER_CP = 150
DECISIVE_CP = 500
# Mate scores are clamped to this so a drop from mate to a big advantage is still a number.
MATE_CP = 10_000
# Phases, so a blunder can be placed. Non-pawn material of both sides, in centipawns.
ENDGAME_MATERIAL = 2_600
OPENING_MOVES = 12
NON_PAWN = {chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}


@dataclass(frozen=True)
class Assessed:
    number: int  # full move number
    fen_before: str
    played: chess.Move
    played_san: str
    best: chess.Move | None
    best_san: str
    before_cp: int  # our view, best play
    after_cp: int  # our view, after what we played
    phase: str

    @property
    def drop(self) -> int:
        return self.before_cp - self.after_cp


def _cp(score: chess.engine.PovScore, colour: chess.Color) -> int:
    return score.pov(colour).score(mate_score=MATE_CP) or 0


def _phase(board: chess.Board, moves_played: int) -> str:
    material = sum(
        value * (len(board.pieces(piece, chess.WHITE)) + len(board.pieces(piece, chess.BLACK)))
        for piece, value in NON_PAWN.items()
    )
    if material <= ENDGAME_MATERIAL:
        return "endgame"
    if moves_played < OPENING_MOVES:
        return "opening"
    return "middlegame"


def analyse(
    game: chess.pgn.Game,
    colour: chess.Color,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
) -> list[Assessed]:
    board = game.board()
    # Evaluate every position once, from our side; a move's drop is the difference between
    # the position before it and the position after it.
    assessed: list[Assessed] = []
    info = engine.analyse(board, limit)
    current_cp = _cp(info["score"], colour)
    current_best = info.get("pv", [None])[0]
    our_moves = 0
    for node in game.mainline():
        move = node.move
        fen_before = board.fen()
        number = board.fullmove_number
        ours = board.turn == colour
        phase = _phase(board, our_moves)
        played_san = board.san(move)
        best_san = board.san(current_best) if current_best else "?"
        board.push(move)
        if board.is_game_over():
            after_cp = current_cp if not ours else MATE_CP if board.is_checkmate() else 0
            after_best = None
        else:
            info = engine.analyse(board, limit)
            after_cp = _cp(info["score"], colour)
            after_best = info.get("pv", [None])[0]
        if ours:
            assessed.append(
                Assessed(
                    number,
                    fen_before,
                    move,
                    played_san,
                    current_best,
                    best_san,
                    current_cp,
                    after_cp,
                    phase,
                )
            )
            our_moves += 1
        current_cp, current_best = after_cp, after_best
    return assessed


def report(assessed: list[Assessed], blunder_cp: int, decisive_cp: int) -> None:
    drops = [max(0, a.drop) for a in assessed]
    acpl = statistics.mean(min(d, 1000) for d in drops) if drops else 0.0
    # A drop where we played the engine's own move is the engine seeing further from the
    # position after than from the position before, not a mistake of ours.
    blunders = [a for a in assessed if a.drop >= blunder_cp and a.played != a.best]
    real = [a for a in blunders if a.after_cp < decisive_cp]
    cosmetic = [a for a in blunders if a.after_cp >= decisive_cp]
    print(
        f"{len(assessed)} of our moves, ACPL {acpl:.0f} (drops capped at 1000), "
        f"{len(blunders)} blunders of {blunder_cp}+ cp: {len(real)} real, {len(cosmetic)} cosmetic"
    )
    for label, group in (("REAL", real), ("COSMETIC", cosmetic)):
        print(f"\n{label} blunders ({len(group)}):")
        if not group:
            print("  none")
        for a in group:
            print(
                f"  move {a.number} {a.played_san} ({a.played.uci()}), {a.phase}: "
                f"{a.before_cp:+d} -> {a.after_cp:+d}, drop {a.drop}; Stockfish preferred "
                f"{a.best_san} ({a.best.uci() if a.best else '?'})\n    before: {a.fen_before}"
            )
    by_phase: dict[str, list[int]] = {"opening": [], "middlegame": [], "endgame": []}
    for a in real:
        by_phase[a.phase].append(a.number)
    print("\nReal blunders by phase: " + ", ".join(f"{k} {len(v)}" for k, v in by_phase.items()))
    print("\nEPD lines for tests/positions (real blunders only):")
    for a in real:
        if a.best is not None:
            print(f'{a.fen_before}; bm {a.best.uci()}; id "round move {a.number}"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Stockfish review of one of our PGNs.")
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--team", default=TEAM, help="our name in the PGN headers")
    parser.add_argument("--engine", default=ENGINE, help="path to a UCI engine binary")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--blunder", type=int, default=BLUNDER_CP)
    parser.add_argument("--decisive", type=int, default=DECISIVE_CP)
    arguments = parser.parse_args()

    binary = shutil.which(arguments.engine) or arguments.engine
    if not Path(binary).exists():
        raise SystemExit(f"No engine at {arguments.engine}. This is a local analysis tool only.")
    game = chess.pgn.read_game(io.StringIO(arguments.pgn.read_text()))
    if game is None:
        raise SystemExit(f"No game in {arguments.pgn}")
    white, black = game.headers.get("White", ""), game.headers.get("Black", "")
    if arguments.team.lower() == white.lower():
        colour = chess.WHITE
    elif arguments.team.lower() == black.lower():
        colour = chess.BLACK
    else:
        raise SystemExit(f"{arguments.team!r} is neither {white!r} nor {black!r}; pass --team")
    print(
        f"{arguments.pgn.name}: {white} vs {black}, we are "
        f"{'white' if colour == chess.WHITE else 'black'}, result {game.headers.get('Result')}, "
        f"engine {binary} depth {arguments.depth}"
    )
    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        engine.configure({"Threads": 1, "Hash": 256})
        assessed = analyse(game, colour, engine, chess.engine.Limit(depth=arguments.depth))
    report(assessed, arguments.blunder, arguments.decisive)


if __name__ == "__main__":
    main()
