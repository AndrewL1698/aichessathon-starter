"""Run the engine over the regression positions and say which it solves, and at what depth.

    uv run python -m tests.positions.run [--agent-dir ../cand-x] [--max-depth 7] [--seconds 20]

Each line of `positions.epd` is a FEN, then `bm <uci>` for the move we should have found, then
`id "..."`. They are the real blunders from rated games, as judged by tools/analyse_game.py.
A measurement tool: never shipped, and the engine never reads it. It reaches into the agent's
search to run fixed depths one at a time, so it reports the first depth at which the wanted
move becomes the engine's choice, which is what a tactical suite is for. From v3.0 the agent
has a compiled engine (`_COMPILED`, `fastsearch.root`) and that is what is measured; an older
agent directory is run through its Python search (`_root`, `_Search`, `_MEMORY`). It measures
sharpness, not strength: a change that solves more here can still lose games, so decide on the
bench's Elo and read this alongside it.
"""

import argparse
import contextlib
import importlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import chess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
POSITIONS = HERE / "positions.epd"


@dataclass(frozen=True)
class Position:
    fen: str
    best: str
    name: str


def load(path: Path) -> list[Position]:
    positions = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fen, *ops = [part.strip() for part in line.split(";")]
        best, name = "", ""
        for op in ops:
            if op.startswith("bm "):
                best = op[3:].strip()
            elif op.startswith("id "):
                name = op[3:].strip().strip('"')
        if not best:
            raise SystemExit(f"No bm in: {line}")
        positions.append(Position(fen, best, name or fen))
    return positions


def _load_agent(agent_dir: Path) -> ModuleType:
    sys.path.insert(0, str(agent_dir))
    for name in ("agent", "fastboard", "fastsearch"):
        sys.modules.pop(name, None)
    with contextlib.redirect_stdout(io.StringIO()):
        return importlib.import_module("agent")


def solve(agent: ModuleType, position: Position, max_depth: int, seconds: float) -> tuple[int, int]:
    """Return (first depth at which the engine picks the wanted move, deepest depth finished).

    The first is 0 when it never did. Depths are searched one at a time with a fresh table so
    each depth's answer is its own, not a leftover from a longer search of the same position.
    """
    if getattr(agent, "_COMPILED", None) is not None:
        return _solve_compiled(agent, position, max_depth, seconds)
    board = chess.Board(position.fen)
    agent._MEMORY.table.clear()
    agent._MEMORY.seen.clear()
    agent._MEMORY.history = [0] * len(agent._MEMORY.history)
    deadline = time.perf_counter() + seconds
    search = agent._Search(deadline=deadline)
    moves = list(board.legal_moves)
    agent._order(board, moves)
    first = moves[0]
    solved_at, reached = 0, 0
    for depth in range(1, max_depth + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                first, _ = agent._root(board, depth, first, search)
        except agent._Timeout:
            break
        reached = depth
        if first.uci() == position.best:
            if not solved_at:
                solved_at = depth
        else:
            solved_at = 0  # it has to hold at the deepest depth finished, not merely appear once
    return solved_at, reached


def _solve_compiled(
    agent: ModuleType, position: Position, max_depth: int, seconds: float
) -> tuple[int, int]:
    """`solve` for an agent with the compiled engine: the same fixed depths through `fastsearch`.

    The compiled search stops on a node count, so the time limit is enforced between root
    moves at a node rate measured as it goes, the way the agent itself does it.
    """
    fs = sys.modules["fastsearch"]
    fb = sys.modules["fastboard"]
    state = agent._COMPILED.state
    board = chess.Board(position.fen)
    contempt = agent._contempt(agent.evaluate(board))
    state.set_position(position.fen)
    state.new_game()
    state.remember(state.key())
    state.begin_move(contempt)
    started = time.perf_counter()

    def budget() -> int:
        elapsed = time.perf_counter() - started
        remaining = seconds - elapsed
        if remaining <= 0.0:
            return 0
        nodes = int(state.nodes)
        rate = nodes / elapsed if nodes and elapsed > 0.005 else agent._COMPILED.rate * 1000.0
        return nodes + int(remaining * rate)

    legal = state.legal_moves()
    first = max(legal, key=lambda move: int(fs.move_score(state.board, move, state.w)))
    solved_at, reached = 0, 0
    for depth in range(1, max_depth + 1):
        try:
            first, _ = fs.root(state, depth, first, budget)
        except fs.Aborted:
            break
        reached = depth
        if fb.move_to_uci(first) == position.best:
            if not solved_at:
                solved_at = depth
        else:
            solved_at = 0
    return solved_at, reached


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression positions from real blunders.")
    parser.add_argument("--agent-dir", type=Path, default=ROOT)
    parser.add_argument("--positions", type=Path, default=POSITIONS)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=20.0, help="per position")
    arguments = parser.parse_args()

    positions = load(arguments.positions)
    agent = _load_agent(arguments.agent_dir.resolve())
    solved = 0
    print(
        f"{arguments.agent_dir}: {len(positions)} positions, up to depth {arguments.max_depth}, "
        f"{arguments.seconds:g} s each"
    )
    for position in positions:
        started = time.perf_counter()
        at, reached = solve(agent, position, arguments.max_depth, arguments.seconds)
        spent = time.perf_counter() - started
        if at:
            solved += 1
            print(
                f"  solved  d{at} (reached d{reached}, {spent:4.1f}s)  "
                f"{position.name}: {position.best}"
            )
        else:
            print(
                f"  missed     (reached d{reached}, {spent:4.1f}s)  "
                f"{position.name}: wanted {position.best}"
            )
    print(f"solved {solved}/{len(positions)}")


if __name__ == "__main__":
    main()
