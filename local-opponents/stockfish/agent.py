"""Stockfish at a fixed depth as a local benchmarking opponent, in the platform's agent contract.

LOCAL ONLY. This directory is hyphenated so no root module can import it, which keeps it out of
`submission.zip` the same way `local-opponents/sunfish` is kept out. It needs a `stockfish`
binary on PATH (`brew install stockfish`) or `$STOCKFISH` pointing at one.

Fixed depth rather than a clock, so the strength is the same on every machine and every run:
`STOCKFISH_DEPTH` (default 10) is the one knob. One engine process per game, opened on the
first move and kept for the rest, single thread.
"""

import atexit
import os
import shutil

import chess
import chess.engine

DEPTH = int(os.environ.get("STOCKFISH_DEPTH", "10"))

_engine: chess.engine.SimpleEngine | None = None


def _open() -> chess.engine.SimpleEngine:
    global _engine
    if _engine is None:
        path = os.environ.get("STOCKFISH") or shutil.which("stockfish")
        if not path:
            raise RuntimeError("no stockfish binary: brew install stockfish or set $STOCKFISH")
        _engine = chess.engine.SimpleEngine.popen_uci(path)
        _engine.configure({"Threads": 1, "Hash": 64})
        atexit.register(_engine.quit)
    return _engine


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    result = _open().play(board, chess.engine.Limit(depth=DEPTH))
    if result.move is None:
        raise RuntimeError("stockfish returned no move")
    return result.move.uci()
