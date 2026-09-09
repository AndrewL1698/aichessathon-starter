"""SMOKE ONLY. A stand-in labeller that reuses the repo's own hand-written evaluation.

This exists so the whole pipeline -- generate, shard, train, export, verify -- can be run end
to end on a machine with no Stockfish installed. A net trained on these labels learns to imitate
`agent.evaluate`, which is a static material-and-piece-square score, so it can only ever be as
good as the thing it copies. It is not a substitute for Path A or Path B.

`tools/` importing a root module is fine; the forbidden direction is a root module importing
`tools/`, and nothing does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import evaluate as _repo_evaluate  # noqa: E402


def smoke_evaluate(board: chess.Board) -> int:
    """Side-to-move-relative centipawns from the repo's hand-written evaluation."""
    return _repo_evaluate(board)
