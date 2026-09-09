"""Train the learned evaluation. Offline only; torch never ships and never runs on the clock.

Architecture: 768 -> H (default 128) -> 32 -> 1, clipped ReLU on both hidden layers.

The clipped ReLU is not a stylistic choice. Runtime inference is int16 in numba, where a hidden
activation lives on a fixed scale and saturates; training with ``clamp(x, 0, 1)`` makes the float
model and the quantised model agree about what saturation does, so export is close to lossless.

The target is ``sigmoid(cp / scale)``: centipawns are not linear in winning chances, and an MSE
straight on centipawns spends all its capacity on lopsided positions nobody has to evaluate well.
The network's raw output is the pre-sigmoid value, so the runtime reads centipawns back as
``raw * scale`` with no sigmoid at all.

Run from the repo root:

    uv run python -m tools.nnue.train --data tools/nnue/data/lichess --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from tools.nnue.features import NUM_FEATURES, features

# Centipawns per unit of network output. 400 is the usual NNUE choice: sigmoid(400/400) = 0.73,
# so a one-pawn edge is a bit under three quarters of a point.
DEFAULT_CP_SCALE = 400.0

# Positions printed after every epoch. Cheap, and a net that gets these wrong is broken in a way
# no loss number makes obvious.
SANITY_POSITIONS: tuple[tuple[str, str], ...] = (
    ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("+1 pawn (white)", "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("+1 rook (white)", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"),
    ("mate in 1 (white)", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"),
)


class Nnue(nn.Module):
    """768 -> hidden -> 32 -> 1 with clipped ReLU. Input is a dense 0/1 feature vector."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(NUM_FEATURES, hidden)
        self.l2 = nn.Linear(hidden, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        first = torch.clamp(self.l1(dense), 0.0, 1.0)
        second = torch.clamp(self.l2(first), 0.0, 1.0)
        return self.l3(second).squeeze(-1)


def densify(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Turn a padded ``[B, 32]`` int index batch into a dense ``[B, 768]`` float batch.

    Padding is -1, so everything is shifted up by one into a 769-wide scratch tensor whose first
    column is then dropped. At 16384 x 769 floats a batch is 50 MB, which is nothing here, so the
    sparse-accumulator trick the runtime needs is not worth its complexity offline.
    """
    rows = indices.shape[0]
    scratch = torch.zeros(rows, NUM_FEATURES + 1, device=device)
    scratch.scatter_(1, (indices.to(device).long() + 1), 1.0)
    return scratch[:, 1:]


def load_shards(
    data_dir: Path, with_hand: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load every shard under ``data_dir`` into RAM: index matrix, cp vector, and, when asked,
    the ``hand`` vector that ``tools.nnue.hand`` adds (missing it is an error, not a zero)."""
    paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no .npz shards under {data_dir}")
    index_blocks: list[np.ndarray] = []
    cp_blocks: list[np.ndarray] = []
    hand_blocks: list[np.ndarray] = []
    for path in paths:
        with np.load(path) as shard:
            index_blocks.append(shard["indices"])
            cp_blocks.append(shard["cp"])
            if with_hand:
                if "hand" not in shard:
                    raise SystemExit(f"{path} has no hand array; run tools.nnue.hand first")
                hand_blocks.append(shard["hand"])
    indices = np.concatenate(index_blocks)
    cp = np.concatenate(cp_blocks)
    hand = np.concatenate(hand_blocks) if with_hand else None
    megabytes = (indices.nbytes + cp.nbytes + (hand.nbytes if hand is not None else 0)) / 1e6
    print(f"loaded {len(paths)} shard(s), {indices.shape[0]:,} positions, {megabytes:.0f} MB")
    return indices, cp, hand


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sanity_table(model: Nnue, device: torch.device, cp_scale: float) -> str:
    """Return one line per sanity position with the model's raw centipawn prediction."""
    boards = [chess.Board(fen) for _, fen in SANITY_POSITIONS]
    padded = np.full((len(boards), 32), -1, dtype=np.int16)
    for row, board in enumerate(boards):
        active = features(board)
        padded[row, : active.size] = active
    model.eval()
    with torch.no_grad():
        raw = model(densify(torch.from_numpy(padded), device)).cpu().numpy()
    model.train()
    parts = [f"{name} {raw[row] * cp_scale:+7.0f}cp" for row, (name, _) in
             enumerate(SANITY_POSITIONS)]
    return "  |  ".join(parts)


def run_epoch(
    model: Nnue,
    optimiser: torch.optim.Optimizer | None,
    indices: np.ndarray,
    target: np.ndarray,
    order: np.ndarray,
    batch_size: int,
    device: torch.device,
    l1_clip: float | None = None,
    offset: np.ndarray | None = None,
) -> float:
    """Run one pass. ``optimiser`` None means evaluation. Returns mean MSE in WDL space.

    ``offset`` is the hand evaluation in units of ``cp_scale``, one per row, for a residual
    net: the sigmoid is taken of ``hand + net`` so the net learns only the correction.

    ``l1_clip`` clamps the layer-1 weights and biases to ``[-l1_clip, l1_clip]`` after every
    step. export.py proves the int16 accumulator safe from the 32 largest weights per neuron
    plus the bias, so a clip of c bounds it by 33 * c * qa: c <= 1.9 always exports at qa=512
    and c <= 3.8 at qa=256. Unconstrained nets pass 2.4 within 30 epochs and keep growing.
    """
    total = 0.0
    seen = 0
    training = optimiser is not None
    for start in range(0, order.size, batch_size):
        rows = order[start : start + batch_size]
        batch = torch.from_numpy(indices[rows])
        wanted = torch.from_numpy(target[rows]).to(device)
        with torch.set_grad_enabled(training):
            raw = model(densify(batch, device))
            if offset is not None:
                raw = raw + torch.from_numpy(offset[rows]).to(device)
            predicted = torch.sigmoid(raw)
            loss = torch.nn.functional.mse_loss(predicted, wanted)
        if training and optimiser is not None:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if l1_clip is not None:
                with torch.no_grad():
                    model.l1.weight.clamp_(-l1_clip, l1_clip)
                    model.l1.bias.clamp_(-l1_clip, l1_clip)
        total += float(loss.item()) * rows.size
        seen += rows.size
    return total / max(seen, 1)


def train(arguments: argparse.Namespace) -> None:
    device = pick_device(arguments.device)
    print(f"device: {device}")
    residual = arguments.target == "residual"
    indices, cp, hand = load_shards(Path(arguments.data), with_hand=residual)
    target = torch.sigmoid(torch.from_numpy(cp.astype(np.float32)) / arguments.cp_scale).numpy()
    offset = hand.astype(np.float32) / arguments.cp_scale if hand is not None else None

    generator = np.random.default_rng(arguments.seed)
    order = generator.permutation(indices.shape[0])
    split = max(1, int(indices.shape[0] * arguments.val_fraction))
    validation, training_rows = order[:split], order[split:]
    print(f"train {training_rows.size:,} positions, validate {validation.size:,}")
    mean_loss = float(np.mean((target[validation] - target[validation].mean()) ** 2))
    print(f"baseline val loss, mean predictor: {mean_loss:.6f}")
    if offset is not None:
        hand_only = torch.sigmoid(torch.from_numpy(offset[validation])).numpy()
        hand_loss = float(np.mean((hand_only - target[validation]) ** 2))
        print(f"baseline val loss, hand evaluation alone (net = 0): {hand_loss:.6f}")

    model = Nnue(arguments.hidden).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=arguments.lr)
    checkpoints = Path(arguments.checkpoints)
    checkpoints.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []

    for epoch in range(1, arguments.epochs + 1):
        started = time.monotonic()
        generator.shuffle(training_rows)
        train_loss = run_epoch(
            model,
            optimiser,
            indices,
            target,
            training_rows,
            arguments.batch_size,
            device,
            arguments.l1_clip,
            offset,
        )
        val_loss = run_epoch(
            model, None, indices, target, validation, arguments.batch_size, device, None, offset
        )
        seconds = time.monotonic() - started
        print(
            f"epoch {epoch:3d}  train {train_loss:.6f}  val {val_loss:.6f}  {seconds:.1f}s\n"
            f"          {sanity_table(model, device, arguments.cp_scale)}",
            flush=True,
        )
        history.append(
            {"epoch": epoch, "train": train_loss, "val": val_loss, "seconds": seconds}
        )
        torch.save(
            {
                "model": model.state_dict(),
                "hidden": arguments.hidden,
                "cp_scale": arguments.cp_scale,
                "target": arguments.target,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            checkpoints / f"epoch_{epoch:03d}.pt",
        )
        (checkpoints / "history.json").write_text(json.dumps(history, indent=2))

    best = min(history, key=lambda row: row["val"])
    print(f"best val {best['val']:.6f} at epoch {int(best['epoch'])}")
    print(f"checkpoints in {checkpoints}, loss curve in {checkpoints / 'history.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the NNUE evaluation.")
    parser.add_argument("--data", default="tools/nnue/data/lichess", help="shard directory")
    parser.add_argument("--checkpoints", default="tools/nnue/checkpoints")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--l1-clip",
        type=float,
        default=None,
        help="clamp layer-1 weights and biases to +-this after every step; 1.9 keeps qa=512",
    )
    parser.add_argument("--cp-scale", type=float, default=DEFAULT_CP_SCALE)
    parser.add_argument(
        "--target",
        choices=("cp", "residual"),
        default="cp",
        help="residual: fit hand + net to cp, needs the hand array from tools.nnue.hand",
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    train(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
