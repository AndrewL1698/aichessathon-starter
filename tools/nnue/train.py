"""Train the learned evaluation. Offline only; torch never ships and never runs on the clock.

Architecture: 3072 -> H (default 128) -> 32 -> 1, clipped ReLU on both hidden layers.
The 3,072 inputs are four king-bucket blocks of 768; `docs/NNUE_KING_BUCKETS.md` is the
spec. Warm start from a trained 768-input checkpoint with `tools/nnue/bucketize.py`.

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
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from torch import nn

from tools.nnue.features import NUM_FEATURES, features
from tools.nnue.nnue_ref import FLAT_SCHEME_VERSION, SCHEME_VERSION

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
    """3072 -> hidden -> 32 -> 1 with clipped ReLU. Input is a dense 0/1 feature vector.

    The 3,072 inputs are four king-bucket blocks of the old 768; only one block is ever active
    in a row, so the dense scratch `densify` builds is four times as wide and just as sparse.
    `tools/nnue/bucketize.py` warm starts this from a trained 768-input checkpoint.
    """

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
    """Turn a padded ``[B, 32]`` int index batch into a dense ``[B, 3072]`` float batch.

    Padding is -1, so everything is shifted up by one into a 3073-wide scratch tensor whose first
    column is then dropped. King buckets made this four times as wide for the same 32 active
    features: at 16384 x 3073 floats a batch is 201 MB against the 50 MB it was. Still one
    allocation per batch and still not worth the sparse-accumulator trick the runtime needs, but
    it is now the largest thing in the training loop, so a machine short of memory should drop
    the batch size before it drops anything else.
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
            # A shard built before king buckets holds 768-input indices. Training a
            # 3,072-input model on those gives a net whose bucket blocks 1 to 3 never saw a
            # position, and every check downstream of here would still pass, so refuse it now.
            scheme = int(shard["scheme"]) if "scheme" in shard else FLAT_SCHEME_VERSION
            if scheme != SCHEME_VERSION:
                raise SystemExit(
                    f"{path} holds scheme {scheme} indices and this trains scheme "
                    f"{SCHEME_VERSION} ({NUM_FEATURES} inputs); rebuild the shards with "
                    f"tools.nnue.data"
                )
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


def checkpoint_digest(path: Path) -> str:
    """The sha256 of the checkpoint file, so a run says which bytes it started from."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_init_checkpoint(path: Path, model: Nnue, arguments: argparse.Namespace) -> dict[str, Any]:
    """Load `path` into `model` as a warm start, or refuse. Returns provenance for the run.

    Everything is checked *before* the state dict is loaded, and every failure is a
    `SystemExit`. There is deliberately no fallback: a run asked to warm start and quietly
    starting from random weights instead would produce a checkpoint that looks like a
    fine-tune, trains like one, and is not one, and nothing downstream could tell. The whole
    point of the king-bucket experiment is that the net begins as the 768 net and moves from
    there, so "it did not load" has to be a stopped run, not a line of output nobody reads.

    What is checked, and why each one matters:

      hidden        a width mismatch is a different net; `Nnue(hidden)` would refuse the load
                    anyway, but with a torch error rather than one that says what to do.
      cp_scale      the target is `sigmoid(cp / cp_scale)`, so training a checkpoint under a
                    different scale silently changes what every label means.
      target        an absolute net and a residual net predict different quantities; loading
                    one as the other is wrong by the whole hand evaluation.
      layer 1 shape the 768-input file has to be bucketised first. This is the mismatch that
                    will actually happen, so it names the tool that fixes it.
    """
    if not path.is_file():
        raise SystemExit(f"--init-checkpoint {path} does not exist")
    try:
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as failure:  # any unreadable checkpoint stops the run, whatever it raised
        raise SystemExit(f"--init-checkpoint {path} could not be read: {failure}") from failure
    for key in ("model", "hidden", "cp_scale"):
        if key not in checkpoint:
            raise SystemExit(f"--init-checkpoint {path} has no {key!r}; it is not a checkpoint")

    hidden = int(checkpoint["hidden"])
    if hidden != arguments.hidden:
        raise SystemExit(
            f"--init-checkpoint {path} is {hidden} wide and this run is --hidden "
            f"{arguments.hidden}"
        )
    cp_scale = float(checkpoint["cp_scale"])
    if abs(cp_scale - float(arguments.cp_scale)) > 1e-6:
        raise SystemExit(
            f"--init-checkpoint {path} was trained at cp_scale {cp_scale:g} and this run is "
            f"--cp-scale {arguments.cp_scale:g}; the labels would mean different things"
        )
    target = str(checkpoint.get("target", "cp"))
    if target != arguments.target:
        raise SystemExit(
            f"--init-checkpoint {path} predicts {target!r} and this run is --target "
            f"{arguments.target!r}"
        )
    state = checkpoint["model"]
    if "l1.weight" not in state:
        raise SystemExit(f"--init-checkpoint {path} has no l1.weight")
    shape = tuple(state["l1.weight"].shape)
    if shape != (arguments.hidden, NUM_FEATURES):
        hint = ""
        if len(shape) == 2 and shape[1] == NUM_FEATURES // 4:
            hint = (
                "; it is a 768-input net, so bucketise it first with "
                "`python -m tools.nnue.bucketize --checkpoint <in> --out <out>`"
            )
        raise SystemExit(
            f"--init-checkpoint {path} has l1.weight {shape}, this run wants "
            f"{(arguments.hidden, NUM_FEATURES)}{hint}"
        )

    # strict=True: a checkpoint missing a layer, or carrying one this model does not have, is a
    # different architecture and stops the run like every other mismatch above.
    model.load_state_dict(state, strict=True)
    provenance = {
        "path": str(path.resolve()),
        "sha256": checkpoint_digest(path),
        "hidden": hidden,
        "cp_scale": cp_scale,
        "target": target,
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "warm_started_from": checkpoint.get("warm_started_from"),
    }
    print(
        f"warm start: loaded {path} (epoch {provenance['epoch']}, val "
        f"{provenance['val_loss']}, sha256 {provenance['sha256'][:12]}), "
        f"l1.weight {shape}"
    )
    return provenance


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

    # The split above is drawn from `generator` and nothing below touches that stream, so the
    # rows a run trains and validates on are a function of `--seed`, `--val-fraction` and the
    # shard order alone -- the same with a warm start as without one. That is what makes a
    # fine-tune's validation loss comparable to the run it started from.
    model = Nnue(arguments.hidden).to(device)
    # Warm start *before* the optimiser is built, so Adam's parameter groups refer to the
    # tensors that will actually be trained. The optimiser itself starts fresh: no moment
    # estimates are carried over from the run that produced the checkpoint. That is a choice,
    # and the reason is that the moments belong to a different objective -- the 768-input net
    # was fitting a feature set a quarter this size -- so first- and second-moment estimates
    # from it describe gradients this model will never see again. A fresh Adam spends its first
    # few hundred steps rebuilding them, which is cheap next to an epoch, and it means a warm
    # start differs from a cold one in exactly one way: where the weights began.
    init_provenance: dict[str, Any] | None = None
    if arguments.init_checkpoint:
        init_provenance = load_init_checkpoint(Path(arguments.init_checkpoint), model, arguments)
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
                # Provenance, in every checkpoint rather than only the first: a fine-tune's
                # descendants are the files that get exported and benched, and "which net did
                # this start from" is the question a bench row cannot answer later. None when
                # the run started from random weights.
                "init_checkpoint": init_provenance,
            },
            checkpoints / f"epoch_{epoch:03d}.pt",
        )
        (checkpoints / "history.json").write_text(json.dumps(history, indent=2))
        if arguments.patience and epoch > arguments.patience:
            recent = min(row["val"] for row in history[-arguments.patience :])
            before = min(row["val"] for row in history[: -arguments.patience])
            if before - recent < arguments.min_delta:
                print(
                    f"stopping: val improved by less than {arguments.min_delta} over the last "
                    f"{arguments.patience} epochs"
                )
                break

    best = min(history, key=lambda row: row["val"])
    print(f"best val {best['val']:.6f} at epoch {int(best['epoch'])}")
    print(f"checkpoints in {checkpoints}, loss curve in {checkpoints / 'history.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the NNUE evaluation.")
    parser.add_argument("--data", default="tools/nnue/data/lichess", help="shard directory")
    parser.add_argument("--checkpoints", default="tools/nnue/checkpoints")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "warm start from this checkpoint instead of random weights; its width, "
            "cp scale, target and layer-1 shape must match this run, and a mismatch "
            "stops the run rather than falling back to random initialisation"
        ),
    )
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
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="stop when val has not improved by --min-delta over this many epochs; 0 = never",
    )
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    train(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
