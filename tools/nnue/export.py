"""Quantise a trained checkpoint into ``weights/nnue.npz``. Offline only.

The scales, and why they are these numbers. They were chosen by measurement, not by taste:
the int-versus-float error over 1000 random positions is 1.9cp mean / 6.5cp worst at these
values, against 26cp mean / 85cp worst at the more obvious qa=64, qb=64, qc=128.

  qa = 1024  layer-1 weights and biases, and therefore the hidden activations. A clipped ReLU
             output lives in [0, 1], so on this scale it lives in [0, 1024]. The accumulator
             stays int16: the worst hidden neuron of the smoke net reaches 5962 of 32767, and
             _check_accumulator below refuses the export if any neuron could exceed the limit.
  qb = 512   layer-2 weights. This is the scale the total error is most sensitive to, because
             every one of the 128 hidden units contributes its own rounding error to each of
             the 32 second-layer outputs. The layer-2 product lands on scale qa*qb in int32
             (worst seen: 1.4e6) and is divided by qb back down to qa before the second
             clipped ReLU. It assumes |layer-2 weight| < 64, which _quantise enforces.
  qc = 128   layer-3 weights. Only 32 inputs and one output, so this layer contributes least;
             128 keeps the final `z3 * cp_scale` product (worst seen: 1.9e8) inside int32 with
             an order of magnitude to spare, which matters because the runtime does that
             multiply in int32.

Every rescale is floor division, never rounding: rounding measured no better here, and floor is
one operation the numba runtime cannot get subtly wrong.

Biases are stored on the scale of the sum they join, not the scale of their own layer:
l1_bias on qa, l2_bias on qa*qb, l3_bias on qa*qc. That way the runtime never rescales a bias.

The int16 accumulator is proved safe here rather than hoped for: for every hidden neuron, the
bias plus the 32 largest weight magnitudes in that neuron's column must fit in int16. At most 32
men can be on the board, so that bound covers every reachable position. If it fails, the export
refuses rather than shipping a net that wraps around in some endgame.

Run from the repo root:

    uv run python -m tools.nnue.export --checkpoint tools/nnue/checkpoints/epoch_030.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from tools.nnue.features import MAX_ACTIVE
from tools.nnue.nnue_ref import SCHEME_VERSION
from tools.nnue.train import Nnue

QA = 1024
QB = 512
QC = 128
INT16_MAX = 32767


def _quantise(array: np.ndarray, scale: float, dtype: type[np.signedinteger]) -> np.ndarray:
    scaled = np.rint(array * scale)
    limit = np.iinfo(dtype).max
    if np.abs(scaled).max(initial=0.0) > limit:
        raise SystemExit(
            f"a weight of {np.abs(scaled).max():.0f} does not fit {np.dtype(dtype).name}; "
            f"lower the scale or retrain with weight decay"
        )
    return scaled.astype(dtype)


def _check_accumulator(l1_weight: np.ndarray, l1_bias: np.ndarray) -> int:
    """Return the smallest int16 headroom over all hidden neurons, refusing if any is negative.

    ``l1_weight`` is [768, hidden]; column ``h`` holds every feature's contribution to neuron h.
    """
    magnitudes = np.abs(l1_weight.astype(np.int32))
    top = np.sort(magnitudes, axis=0)[-MAX_ACTIVE:].sum(axis=0)
    worst = top + np.abs(l1_bias.astype(np.int32))
    headroom = int(INT16_MAX - worst.max())
    if headroom < 0:
        raise SystemExit(
            f"the int16 accumulator can overflow by {-headroom}: worst neuron reaches "
            f"{int(worst.max())} against a limit of {INT16_MAX}. Lower qa or retrain with "
            f"weight decay."
        )
    return headroom


def export(arguments: argparse.Namespace) -> Path:
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    hidden = int(checkpoint["hidden"])
    cp_scale = round(float(checkpoint["cp_scale"]))
    model = Nnue(hidden)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # qa is a flag because a net trained without a weight constraint can carry layer-1
    # weights past 1.0, and then the int16 proof below fails at 1024 on the 32-men bound
    # even though real positions stay far below it. Halving qa halves the bound; the runtime
    # reads the scale from the file, so nothing downstream changes.
    qa = arguments.qa
    qb = arguments.qb
    with torch.no_grad():
        # torch.nn.Linear stores [out, in]; the runtime wants to gather whole feature rows, so
        # layer 1 is transposed to [768, hidden] and layer 2 to [hidden, 32].
        l1_weight = _quantise(model.l1.weight.T.numpy(), qa, np.int16)
        l1_bias = _quantise(model.l1.bias.numpy(), qa, np.int16)
        l2_weight = _quantise(model.l2.weight.T.numpy(), qb, np.int16)
        l2_bias = _quantise(model.l2.bias.numpy(), qa * qb, np.int32)
        l3_weight = _quantise(model.l3.weight.reshape(-1).numpy(), QC, np.int16)
        l3_bias = _quantise(model.l3.bias.numpy(), qa * QC, np.int32)

    headroom = _check_accumulator(l1_weight, l1_bias)
    print(f"int16 accumulator headroom: {headroom} of {INT16_MAX} (worst hidden neuron, 32 men)")

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        version=np.int32(SCHEME_VERSION),
        hidden=np.int32(hidden),
        qa=np.int32(qa),
        qb=np.int32(qb),
        qc=np.int32(QC),
        cp_scale=np.int32(cp_scale),
        l1_weight=l1_weight,
        l1_bias=l1_bias,
        l2_weight=l2_weight,
        l2_bias=l2_bias,
        l3_weight=l3_weight,
        l3_bias=np.int32(l3_bias.reshape(())),
    )
    size = out.stat().st_size
    print(
        f"wrote {out} ({size:,} bytes, {size / 1e6:.2f} MB) "
        f"hidden={hidden} qa={qa} qb={qb} qc={QC} cp_scale={cp_scale} "
        f"scheme_version={SCHEME_VERSION}"
    )
    if size > 40_000_000:
        raise SystemExit("weight file is too large to leave room under the 50 MB zip cap")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantise a checkpoint to int16 weights.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="weights/nnue.npz")
    parser.add_argument(
        "--qa", type=int, default=QA, help="layer-1 scale; halve it if the int16 proof fails"
    )
    parser.add_argument(
        "--qb", type=int, default=QB, help="layer-2 scale; doubling it buys back layer-2 rounding"
    )
    export(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
