# tools/nnue — offline training for the learned evaluation

This directory trains a small NNUE-style evaluation network for the agent. **Nothing in here
ships.** It is not imported by `agent.py`, `fastboard.py`, `fastsearch.py` or `harness/`, it is
not in `submission.zip`, and it never runs on the clock. The only thing that crosses the line
is one file of weights, `weights/nnue.npz`, and that arrives by a deliberate later PR.

Runtime inference will be written in numba, not torch, in a separate PR. That is why the export
is plain numpy integer arrays and why `nnue_ref.py` exists: it is the executable specification
that PR has to reproduce. Read `nnue_ref.py` and the quantisation section below and you can
write the runtime without reading anything else here.

## Why this is allowed

The rules (https://aichessathon.com/docs/rules.md, fetched 2026-09-08) say *"any network you ship
is one you trained yourself"* and *"training it on positions an existing engine labelled is
allowed"*. What is banned is shipping a third-party engine or a published net, and shipping a
table of another engine's evaluations for lookup at runtime.

So: the architecture, the training code and the weights are ours. Stockfish appears **offline
only, as a labeller**, and only under `tools/`. No Stockfish binary, no Stockfish-derived table
and no published net goes into the zip. What ships is 205 KB of integer weights the training run
in this directory produced.

## The two data paths

Both produce the same shard format, so you can train on either or on both.

**Path A — the lichess evaluations database.** `https://database.lichess.org/#evals` publishes
`lichess_db_eval.jsonl.zst`: ~395 million positions that Stockfish has already evaluated, CC0.
No engine needed locally, and it is by far the cheaper path. The catch is that the file is tens
of gigabytes compressed. You do not need all of it — the ingester streams and stops as soon as
it has the positions you asked for, so a **partial range download is enough**:

```bash
# 5M positions needs roughly 5M/395M of the file. Take a generous prefix; the ingester stops
# on its own and handles the half-frame and half-line at the end of a truncated download.
curl -r 0-3000000000 -o lichess_db_eval.jsonl.zst \
  https://database.lichess.org/lichess_db_eval.jsonl.zst
```

Check the real size first with `curl -sI https://database.lichess.org/lichess_db_eval.jsonl.zst`
and scale the range: the prefix you need is about `total_bytes * positions_wanted / 395e6`, times
two for headroom. Decompression uses the `zstandard` package (installed by `uv sync` from the dev
group) and falls back to shelling out to `zstd -dc` (`brew install zstd`) if it is missing.

One thing to know about this file: **`cp` is from White's point of view**, and the target has to
be side-to-move relative, so the ingester negates it when Black is to move. That convention is
worth a guard rather than a comment, because getting it wrong poisons every label silently, so
the ingester correlates the first 2000 labels against a crude material count **separately for
each side to move** and refuses to continue if either side comes out negative. The split matters:
with the convention wrong, only the Black-to-move half flips, and the pooled correlation stays
weakly positive (+0.06) and would sail through. Split by side it shows up as +0.71 / −0.63.

**Path B — self-generated games, labelled by a local Stockfish.** Plays fast games (a mixture of
random moves and shallow engine moves), samples positions from them, and labels each with
Stockfish at a fixed node count over a pool of workers, one engine process per worker. Slower
than Path A but it is *our* data: you control the position distribution, and you can aim it at
the kind of position the search actually reaches. `--pgn-dir` labels positions from existing PGNs
instead of generating games. Needs `brew install stockfish` (or `--stockfish /path`, or
`$STOCKFISH`).

Positions in the first 6 plies are skipped (opening theory is book knowledge, not evaluation
knowledge) and so are positions in check (their score is dominated by one forced reply rather
than by the features).

### Shard format

Every shard is a `.npz` with exactly three arrays:

| array | dtype | shape | meaning |
|---|---|---|---|
| `indices` | int16 | `[N, 32]` | active feature indices, padded with `-1` |
| `cp` | int16 | `[N]` | target: side-to-move relative centipawns, clipped to ±2000 |
| `wdl` | int8 | `[N]` | game result from the side to move: `+1` win, `0` draw or unknown, `-1` loss |

`wdl` is zero for every position on the lichess path, which has no game attached. `train.py`
does not use it yet; it is recorded because going back for it later would mean another full
labelling pass.

### `--mirror`, and a correction to the obvious version of it

`--mirror` emits a second copy of each position, **flipped left-to-right** (a↔h), with the same
label. It skips any position where either side still has castling rights, since the flip puts the
king on d1 with the rooks in the wrong places and the position is no longer equivalent.

It is worth being explicit that the *other* mirror — flip vertically and swap colours — is **not**
useful augmentation here, even though it is the one that first comes to mind. That transform is
already built into the feature scheme, so it produces a byte-identical feature vector and exactly
zero new information. `test_export.py` asserts that identity rather than assuming it.

## Residual target: the net as a correction to the hand evaluation

`train.py --target residual` fits `sigmoid((hand + net) / cp_scale)` to the same target as
before, where `hand` is `fasteval.evaluate` for the position. The net then learns only what the
hand evaluation gets wrong, and material sanity is the hand evaluation's by construction. The
loss is the same MSE in WDL space, so a residual run's validation loss is directly comparable
to a plain run's, and `train.py` prints two baselines at the start: the mean predictor and the
hand evaluation alone (net = 0). The sanity table for a residual net prints the *correction* in
centipawns, so expect numbers near zero.

The hand evaluation is added to the shards by `tools.nnue.hand`, which rebuilds each position
from its feature row (the shards hold no fen) and evaluates it with the modules in
`--engine-dir`. That rebuild is exact because the evaluation is colour-symmetric and reads
nothing the row lacks; `--verify` proves it on real records and refuses to run otherwise.

```bash
uv run python -m tools.nnue.hand --data tools/nnue/data/lichess --out tools/nnue/data/lichess-hand \
  --engine-dir . --verify lichess_db_eval.jsonl.zst
uv run python -m tools.nnue.train --data tools/nnue/data/lichess-hand --target residual \
  --hidden 256 --l1-clip 3.8
```

The exported file records `target` (`cp` or `residual`) so the runtime knows whether to add
the hand evaluation to the net's output. `--l1-clip 3.8` keeps every epoch exportable at
qa=256 (1.9 for qa=512); without it the layer-1 weights outgrow the int16 proof.

## Feature scheme

768 inputs = 12 piece planes × 64 squares, from the **side to move's** perspective. The index
formula, which the numba runtime must reproduce exactly:

```
INDEX = ((0 if piece_colour == side_to_move else 6) + (piece_type - 1)) * 64
        + (square ^ (56 if side_to_move == BLACK else 0))
```

- planes 0–5: **our** pawn, knight, bishop, rook, queen, king (`piece_type` 1–6)
- planes 6–11: **their** pawn, knight, bishop, rook, queen, king
- `square ^ 56` flips the board vertically (a1↔a8), so when Black is to move the network sees the
  position from Black's side with Black's men on the low ranks

Colour swap plus vertical flip means the net only ever learns "us versus them", which halves what
it has to learn and makes the evaluation automatically symmetric. At most 32 men can be on the
board, so at most 32 features are active, and index arrays are padded to 32 with `-1`.

## Architecture and target

`768 → 128 → 32 → 1`, clipped ReLU (`clamp(x, 0, 1)`) on both hidden layers. The clipping is not
decoration: quantised inference saturates at a fixed scale, and training with the same clamp makes
the float model and the integer model agree about what saturation does, so export is nearly
lossless.

The loss is MSE on `sigmoid(cp / 400)` (`--cp-scale` to change it). Centipawns are not linear in
winning chances, and an MSE straight on centipawns spends the network's capacity on lopsided
positions nobody needs evaluated precisely. The network's raw output is the **pre-sigmoid** value,
so the runtime reads centipawns back as `raw * 400` with no sigmoid at inference at all.

## Quantisation

`export.py` writes int16 weights with the scales stored in the file. The scales were chosen by
measurement, not by taste — the int-versus-float error over 1000 random positions is **1.9 cp
mean / 6.5 cp worst** at these values, against **26 cp mean / 85 cp worst** at the more obvious
`qa=64, qb=64, qc=128`:

| scale | value | applies to |
|---|---|---|
| `qa` | 1024 | layer-1 weights and biases, and therefore the hidden activations |
| `qb` | 512 | layer-2 weights |
| `qc` | 128 | layer-3 weights |
| `cp_scale` | 400 | centipawns per unit of network output |

Biases are stored on the scale of the sum they join, not of their own layer — `l1_bias` on `qa`,
`l2_bias` on `qa*qb`, `l3_bias` on `qa*qc` — so the runtime never has to rescale a bias.

`qb` is the scale the total error is most sensitive to, because all 128 hidden units contribute
their own rounding error to each of the 32 second-layer outputs. `qc` is deliberately small: it
keeps the final `z3 * cp_scale` product (worst observed 1.9e8) inside int32 with an order of
magnitude spare, which matters because the runtime does that multiply in int32.

The exported file, `weights/nnue.npz`, holds `version`, `hidden`, `qa`, `qb`, `qc`, `cp_scale`,
`l1_weight` int16 `[768,128]`, `l1_bias` int16 `[128]`, `l2_weight` int16 `[128,32]`, `l2_bias`
int32 `[32]`, `l3_weight` int16 `[32]`, `l3_bias` int32 scalar. **205,276 bytes uncompressed,
~110 KB on disk** — 0.2% of the 50 MB cap.

The int16 accumulator is *proved* safe rather than hoped to be: for every hidden neuron, the bias
plus the 32 largest weight magnitudes in that neuron's column must fit in int16. At most 32 men
can stand on a board, so that bound covers every reachable position. `export.py` prints the
headroom and refuses the export if any neuron could overflow.

## Commands, in order

Run everything from the repo root (these are namespace packages; `python -m` is required, running
the files directly will not resolve the imports).

```bash
uv sync                                   # installs zstandard into the dev group

# --- Path A: lichess evals (no Stockfish needed) ---
curl -sI https://database.lichess.org/lichess_db_eval.jsonl.zst      # check the size
curl -r 0-3000000000 -o lichess_db_eval.jsonl.zst \
  https://database.lichess.org/lichess_db_eval.jsonl.zst             # a prefix is enough
uv run python -m tools.nnue.data lichess \
  --input lichess_db_eval.jsonl.zst \
  --positions 5000000 --out tools/nnue/data/lichess --mirror

# --- Path B: our own games, Stockfish labels (alternative or supplement) ---
brew install stockfish
uv run python -m tools.nnue.data selfplay \
  --positions 5000000 --out tools/nnue/data/selfplay \
  --workers 8 --nodes 20000 --per-game 30 --mirror

# --- train ---
uv run python -m tools.nnue.train \
  --data tools/nnue/data/lichess --epochs 30 --hidden 128

# --- export and verify ---
uv run python -m tools.nnue.export \
  --checkpoint tools/nnue/checkpoints/epoch_030.pt --out weights/nnue.npz
uv run python -m tools.nnue.test_export \
  --weights weights/nnue.npz --checkpoint tools/nnue/checkpoints/epoch_030.pt
```

`--per-game 30` on Path B is deliberate: generating the game is most of the cost, so sampling
more positions from each one is close to free. See the timing note below.

## Timings

**Measured** on the machine this was built on: Apple M2 Pro, 10 cores (6P+4E), 16 GB, macOS 26.0,
Python 3.12, torch 2.13.0 with MPS.

| stage | scale | wall | rate |
|---|---|---|---|
| Path A ingest | 50k positions, 1 core | 3.4 s | 14,600 pos/s |
| Path B data, `repo-eval` labeller | 20k positions, 8 workers, `--mirror` | 5.1 s | 3,900 pos/s |
| Path B data, `repo-eval` labeller | 500k positions, 8 workers, `--mirror` | 180 s | 2,790 pos/s |
| train | 2 epochs on 20k | 2.8 s | — |
| train | 30 epochs on 500k | 16.7 s | 0.35 s/epoch |
| train | 3 epochs on 4M | 9.1 s | 1.65 s/epoch + 2.6 s load |
| export | H=128 | 0.75 s | — |
| `test_export` | 1000 positions | 0.97 s | — |

Path A was measured against a synthetic file built to the documented schema with realistic record
sizes (~1.5 KB/record uncompressed, 3 evals, multiple PVs, full-length UCI lines), because the
real 60 GB file is not on this machine. JSON parsing dominates, and that is what the record size
controls, so the number should hold; treat it as ±30%.

**Path B with a real Stockfish is not measured** — there is no Stockfish on this machine, which is
also why the smoke test below uses a stand-in labeller. The estimate, with the arithmetic shown so
you can correct it: Stockfish on an Apple P-core does roughly 3 Mnodes/s single-threaded, so 20k
nodes is ~6 ms per label; game generation at `--play-depth 4` costs ~1.5 ms per ply, so a
100-ply game costs ~110 ms before any labelling. At `--per-game 10` that is ~17 ms per position
per worker; at `--per-game 30` it amortises to ~10 ms. On 8 workers, ~800–850 pos/s.

### Estimates for the M5

**Stated assumption: the M5 is 1.5× the M2 Pro per core and 1.5× for this GPU workload, at the
same core count (10).** The measured M2 Pro column is the ground truth; if the M5 turns out to be
1.3× or 2×, rescale. Training uses MPS, data generation uses the CPU pool.

| | 5M positions | 20M positions |
|---|---|---|
| Path A ingest (1 core) | **~4 min** | **~15 min** |
| Path B, Stockfish 20k nodes, 8 workers, `--per-game 30` (estimate) | **~1.5 h** | **~6 h** |
| train, 30 epochs | **~1.5 min** | **~5 min** |
| export + verify | **~2 s** | **~2 s** |

Training is not the bottleneck and never will be at this network size — data is. If you have the
choice, spend the time on Path A volume, not on epochs.

### Disk and memory

- **Shards on disk:** 14–27 bytes/position compressed (27 on the lichess path, where the labels
  are more varied and compress less). So **5M ≈ 150 MB, 20M ≈ 600 MB**.
- **Download for Path A:** the range prefix, which is the dominant disk cost. Budget a few GB for
  5M positions and scale up from the `curl -sI` size; the compressed file averages well under a
  kilobyte per position.
- **RAM while training:** 67 bytes/position (32 int16 indices + int16 cp + int8 wdl), so **5M =
  335 MB, 20M = 1.34 GB**, plus a float32 target array (`4 × N`) and a transient doubling while
  `load_shards` concatenates. 20M peaks near 3 GB. Comfortable in 48 GB.
- **Checkpoints:** ~1 MB each and one per epoch, so ~30 MB for a 30-epoch run.

## How to check the result

Three things, in increasing order of how much they tell you.

**1. The loss curve.** `train.py` prints train and validation loss every epoch and writes
`tools/nnue/checkpoints/history.json` with `{epoch, train, val, seconds}`. Validation is a 2%
split. Loss is MSE in WDL space, so the numbers are small; what matters is that validation keeps
falling with training and does not turn up. If validation diverges from training, you are out of
data, not out of epochs.

**2. The sanity table**, printed after every epoch: raw centipawn predictions for the start
position, a position a pawn up, a position a rook up, and a mate in one. This catches the
failures a loss number hides — a sign error, a broken perspective flip, a dead output. A working
run should show roughly 0 for the start position and a clear ordering pawn < rook. From the smoke
run in this directory, after 30 epochs:

```
start  -1cp  |  +1 pawn (white)  +91cp  |  +1 rook (white)  +497cp  |  mate in 1 (white)  +161cp
```

The pawn and rook numbers landing on ~100 and ~500 is the whole pipeline agreeing with itself:
features, perspective, sign convention, target transform and the read-back scale. (The mate-in-1
number being unremarkable is correct and expected — the network is a *static* evaluation, it
cannot see that a mate is available. That is the search's job.)

**3. `test_export.py`** — the exact-match test, and the one that gates the export. 1000 positions
from random games, and three assertions:

- **Agreement:** integer inference matches the torch float model within 5 cp mean / 25 cp worst.
  This is what catches a wrong scale, a transposed layer, a bias on the wrong scale, or an
  accumulator that wraps. The rejected scale set failed it at 26 cp / 85 cp.
- **Exact match:** re-encoding the same positions gives bit-identical integer output.
- **Perspective invariant:** `board.mirror()` (vertical flip + colour swap + turn swap) must
  produce identical features and therefore an *identical* evaluation.

A note on that last one, because the expected form is the negated one: `eval(pos) ==
-eval(pos with the turn flipped)` is **not** true, here or for any NNUE. Flipping only the side
to move is a null move — a different position with a different feature set — and there is no
invariant relating the two. The statement that *is* exactly true, and is what the test asserts,
is the mirror identity above: both numbers mean "how good this is for the player to move", and
the mirror is the same position seen from the other side.

## The smoke test in this directory

There is no Stockfish on the machine this was built on, so the end-to-end run used the
`--labeller repo-eval` stand-in, which labels with the repo's own hand-written `agent.evaluate`
(`smoke_eval.py`, which imports it — note the direction: `tools/` importing a root module is
fine, a root module importing `tools/` is what must never happen). **This is a smoke path only.**
A net trained on those labels learns to imitate a static material-and-piece-square score, so it
can never be better than the thing it copies. It exists to prove the wiring, and it does: the
sanity table above is the smoke net correctly reproducing ~100 cp for a pawn and ~500 cp for a
rook.

The four stages, exactly as run:

```
data     20,008 positions, 8 workers, --mirror     5.1 s
train    2 epochs                                  2.8 s
export   205,276 B uncompressed, 109,994 B on disk  0.75 s
verify   1000 positions: 0.90 cp mean, 1.79 cp max  0.97 s   OK
```

A longer run — 500k positions, 30 epochs — produced the sanity table quoted above and verified at
1.92 cp mean / 6.49 cp worst.

Path A was exercised separately against synthetic files in the documented schema: the whole file,
a deliberately truncated `curl -r` prefix (stops cleanly), the `zstd -dc` fallback with the
`zstandard` import blocked, and a deliberately wrong `--pov` (aborts, as intended).

## Files

| file | what it is |
|---|---|
| `features.py` | the 768-feature scheme; single-position and batch encoders |
| `data.py` | both data paths, both writing the same `.npz` shards |
| `train.py` | the torch model and training loop |
| `export.py` | quantise a checkpoint to `weights/nnue.npz` |
| `nnue_ref.py` | integer inference — **the spec for the numba runtime** |
| `test_export.py` | the three checks above |
| `smoke_eval.py` | smoke-only stand-in labeller using the repo's own `evaluate` |

`tools/nnue/data/`, `tools/nnue/checkpoints/` and `weights/*.npz` are gitignored. Data, shards,
checkpoints and weights are never committed; the real weights ship later by a deliberate PR.

One thing the later PR should know: `harness/package.py` already has `DEFAULT_INCLUDES =
("weights",)`, so a `weights/` directory is packaged into `submission.zip` automatically and no
packaging change is needed. The flip side is that **any** `weights/nnue.npz` sitting in the tree
gets packaged, smoke net included, so this branch deliberately leaves no weight file behind —
re-run `export.py` when you want one.
