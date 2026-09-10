# k4-v44 warm start — the fine-tune is set up and blocked on data

Branch `nnue/king-buckets-4-v44`, based on `origin/prod` **83cd33b** (engine v4.4, tag
**f96597d**). This note records what is established and what the training run still needs. **No
fine-tuned candidate exists yet and `weights/nnue.npz` is untouched** — it is still the shipped
v4.4 file, `fc7c862f1b0be11d5563f4bcf46e8a93361db743fe1d513a9993823edd74543f`.

## What is blocked, and why nothing was faked

The fine-tune asked for was: 3072 → 256 → 32 → 1, target cp, cp_scale 400, seed 1,
val-fraction 0.02, lr 3e-4, batch 8192, 8 epochs, patience 3, min-delta 1e-5, l1-clip 3.8, warm
started from the h256 epoch-60 checkpoint over the 90M-row scheme-1 shards.

**Neither asset is on this machine.** A search of `/Users/andrew` and `/Volumes` to depth 6-7
found no `shard_*.npz`, no `epoch_*.pt`, no `lichess_db_eval*`, and no `lichess*` or
`checkpoints*` directory:

| asset | expected at | status |
|---|---|---|
| 90M-row scheme-1 shards | `tools/nnue/data/lichess-48m` | **absent** |
| float checkpoint | `tools/nnue/checkpoints-h256-90m-long/epoch_060.pt` | **absent** |
| raw evaluations database | `lichess_db_eval.jsonl.zst` | **absent** |

The checkpoint half is solved below. The **shards are the hard blocker**: there is no training
without them, and the two ways to manufacture some would both be worse than not training.
Re-ingesting the source database is a 15 GB download and hours of ingestion, not the "quick"
run this was scoped as. Labelling self-play with our own hand evaluation would train the net to
imitate `fasteval`, which is a different experiment with a known answer, and a net from it must
never be described as a fine-tune of the shipped one. So no weights were produced.

## What is established

### The float checkpoint, rebuilt from the shipped weights

`tools/nnue/dequantize.py` reconstructs the float model the export was rounded from: each layer
divided by the scale `export.py` multiplied it by, biases on the scale of the sum they join.
Because the runtime already expands a scheme-1 file into four identical blocks, the rebuilt
checkpoint comes out **3072 x 256 directly**, which is the warm start — no separate
`bucketize` step is needed for this path.

```
uv run python -m tools.nnue.dequantize --weights weights/nnue.npz \
    --out tools/nnue/checkpoints-dequantised/epoch_000.pt --positions 2000
```

    rebuilt 3072 x 256 from weights/nnue.npz (qa 256, qb 1024, qc 128, cp_scale 400)
    float model against the integer pipeline over 2000 positions:
        1.40 cp mean, 6.49 cp worst   (tolerance 5 / 25)

Checkpoint sha256 `a34780f0d4eee9295ec8ff0decfd940ff4bc017115adb7205c3db199bafd11fb`,
cp_scale 400.0, target `cp`, four blocks verified identical. The tolerance is the one
`tools/nnue/test_export.py` holds the opposite direction to, and the gap is the quantisation
step itself: dequantising cannot recover what rounding discarded, and does not need to, because
a warm start needs the position in weight space rather than its last bit.

### The shard converter

`tools/nnue/rebucket.py` turns scheme-1 shards into scheme 2 without relabelling: it finds the
one plane-5 friendly-king feature in each row, computes the four-way bucket from its square, and
adds `bucket * 768` to every active index. Row order, shard boundaries, `cp`, `wdl` and any
array it does not recognise are passed through untouched, and the marker is set to `scheme=2`.
It refuses malformed padding, out-of-range indices, a row with no friendly king, a row with two,
and a shard that is already scheme 2.

Verified in `tests/test_king_buckets.py::check_rebucket` against **direct scheme-2 feature
extraction** on 2,600 real positions, half of them mirrors, covering all four buckets, plus the
explicit property that **every converted index equals the original modulo 768** and that a row's
offset is always a whole block.

### The chain, end to end

Proved on a synthetic 4,000-row scheme-1 shard, because the shape of the pipeline can be tested
without the real corpus:

1. `tools.nnue.train` **refuses** the un-converted scheme-1 shards and writes nothing.
2. `tools.nnue.rebucket` converts them; the reported bucket spread on that sample was
   bucket 0 8.2%, bucket 1 91.6%, bucket 2 0.0%, bucket 3 0.2%.
3. `tools.nnue.train --init-checkpoint` loads the dequantised checkpoint and trains: warm start
   logged with its sha256, split identical to a cold run, provenance in the checkpoint.
4. After one epoch the four first-layer blocks are **no longer identical**, which is the thing a
   warm start has to make possible.

**Read that bucket spread as a warning for the real run.** Kings sit in their own half for most
of a game, so buckets 2 and 3 (the far half) are rare. Before selecting any fine-tuned
checkpoint, the per-bucket row counts have to be checked on the real corpus: a bucket with too
few positions learns noise, and that is the failure mode this whole four-bucket design was
chosen to avoid.

## What the run needs

1. **The 90M-row scheme-1 shards** from the M5, copied to this machine or the run moved there.
2. `uv sync --extra nnue` in whichever worktree trains; torch is the optional extra.
3. `uv run python -m tools.nnue.rebucket --data <shards> --out <converted>` into a directory
   outside `weights/`, leaving the originals untouched.
4. `uv run python -m tools.nnue.dequantize` as above, **or** the real
   `checkpoints-h256-90m-long/epoch_060.pt` if it can be fetched, which is preferable: it is the
   float model itself rather than a reconstruction of it.
5. The configured run, checkpoints outside `weights/`:

```
uv run python -m tools.nnue.train --data <converted> --checkpoints tools/nnue/checkpoints-k4 \
    --hidden 256 --epochs 8 --batch-size 8192 --lr 3e-4 --cp-scale 400 --target cp \
    --seed 1 --val-fraction 0.02 --patience 3 --min-delta 1e-5 --l1-clip 3.8 \
    --init-checkpoint tools/nnue/checkpoints-dequantised/epoch_000.pt
```

6. Export the lowest-validation-loss epoch at `--qa 256 --qb 1024` (qc 128, cp_scale 400,
   scheme 2), then `tools.nnue.test_export`, the king-bucket suite, `tests.test_nnue --full`,
   `make gate`, `make zip`, and only then a bench against `local-opponents/v4.4`.

Memory note: `densify` builds a dense batch, so batch 8192 is about 100 MB and batch 16384 about
201 MB; drop the batch size before anything else if the machine is tight.

## Runtime cost, measured on this tree

The four-bucket runtime is the same tree measured on `nnue/king-buckets-4-vnext`, whose content
is identical to this branch's: **fixed-depth 7 over the regression positions, 65/65 identical
scores and moves against v4.4 and a single node count (44,988,245) across twelve runs**, at a
node-rate cost of about **3%** (quiet-pass ratios 0.97 / 0.96 / 0.98 / 0.99 / 0.95; the
least-contended pair reads -1.5%). Warm-start identity is exact by construction and stays exact
until a fine-tune moves the blocks apart.
