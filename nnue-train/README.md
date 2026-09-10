# NNUE training — "2 Pawns and a Queen"

Trainer side of the split: this box owns data generation and training and produces a
weights file. The engine session owns `agent.py`, `fastboard.py`, `fasteval.py`,
`fastsearch.py`, `fastnnue.py` and `harness/`. Nothing here touches those.

## What the engine session needs

| file | what it is |
|---|---|
| `spec/king_relative_v1.md` | the input/architecture/output contract — read this first |
| `spec/test_vectors.json` | 8 positions with expected feature indices; use to verify `fastnnue.py` |
| `src/reference_eval.py` | standalone numpy implementation of the whole evaluation |
| `runs/nnue_v2_512.npz` | **recommended weights** (float; see spec section 7) |
| `runs/nnue_v1.npz`, `runs/nnue_v3_512quiet.npz` | narrower / quiet-trained alternatives |

`reference_eval.py` is the spec in executable form and is verified against the test
vectors. If `fastnnue.py` agrees with it on those 8 positions, the encoding is right.

## Pipeline

```
lichess_db_eval.jsonl.zst      22 GB, ~210M Stockfish-evaluated positions
  -> src/pack.py               -> 32-byte packed records (board + score), ~600k pos/s
  -> src/train.py              -> GPU training, ~2.5M pos/s
  -> src/export.py             -> .npz (float) + .nnue (quantised), with a drift check
```

**The packed format stores boards, not features.** That is deliberate: the king-relative
encoding was chosen without the engine session reachable, so if the contract changes,
re-encoding costs minutes rather than re-labelling 210M positions.

## Dataset

409,562,339 positions from the lichess evaluation database (22 GB compressed, dated
2026-09-10), each labelled with a deep Stockfish search. Median search depth 24.
12.3% are mate announcements; 0.2% are exact duplicates. 147,774 illegal positions were
rejected during packing. All 409M packed records pass an integrity sweep
(`src/verify_full.py`): occupancy/nibble agreement, piece counts, one king per side,
score range, padding.

## Verification done

Every stage is checked against an independent implementation rather than eyeballed:

- `src/check_sign.py` — proves the lichess `cp` field is **white-relative**, not
  side-to-move-relative (82% vs 56% agreement with material on decisive positions).
  A flip here would have inverted every label.
- `src/test_posformat.py` — fast FEN parser vs python-chess: 28545/28545 exact.
  Found that the lichess DB contains **illegal user-constructed positions** (17 pieces
  for one side, 56 white pawns); these are now rejected, ~0.04% of the file.
- `src/test_features.py` — vectorised encoder vs a literal transcription of the spec:
  20000/20000 exact on both perspectives.
- GPU feature expansion vs the verified numpy encoder: bit-identical.
- `src/reference_eval.py` vs `spec/test_vectors.json`: 8/8.
- `src/export.py` — quantised net checked against float, max drift reported in cp.
  Caught that **numpy silently wraps on integer overflow**: 6.25% of output-layer
  weights exceeded int8 at scale 64 and wrapped, corrupting the net by 222 cp mean.
  Now the trainer constrains hidden weights and the exporter refuses to wrap.
- `src/verify_full.py` — all 409,562,339 packed records checked for internal consistency.
- `src/test_export_roundtrip.py` — torch model vs float export vs integer export agree
  (float to 0.0003 cp, integer to <5 cp).

Run everything with `./run_tests.sh <weights-prefix> <checkpoint.pt>`.

## Known limitations

- Positions where the side to move is **in check** are not filtered (would need movegen
  in the packer). Search normally extends out of check, so the net is rarely asked to
  evaluate these.
- Labels are deep-search evals, so they encode tactics the net cannot structurally see.
  This is standard for publicly-trained NNUEs and works in practice, but it caps how low
  the loss can go.
- 0.2% of positions are exact duplicates; train/val leakage from this is negligible.
- **Endgame precision is soft, and width helps.** On `8/8/8/4k3/8/8/4P3/4K3 w` (a
  theoretically drawn K+P vs K, black king holding the square in front of the pawn) the
  L1=256 net says +213 cp — wrong, it is a draw — while L1=512 says +55 cp and passes.
  Drawn-but-material-up endgames need exact calculation and are rare in the training
  distribution, so this is the expected soft spot of an eval-only net. The sanity suite
  keeps the case at a chess-correct threshold rather than a net-flattering one, so the
  weakness stays visible. Search depth and/or tablebases cover it in practice.

## Reproducing

```bash
.venv/bin/python src/pack.py  data/lichess_db_eval.jsonl.zst data/full2.bin --workers 20
.venv/bin/python src/train.py data/full2.bin --out runs/full512 --epochs 10 --l1 512 --min-depth 16
.venv/bin/python src/export.py runs/full512/best.pt --out runs/nnue_v2_512 --data data/full2.bin
./run_tests.sh runs/nnue_v2_512 runs/full512/best.pt
```

## If the engine needs a different feature format

The claim that a spec change is cheap is concrete, not aspirational — the packed dataset
stores boards, so nothing needs re-labelling. To switch (say to HalfKP, or to drop
mirroring):

1. Edit the index formula in `src/features.py:encode_batch` and the matching
   `NUM_FEATURES`, then mirror the same change in `src/model.py:expand_features` and
   `src/reference_eval.py:feature_indices`. These three must agree — that is exactly what
   the tests below check.
2. Update the reference transcription in `src/test_features.py:ref_features` and the
   generator in `src/make_test_vectors.py` (they are deliberately independent
   transcriptions of the spec, so do not copy-paste from `features.py`).
3. Re-run `./run_tests.sh` — `test_features` (encoder vs spec), the GPU-vs-numpy check in
   `model.py`, and `test_symmetry` (colour-mirror invariance) will catch a mismatch.
4. Retrain: ~30 min for 10 epochs over 403M positions on this GPU.

Total turnaround is well under an hour. The 22 GB download and the 11-minute pack do not
need to be repeated.

## Results

All figures on the **same 1.5M-position holdout**, which no run trained on. The tactical
filter is applied to the training split only, precisely so the holdout stays common —
filtering the validation set too would flatter the filtered net, since quiet positions
are simply easier to evaluate.

"quiet" = best move is neither a capture nor a promotion, i.e. what a quiescence search
actually hands to a static eval (76.2% of the holdout).

| net | L1 | trained on | all positions | quiet positions |
|---|---|---|---|---|
| pilot | 256 | 88M | 0.01406* / 130.8 cp / 91.6% | — |
| `nnue_v1` | 256 | 403.6M | 0.01169 / 117.4 cp / 92.9% | 0.00966 / 106.2 cp / 94.6% |
| `nnue_v2_512` | 512 | 403.6M | **0.01088 / 114.1 cp / 93.5%** | 0.00903 / 103.6 cp / 95.1% |
| `nnue_v3_512quiet` | 512 | 307.4M quiet | 0.01440 / 133.8 cp / 92.2% | **0.00842 / 100.1 cp / 95.5%** |

\* the pilot trained on the first 90M positions of the file, so part of this holdout was
in its training data — its true number is *worse* than shown. Included only to show the
gain from using the whole database.

Three things this establishes:

1. **More data helps a lot.** 88M -> 403M cut val_loss 17% (0.01406 -> 0.01169) at
   identical architecture.
2. **Width is nearly free and clearly better.** L1 256 -> 512 cut val_loss another 7%,
   and costs only ~15% more inference time in numpy (see spec section 6). The wider net
   also fixed a drawn K+P endgame the narrow one got wrong (+55 cp vs +213 cp).
3. **Filtering tactical positions is a real trade, not a free win.** Dropping the 23.8%
   of positions whose best move is a capture makes the net 6.8% better on quiet positions
   and much worse everywhere else. Whether that is the right trade depends on whether
   `fastsearch.py` calls eval only at quiescence leaves — see spec section 7.

`nnue_v2_512.npz` is the recommended default.

