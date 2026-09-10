# King-relative NNUE input spec — v1

Contract between the trainer (GPU box) and the engine (`fastnnue.py`).
**Any mismatch here is silent: the net will load, evaluate, and be wrong.**
Verify against the test vectors in `spec/test_vectors.json` before trusting a net.

## 1. Feature set — HalfKA, mirrored

Two feature vectors per position, one per perspective. Each is indexed by **that
perspective's own king square**, so the net sees the position from its own king's view.

```
NUM_FEATURES = 32 (king buckets) * 12 (piece planes) * 64 (squares) = 24576
```

Squares use **a1 = 0, b1 = 1, ..., h8 = 63** (little-endian rank-file).

### Transform

For perspective `P` (0 = white, 1 = black):

```python
orient(sq)    = sq ^ 56 if P == BLACK else sq        # vertical flip only, never ^63
mirror        = (orient(own_king_sq) & 7) >= 4       # own king on files e-h
transform(sq) = orient(sq) ^ 7 if mirror else orient(sq)
```

`transform` is applied to **every square**, the king's included. After mirroring the
king always stands on files a–d, which is what halves 64 king squares to 32.

### Index

```python
tksq        = transform(own_king_sq)
king_bucket = (tksq >> 3) * 4 + (tksq & 7)     # 0..31
rel_colour  = 0 if piece_colour == P else 1    # 0 = own, 1 = enemy
p_idx       = rel_colour * 6 + piece_type      # piece_type: P,N,B,R,Q,K = 0..5
index       = king_bucket * 768 + p_idx * 64 + transform(piece_sq)
```

Every piece contributes one index to each perspective, **kings included** (HalfKA, not
HalfKP). Piece count = active feature count, so 3–32 indices per perspective.

Rationale: HalfKP drops both kings from the piece planes, so a HalfKP net cannot see
where the enemy king is — a real cost in endgames and king safety. Including them costs
20% more features, and mirroring more than pays that back: 24576 features vs 40960 for
unmirrored HalfKP, so the feature transformer is *smaller* while strictly better informed.

## 2. Architecture

```
HalfKA-24576 -> 256x2 -> 32 -> 32 -> 1
```

1. Accumulator: for each perspective, sum the 256-wide rows of the active features, then
   add the shared `ft_bias` (one bias vector, used by both perspectives).
2. Concatenate **side-to-move first**: `x = [acc_stm, acc_nstm]`, width 512.
3. `x = clamp(x, 0, 1)`
4. `x = clamp(lin1(x), 0, 1)`   (512 -> 32)
5. `x = clamp(lin2(x), 0, 1)`   (32 -> 32)
6. `out = lin3(x)`              (32 -> 1)

Activation is **clipped ReLU on [0, 1]** everywhere (not [0, 127] — that is the
integer-domain equivalent; see quantisation).

## 3. Output convention

`out` is **side-to-move relative** (positive = good for the player to move) and is in
logit units:

```
centipawns = 400 * out
```

The trainer optimises `MSE(sigmoid(out), sigmoid(cp_stm / 400))`, so the net is most
accurate in the decisive-but-not-won range, which is where search actually needs it.

Note the training data (lichess eval DB) is *white*-relative; the trainer negates it for
black to move. The engine must not negate again.

## 4. Incremental updates

The accumulator is incrementally updatable, but note the king-relative catch: **when the
perspective's own king moves, every feature for that perspective changes** and its
accumulator must be rebuilt from scratch. The other perspective is unaffected. A king
move that crosses the d/e file boundary also flips `mirror`, which is already covered by
the full rebuild.

For a non-king move, update only the affected pieces:
`acc -= row(from_feature)`, `acc += row(to_feature)`, and `acc -= row(captured)`.
Castling moves two pieces; a promotion removes a pawn and adds the promoted piece;
en passant removes a pawn from a square the moving piece never occupied.

**Cache the per-perspective transform.** `king_bucket`, the orientation flip and the
mirror flag depend only on that perspective's own king square, so compute them once and
store them next to the accumulator, recomputing only on an own-king move
(`KingRelativeNNUE.king_context` in `reference_eval.py` does exactly this). Recomputing
them per touched piece costs ~15% of the update.

## 5. Weight export

`export.py` writes two files; use whichever suits the engine.

**Float (`*.npz`)** — simplest, recommended first:
| array | shape | notes |
|---|---|---|
| `ft.weight`   | (24576, 256) | row `i` = feature `i`; row 24576 (padding) is dropped |
| `ft.bias`     | (256,)       | shared by both perspectives |
| `lin1.weight` | (32, 512)    | row-major, `y = W @ x + b` |
| `lin1.bias`   | (32,)        | |
| `lin2.weight` | (32, 32)     | |
| `lin2.bias`   | (32,)        | |
| `lin3.weight` | (1, 32)      | |
| `lin3.bias`   | (1,)         | |

**Quantised (`*.nnue`)** — integer path, ~1.5 cp mean drift from the float net:

Header (little-endian int32 after a 4-byte magic):

| offset | field |
|---|---|
| 0 | magic `2PQN` |
| 4 | spec version (1) |
| 8 | num features (24576) |
| 12 | accumulator width L1 |
| 16 | `QA` — activation scale (2032) |
| 20 | `QH` — hidden weight scale (1024) |

Then, contiguous: `ft_w` int16 `(24576, L1)`, `ft_b` int16 `(L1,)`, then for each of
lin1/lin2/lin3 an int16 weight block followed by an int32 bias block.

Integer evaluation:

```python
acc   = ft_b.astype(int32) + ft_w[active_features].sum(0)   # int32, NOT int16
x     = clip(concat([acc_stm, acc_nstm]), 0, QA)
for W, b in (lin1, lin2):
    x = clip((W @ x + b + QH//2) // QH, 0, QA)     # round, do not truncate
out   = (lin3_W @ x + lin3_b) / (QA * QH)
cp    = 400 * out
```

Two deliberate departures from the Stockfish convention, both measured:

- **int16 hidden weights at scale 1024, not int8 at 64.** int8 costs 18 cp of mean drift.
  The hidden layers are only ~17k weights, so int16 costs 17 KB — nothing. 1.5 cp.
- **QA = 2032, not 127, with an int32 accumulator.** Stockfish keeps QA=127 so the
  accumulator fits int16 and SIMD gets twice the lanes. numpy has no int8/int16 SIMD path
  to exploit, so that trade buys nothing here and costs ~10 cp of resolution.

If `fastnnue.py` needs an int16 accumulator after all, say so and I will re-export at
QA=127 — but expect ~18 cp of drift, and prefer the float net in that case.

## 6. Measured inference cost (numpy, this net, L1=256)

| operation | cost | note |
|---|---|---|
| full accumulator refresh (both sides) + layers, float | 49 us | ~20k eval/s |
| same, quantised int16 | 75 us | **slower** — see below |
| incremental update, one piece moved | 1.9 us | ~540k updates/s |
| dense tail 512->32->32->1 | 12.9 us | dominates once accumulators are incremental |
| **incremental eval per node** | **~14.8 us** | ~68k nodes/s from eval alone |

Three consequences worth acting on:

1. **Use the float `.npz` net, not the quantised one.** Measured: the integer path is
   *50% slower* in numpy (75 us vs 49 us), because numpy dispatches float matmuls to BLAS
   and has no equivalent integer path. The usual NNUE argument for quantisation assumes
   hand-written SIMD, which does not apply here. The float net is simultaneously more
   accurate (exact rather than ~1 cp drift) and faster. The `.nnue` file is provided only
   in case `fastnnue.py` has its own integer kernel that beats numpy.

2. **Incremental accumulators are worth the complexity** — 14.8 us vs 49 us per node,
   a 3.3x difference. Remember the king-move rebuild rule in section 4.
3. **Do NOT shrink L1 to gain speed.** The dense tail is numpy *call-overhead* bound,
   not FLOP bound. Measured end-to-end tail cost:

   | L1 | MACs | tail cost |
   |---|---|---|
   | 64 | 5,152 | 14.6 us |
   | 128 | 9,248 | 14.4 us |
   | 256 | 17,440 | 15.8 us |
   | 512 | 33,824 | 16.8 us |

   Eight times the width costs 15% more time. If anything the accumulator should be
   *wider*, not narrower — a bigger net is nearly free here and strictly more accurate.
   The real speed lever is reducing the *number* of numpy calls per node (fuse the clip
   into the matmul, keep everything float32, avoid temporaries), not the layer sizes.

## 7. Which weights to ship

Three trained nets, all to this spec and interchangeable at the file level — only `L1`
and the training data differ. `reference_eval.py` reads any of them unchanged.

Scored on the same 1.5M-position holdout that no run trained on. "quiet" means the best
move is neither a capture nor a promotion — i.e. positions a quiescence search would
actually hand to a static eval.

| file | L1 | trained on | all positions | quiet positions |
|---|---|---|---|---|
| `nnue_v1.npz` | 256 | 403.6M | 0.01169 / 117.4 cp / 92.9% | 0.00966 / 106.2 cp / 94.6% |
| `nnue_v2_512.npz` | 512 | 403.6M | **0.01088 / 114.1 cp / 93.5%** | 0.00903 / 103.6 cp / 95.1% |
| `nnue_v3_512quiet.npz` | 512 | 307.4M quiet only | 0.01440 / 133.8 cp / 92.2% | **0.00842 / 100.1 cp / 95.5%** |

**Default: `nnue_v2_512.npz`.** It is the strongest net that is accurate *everywhere*.

**`nnue_v3_512quiet.npz` is better — but only if your search never asks it to judge a
position with a live capture.** It was trained only on positions whose best move is quiet
(23.8% of the database dropped), and it is clearly best on exactly that distribution
(0.00842 vs 0.00903, 3.5 cp better MAE). It is correspondingly much worse elsewhere.

Which one wins depends on a property of `fastsearch.py` that I cannot see from here:

- If eval is called **only at quiescence leaves**, after captures are resolved → use v3.
- If eval is also called at nodes with captures available — static null-move pruning,
  futility/razoring margins, move ordering by static eval — → use v2. v3 will be badly
  wrong at exactly those nodes, and pruning decisions amplify eval errors.

Most simple engines do the latter, which is why v2 is the default. If you know your eval
is quiescence-only, switch to v3 and expect a small gain.

## 8. Open items for the engine session

These were chosen by the trainer because the engine session was unreachable
(bridge ID `session_01SxBTWpVXqgWAcDLW4ggnji` returns HTTP 404). All are cheap to change
— the packed dataset stores *boards*, not features, so re-encoding is minutes:

- accumulator width 256 (per section 6, going *wider* is nearly free in numpy;
  512 is worth considering if the extra accuracy shows up in play)
- 32 king buckets via mirroring (could go to 64 unmirrored, or fewer coarse buckets)
- kings included (HalfKA) vs excluded (HalfKP)
- float vs quantised inference

Tell me which of these you need changed and I will retrain against it.
