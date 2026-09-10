# King buckets: the four-bucket experiment

The question this branch exists to answer: **is the 768-input feature set the thing holding the
learned evaluation back?** Cycle 7 in `docs/BENCH_LOG.md` measured the alternative hypothesis and
rejected it -- a net trained on 163M positions instead of 52M, with a better validation loss and a
more precise export, played at 44.8% over 144 games against the file that ships. More data on the
same features bought nothing twice in a row. What the 768 scheme cannot express is *where our own
king is*: a knight on f5 is one feature whether our king is castled on g1 or walking on e4.

This is deliberately **not** Stockfish's HalfKAv2_hm. It is the smallest change that tests the
king-conditioning hypothesis: four buckets instead of that scheme's 32, no king-file mirroring, no
change to the piece-square half of the feature, and no change to anything outside the model.

## The mapping

Every feature is `bucket * 768 + plane * 64 + square`, where `plane * 64 + square` is exactly the
768-input index the shipped net uses (`tools/nnue/features.py`) and `bucket` is a function of the
**perspective-oriented friendly king square**:

```
KING_BUCKET[square] = 2 * (rank >= 4) + (file >= 4)

        file a b c d | e f g h
  rank 8    2 2 2 2  | 3 3 3 3      2 = far half,  queenside      3 = far half,  kingside
  rank 7    2 2 2 2  | 3 3 3 3
  rank 6    2 2 2 2  | 3 3 3 3
  rank 5    2 2 2 2  | 3 3 3 3
  ---------------------------
  rank 4    0 0 0 0  | 1 1 1 1      0 = own half,  queenside      1 = own half,  kingside
  rank 3    0 0 0 0  | 1 1 1 1
  rank 2    0 0 0 0  | 1 1 1 1
  rank 1    0 0 0 0  | 1 1 1 1
```

`square` here is already perspective-oriented: `square ^ 56` has been applied for the Black
perspective, exactly as the 768 scheme does it, so "rank 1" means *that side's own back rank* and
"own half" means the half the king started in. **The mapping is therefore colour-symmetric by
construction**, and the mirror invariant the 768 scheme has survives unchanged:

    features(board) == features(board.mirror())

Each perspective is conditioned on **its own** king. The White-perspective accumulator is offset by
the bucket of White's king; the Black-perspective accumulator by the bucket of Black's king. That
is what makes a White king move leave the Black accumulator alone.

### Why these four

Two splits, each of which the 768 scheme cannot express and each of which changes what a piece is
worth:

- **File half** separates a kingside castle from a queenside one. Pawn-storm and shelter features
  are the classic reason engines bucket on the king at all.
- **Rank half** separates a sheltered king from one that has left home -- an advanced king is an
  endgame king, and the pieces around it are worth different things.

Four buckets of 32 squares each is also the largest bucketing that keeps every bucket
well-populated in training: a 32-bucket scheme divides the same corpus 32 ways, and the reason to
start here is that a bucket with too few positions learns noise. If this experiment pays, the
32-bucket version is the follow-up, not the starting point.

## The four places it has to be identical

A bucketed feature index computed differently in any of these is a net that trains on one thing and
plays another, and nothing else in the system would notice:

| where | what computes the index |
|---|---|
| training feature extraction | `tools/nnue/features.py`, `features()` -- the definition of record |
| weight export | `tools/nnue/export.py`, which lays out `l1_weight` as `[3072, hidden]` in that order |
| python reference inference | `tools/nnue/nnue_ref.py`, which indexes `l1_weight` by that number |
| numba runtime | `fastnnue.py`, `KING_OFFSET` + `FEATURE`, and the accumulator updates |

`tests/test_king_buckets.py` asserts the runtime's tables equal the offline ones square by square
and piece by piece, rather than trusting that two copies of a formula agree.

## Accumulator rules

The cost model is the point of the design. A 3072-row first layer is four times the memory, and if
every move paid for a rebuild the node rate would collapse.

- **Any move that is not a king move**: unchanged from the 768 runtime -- one row subtracted, one
  added, per perspective, plus the capture and the castling rook.
- **A king move that stays inside its bucket**: also unchanged. The king is a piece like any other
  in the 768 half of the feature, so its own row moves; the bucket offset does not.
- **A king move that crosses a bucket boundary**: every feature in *that perspective* changes, so
  that perspective is rebuilt from the board with the move applied. **The other perspective stays
  incremental** -- its king did not move, so its offset did not change.
- **Castling** is a king move of two files and is handled explicitly on both paths: `e1g1` crosses
  no boundary (bucket 1 to bucket 1) but `e1c1` does (bucket 1 to bucket 0), and the rook's own
  two rows move on the incremental path or arrive on the rebuilt one.

The perspective's king square is carried in the accumulator stack itself, in one extra int16 column
per perspective, so the search needs no new state and `fastsearch.py` is untouched.

## The warm start

The four bucket blocks are initialised as **four copies of the trained 768x256 first layer**, and
the second and third layers, both biases and all four quantisation scales are reused as they are.
Because every position activates features from exactly one block per perspective, and every block
holds the same rows, the warm-started net returns **the same integer as the 768 net on every
position**. `tests/test_king_buckets.py` proves that on 10,000 positions rather than asserting it.

This is a starting point for fine-tuning and nothing more. **A warm-started file is not a
king-relative model**: until it is fine-tuned on bucketed features its four blocks are identical,
which is another way of saying it has learned nothing about where the king is. Any strength claim
has to come from a bench after retraining.

The runtime reads both scheme versions: a version-1 (768-row) file is expanded into four identical
blocks at load time, which is the same warm start done in memory, and the init line says so.

## Go / no-go

In order, and each one gates the next:

1. **Exactness.** The tests in `tests/test_king_buckets.py` pass, including reference-versus-numba
   equality and 10,000 randomised make/unmake sequences.
2. **Speed.** The first layer is 3072 x 256 x 2 bytes = **1.5 MiB**, against 384 KiB for the 768
   net, and the platform core (Zen 4) has 1 MiB of L2. Evaluator latency and search node rate are
   measured against the unchanged 768 build *before* any strength testing. A node-rate loss large
   enough to cost a ply has to be paid back by the evaluation, and that is a much higher bar than
   parity.
3. **Strength, after retraining.** Only a fine-tuned net benched against `local-opponents/v4.1` at
   both controls decides this. The warm start scores exactly 50% by construction and proves only
   that the plumbing is right.

**Do not start the 32-bucket HalfKAv2_hm version unless step 3 produces a statistically credible
improvement that outweighs whatever step 2 costs.**
