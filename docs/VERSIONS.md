# Version history

Every build that has been uploaded has a version. `vX.Y`: a new `X` is a new architecture or a
revamp of how the engine plays; a new `Y` is a change inside the same architecture. Each version
is an annotated git tag on the exact commit, a frozen copy under `local-opponents/vX.Y/` that
the bench plays against, and a row here saying what changed and why. Candidates that never
shipped keep their branch names; they are listed at the bottom and explained in
`docs/LOGBOOK.md` Part 2, rejections included.

The current baseline for the bench is the newest shipped version (v4.0 as of 2026-09-09 06:00).

## Shipped

| Version | Date | Commit | What changed | Why | Where it played |
|---|---|---|---|---|---|
| v1.0 | 2026-09-08 | 1151aa3 | Two-ply negamax with material and piece-square tables. No alpha-beta, no move ordering, no quiescence, ignores the clock. | First working submission; the reference everything after it had to beat. | Local only (`local-opponents/v1.0`). |
| v2.0 | 2026-09-08 | 6a731de (PR #2) | Iterative deepening, alpha-beta, quiescence on captures and queen promotions, MVV-LVA ordering, soft and hard time budgets with an abort path. | v1.0 spent 40 ms of a 120 s clock at depth 2; search depth is where the strength is. | Calibration upload (`submission-v1-search`). 92% vs minimax, 44% vs Sunfish locally. |
| v2.1 | 2026-09-08 | 4e43528 (PR #4) | Transposition table kept for the game, killer moves, history heuristic, repetition and fifty-move draws with contempt. | Ordering and memory: the same depth in far fewer nodes, and no more drawn wins by repetition. | Calibration upload (`submission-v2-memory`). 100% vs v1.0, 84% vs v2.0. |
| v2.2 | 2026-09-08 | a6c1fa6 (PR #6) | `fastboard.py` (numba board, move generation, Zobrist) shipped in the zip but not imported by `agent.py`. Plays identically to v2.1. | Groundwork for v3.0; shipping it early proved the packager and the platform accept it. | **Rated rounds 73 (won vs Castling, 1462) and 74 (lost vs Makina).** Depth 4 to 6, 28% and 57% of the clock unspent, 8 and 4 real blunders. |
| v2.3 | 2026-09-08 ~17:00 | 8cc4670 (PR #7) | The iteration gate in `_think`: a new depth starts while the soft budget is unspent and the whole projected iteration fits the hard budget. Hard budget and abort path unchanged. | v2.2 refused depth 5 with most of its budget unspent because the cost projection is capped at 8x and table-warmed early depths trip the cap; all four round 74 blunders were such moves. | **Uploaded for round 75.** Fast bench vs v2.2: 53.1%, 51.8%, 54.2% across three runs, no disqualifiers; deeper by about a third of a ply at 120 s. |
| v2.4 | 2026-09-08 (built ~17:35, awaiting upload) | 1077652 (PR #5) | Tapered evaluation (middlegame and endgame tables blended by material), mop-up for won endings, passed, isolated and doubled pawns, king shield. Evaluation also faster: median 62k nps vs 50k on the same positions. | v2.2 and v2.3 had material plus fixed piece-square tables and nothing else; the ladder above Sunfish is evaluation. Written by a teammate on `phase0/eval`, merged to prod without a bench; benched here before shipping. | Bench vs v2.3: **82.8%** (+25 =3 -4), Elo +273, interval +153 to +510, 32 games; 71.9% vs Sunfish; suite 3 of 12. Clean on disqualifiers, peak RSS 243 MB in a 120 s game. |
| v3.0 | 2026-09-08 late | PR #10 (`phase1/search`, 227d8ee) | Search and evaluation moved onto the numba board: `fasteval.py`, `fastsearch.py`; python-chess engine kept as the fallback. Same tree as v2.4, proven by equality tests. | Depth is speed: 1.2–3.3M nps vs 50–70k gives two to three more plies, which is where the rated-game blunders were. | Bench vs v2.4: **93.8%** (+30 =0 -2) at 10 s, 93.8% vs Sunfish, 100% vs minimax; 0 disqualifiers, worst move 1.29 s, peak RSS ~270 MB. Suite 3 of 12 at d7–9. |
| v3.1 | 2026-09-08 late | `search/clock-backstop` | A timer thread that expires the search at the hard deadline, alongside the node-counted clock read; the search functions release the interpreter lock. Same tree as v3.0. | The clock read is counted in nodes; one slow subtree near the deadline on the 0.4x platform is an overrun with no second stop. Carried over from PR #11, with the thread joined on exit so a late wake cannot expire the next move. | Disqualifier check vs v3.0: 16 fast games, 46.9%, 0 / 0 / 0 / 0. Backstop test: depth-40 searches with the clock read disabled stop within 5 ms of the deadline. |
| v3.2 | 2026-09-09 | PR #13 (`book/opening`) | A polyglot opening book, `weights/book.bin` (24,479 entries from 1.13M over-the-board master games, built by `tools/book/build.py`), played up to ply 20 and committed to both engines' history like a searched move. Search unchanged. | The first searched moves at 120 s cost 4 s each on positions master practice already answers; where the book has coverage it hands the clock to the middlegame. Coverage is thin on the ladder's curated openings (four of the eight samples have no master games), so it fires in a minority of rated games. | Disqualifier check vs v3.1: 32 fast games, 42.2% (interval 26 to 58%, includes 50), 0 / 0 / 0 / 0. 120 s from the Sveshnikov: one book move, 85.8 s vs 75.4 s on the clock after move 10. |
| v4.0 | 2026-09-09 | `nnue/v4.0` on PR #14 | The learned evaluation on: `weights/nnue.npz` (768-256-32-1, trained on the M5 on 52M lichess-evaluated positions, `tools/nnue/runs/h256-52m-e60.md`), scored as the average of the hand evaluation and the network at every leaf, hand tables past the 3-man line, `fastnnue.py` compiled with two side-relative accumulators per ply. Search unchanged. | Three extra plies solved no blunders; the evaluation was the ceiling. A residual net measured at parity, the plain net alone at -106, and the average at +190 to +240: the hand evaluation supplies material sanity, the net supplies what it learned. | Bench vs v3.2: 64 games **75.0%** (+191) by the doer, **96 games 80.2%** (+243, +177 to +329) re-run from scratch, 45 s proxy 24 games **85.4%** (+307); two 120 s games both won, depth 6.7 / 6.2 vs 8.0 on the other side; 0 disqualifiers anywhere, worst overshoot 1 ms, node rate 1.3M vs 2.4M. |

## Not shipped

| Branch | Commit | What | Why not |
|---|---|---|---|
| `eval/mobility` | PR #19, v4.1 candidate, not proven | Knight, bishop, rook and queen mobility in both evaluations: squares attacked, less our own men and the squares enemy pawns cover, scored against a typical count per piece. | 53.1% over 64 games vs `prod` df8f1fb, Elo +22, interval **-54 to +100**, no disqualifiers: right sign, lower bound below zero. Cheap enough to keep arguing about — 12% of the node rate for a 7% smaller tree, a tenth of a ply — and exact against `agent.py` on 10,000 positions. The 45 s proxy is the second control; a fast-control rerun cannot settle it, since 8 openings by 2 colours means 64 games is 32 unique pairings. |
| `eval/kpk-rook-pawn-draw` | PR #17, merged into prod 2026-09-09, part of v4.1 | King and rook pawn against a bare king scores 0 when the defender is in front, in both evaluations. | Not a strength change the bench can see: fired in 3 of 96 games (all drawn, all correctly). 45.8%, -88 to +28, no disqualifiers. Decided on the position tests and the round 82 shuffle it removes. |
| `cand/null-move` | 132a964 | Null move pruning | 53.6% vs v2.2 over 88 games, lower bound below zero. The only pruning candidate with a consistent positive sign; worth 300+ games. |
| `cand/lmr` | 5990f59 | Late move reductions | 48.4% vs v2.2. Needs principal variation search underneath it to pay. |
| `cand/futility` | 4d8f361 | Frontier futility pruning | 50.0% vs v2.2. No effect at 32 games. |
| `phase0/spend-the-clock` | b12c8f0 | Soft divisor 25 to 16 and table cap 1M | Ran the clock to 4.4 s in a 130-move game; the formula has no floor. Table cap part is fine (514 MB at 1M entries). |
| `time/growth-cap` | 7a5d60d | Projection growth cap 8 to 4 | 42.2% vs v2.2 at the fast control, 60.4% at the 45 s platform proxy (v2.3: 54.2%), 24 games each, overlapping intervals. v2.3 reached deeper at 120 s; a 100+ game tie-break at the proxy control is cycle 3's first job. |
| `time/reserve-floor` | ace9ebc | Soft budget from clock minus 10 s | 60.9% vs v2.2 on 32 games did not reproduce: 44.6% on 56. Does not add depth. Floors the clock at 7 s where v2.2 sinks to 4.7 s, so still a safety candidate for v2.4, measured at 120 s rather than on Elo. |
| `time/unstable-extend` | de956ad | Extend the soft budget once when the root is unstable | 51.6% vs v2.4 fast, 50.0% at the proxy, same depth at 120 s. No effect. |
| `time/growth-cap-4` | e27abf8 | Growth cap 4 on top of v2.3's gate | 48.4% fast, 52.1% proxy; 5% more time for equal depth, clock to 9.3 s. |
| `search/qs-checks` | bb0690e | Quiet checks at the first quiescence ply | 40.6% vs v2.4: a third of the speed on tactical positions, sharper and weaker. Revisit on the compiled board. |

## Planned numbering

- v2.4 and on: further changes inside the python-chess engine (reserve floor, dynamic time
  extension, checks in quiescence, evaluation terms).
- v3.0: the search moved onto `fastboard.py`, the numba board. A different engine in speed and
  shape, so a new major.
