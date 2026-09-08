# Version history

Every build that has been uploaded has a version. `vX.Y`: a new `X` is a new architecture or a
revamp of how the engine plays; a new `Y` is a change inside the same architecture. Each version
is an annotated git tag on the exact commit, a frozen copy under `local-opponents/vX.Y/` that
the bench plays against, and a row here saying what changed and why. Candidates that never
shipped keep their branch names; they are listed at the bottom and explained in
`docs/LOGBOOK.md` Part 2, rejections included.

The current baseline for the bench is the newest shipped version (v2.4 as of 2026-09-08 18:00).

## Shipped

| Version | Date | Commit | What changed | Why | Where it played |
|---|---|---|---|---|---|
| v1.0 | 2026-09-08 | 1151aa3 | Two-ply negamax with material and piece-square tables. No alpha-beta, no move ordering, no quiescence, ignores the clock. | First working submission; the reference everything after it had to beat. | Local only (`local-opponents/v1.0`). |
| v2.0 | 2026-09-08 | 6a731de (PR #2) | Iterative deepening, alpha-beta, quiescence on captures and queen promotions, MVV-LVA ordering, soft and hard time budgets with an abort path. | v1.0 spent 40 ms of a 120 s clock at depth 2; search depth is where the strength is. | Calibration upload (`submission-v1-search`). 92% vs minimax, 44% vs Sunfish locally. |
| v2.1 | 2026-09-08 | 4e43528 (PR #4) | Transposition table kept for the game, killer moves, history heuristic, repetition and fifty-move draws with contempt. | Ordering and memory: the same depth in far fewer nodes, and no more drawn wins by repetition. | Calibration upload (`submission-v2-memory`). 100% vs v1.0, 84% vs v2.0. |
| v2.2 | 2026-09-08 | a6c1fa6 (PR #6) | `fastboard.py` (numba board, move generation, Zobrist) shipped in the zip but not imported by `agent.py`. Plays identically to v2.1. | Groundwork for v3.0; shipping it early proved the packager and the platform accept it. | **Rated rounds 73 (won vs Castling, 1462) and 74 (lost vs Makina).** Depth 4 to 6, 28% and 57% of the clock unspent, 8 and 4 real blunders. |
| v2.3 | 2026-09-08 ~17:00 | 8cc4670 (PR #7) | The iteration gate in `_think`: a new depth starts while the soft budget is unspent and the whole projected iteration fits the hard budget. Hard budget and abort path unchanged. | v2.2 refused depth 5 with most of its budget unspent because the cost projection is capped at 8x and table-warmed early depths trip the cap; all four round 74 blunders were such moves. | **Uploaded for round 75.** Fast bench vs v2.2: 53.1%, 51.8%, 54.2% across three runs, no disqualifiers; deeper by about a third of a ply at 120 s. |
| v2.4 | 2026-09-08 (built ~17:35, awaiting upload) | 1077652 (PR #5) | Tapered evaluation (middlegame and endgame tables blended by material), mop-up for won endings, passed, isolated and doubled pawns, king shield. Evaluation also faster: median 62k nps vs 50k on the same positions. | v2.2 and v2.3 had material plus fixed piece-square tables and nothing else; the ladder above Sunfish is evaluation. Written by a teammate on `phase0/eval`, merged to prod without a bench; benched here before shipping. | Bench vs v2.3: **82.8%** (+25 =3 -4), Elo +273, interval +153 to +510, 32 games; 71.9% vs Sunfish; suite 3 of 12. Clean on disqualifiers, peak RSS 243 MB in a 120 s game. |

## Not shipped

| Branch | Commit | What | Why not |
|---|---|---|---|
| `cand/null-move` | 132a964 | Null move pruning | 53.6% vs v2.2 over 88 games, lower bound below zero. The only pruning candidate with a consistent positive sign; worth 300+ games. |
| `cand/lmr` | 5990f59 | Late move reductions | 48.4% vs v2.2. Needs principal variation search underneath it to pay. |
| `cand/futility` | 4d8f361 | Frontier futility pruning | 50.0% vs v2.2. No effect at 32 games. |
| `phase0/spend-the-clock` | b12c8f0 | Soft divisor 25 to 16 and table cap 1M | Ran the clock to 4.4 s in a 130-move game; the formula has no floor. Table cap part is fine (514 MB at 1M entries). |
| `time/growth-cap` | 7a5d60d | Projection growth cap 8 to 4 | 42.2% vs v2.2 at the fast control, 60.4% at the 45 s platform proxy (v2.3: 54.2%), 24 games each, overlapping intervals. v2.3 reached deeper at 120 s; a 100+ game tie-break at the proxy control is cycle 3's first job. |
| `time/reserve-floor` | ace9ebc | Soft budget from clock minus 10 s | 60.9% vs v2.2 on 32 games did not reproduce: 44.6% on 56. Does not add depth. Floors the clock at 7 s where v2.2 sinks to 4.7 s, so still a safety candidate for v2.4, measured at 120 s rather than on Elo. |

## Planned numbering

- v2.4 and on: further changes inside the python-chess engine (reserve floor, dynamic time
  extension, checks in quiescence, evaluation terms).
- v3.0: the search moved onto `fastboard.py`, the numba board. A different engine in speed and
  shape, so a new major.
