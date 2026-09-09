# Bench log

Every improvement cycle appends here. The frozen reference is the shipped build under its version
name (`docs/VERSIONS.md`): `local-opponents/v2.2` (a6c1fa6) for cycles 1 and 2, the build that
played rounds 73 and 74; `local-opponents/v2.3` (8cc4670) from the 17:00 upload on. Never edit
a frozen copy; freeze the next version beside it.

## How a run is scored

`uv run python -m harness.bench --candidate <dir> --jobs 4` plays the gauntlet: 32 games
against the baseline, 16 against Sunfish, 16 against `baselines/minimax`, paired openings with
colours swapped, at the arena's 10 s + 0.1 s. It reuses `harness.arena`'s pairing, its Elo
conversion and its 95% interval, and adds the four disqualifiers:

- **ill** illegal moves by the candidate
- **exc** exceptions: crashes, failed inits, and tracebacks the agent printed while swallowing
  an exception inside `get_move` (only the first and last 4 KB of the log survive)
- **tmo** timeouts: flag falls by the candidate
- **over** moves that spent more than a quarter of the clock the candidate had, double the
  engine's own hard divisor, so a clock-check slice overshoot of tens of milliseconds does
  not count and a real time-management break does

Any nonzero count disqualifies regardless of Elo. "worst" is the candidate's slowest move.

Openings are the harness's eight plus twenty mainstream lines in `harness/bench.py`, because
two deterministic engines replay the same game from the same opening. Games run four at a
time on a 10-core M4 (the sanity run below shows no timing distortion at that load).
"overall" pools every game, so it is an Elo against the gauntlet mix, not against any one
opponent.

## Setup, 2026-09-08

Sanity: the baseline against itself must come back near zero with the interval straddling it.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst |
|---|---|---|---|---|---|---|---|---|---|
| sanity | baseline | 10s+0.1s | 24 | +8 =8 -8 | 50.0% | +0 | -121 to +121 | 0 / 0 / 0 / 0 | 1.31s |
| sanity | sunfish | 10s+0.1s | 4 | +0 =2 -2 | 25.0% | -191 | -inf to +23 | 0 / 0 / 0 / 0 | 1.26s |
| sanity | minimax | 10s+0.1s | 4 | +4 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.22s |

Passed: exactly 50%, no disqualifiers, 3.7 minutes for 32 games. The colour pairs diverged
(the Scotch pair went 2-0, the French Classical pair 1-1), so the fast control does not
replay games and the counts are real. Worst move 1.31 s at a 10 s clock is the engine's
1.25 s hard budget plus one clock-check slice.

## Cycle 1, 2026-09-08, forward pruning

Three candidates, one change each, every branch off `main` at `a6c1fa6`:

- `cand/null-move` (132a964): null move pruning, R=2 from depth 2, skipped in check, with
  only king and pawns, after another pass, and around mate scores.
- `cand/lmr` (5990f59): late move reductions, quiet non-checking non-killer moves after the
  first three at depth 3 and up searched a ply shallower, re-searched at full depth on beating
  alpha.
- `cand/futility` (4d8f361): frontier futility, at depth 1 quiet non-checking moves skipped
  when static evaluation plus 150 cp cannot reach alpha.

Gauntlet at 10 s + 0.1 s, 64 games each (32 baseline, 16 Sunfish, 16 minimax), about 7
minutes per candidate with four games at a time.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst |
|---|---|---|---|---|---|---|---|---|---|
| null-move | baseline | 10s+0.1s | 32 | +12 =10 -10 | 53.1% | +22 | -81 to +128 | 0 / 0 / 0 / 0 | 1.30s |
| null-move | sunfish | 10s+0.1s | 16 | +6 =9 -1 | 65.6% | +112 | +6 to +245 | 0 / 0 / 0 / 0 | 1.29s |
| null-move | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.25s |
| null-move | overall | 10s+0.1s | 64 | +34 =19 -11 | 68.0% | +131 | +60 to +213 | 0 / 0 / 0 / 0 | 1.30s |
| lmr | baseline | 10s+0.1s | 32 | +10 =11 -11 | 48.4% | -11 | -114 to +90 | 0 / 0 / 0 / 0 | 1.29s |
| lmr | sunfish | 10s+0.1s | 16 | +4 =8 -4 | 50.0% | +0 | -130 to +130 | 0 / 0 / 0 / 0 | 1.30s |
| lmr | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.29s |
| lmr | overall | 10s+0.1s | 64 | +30 =19 -15 | 61.7% | +83 | +12 to +161 | 0 / 0 / 0 / 0 | 1.30s |
| futility | baseline | 10s+0.1s | 32 | +13 =6 -13 | 50.0% | +0 | -114 to +114 | 0 / 0 / 0 / 0 | 1.31s |
| futility | sunfish | 10s+0.1s | 16 | +8 =5 -3 | 65.6% | +112 | -27 to +302 | 0 / 0 / 0 / 0 | 1.34s |
| futility | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.36s |
| futility | overall | 10s+0.1s | 64 | +37 =11 -16 | 66.4% | +118 | +41 to +209 | 0 / 0 / 0 / 0 | 1.36s |

No disqualifiers anywhere; every worst move is the 1.25 s hard budget plus one clock-check
slice. None of the three moved the baseline column: the intervals are about ±120 Elo wide at
32 games, and a pruning change in this engine is worth tens of Elo, not hundreds. The
"overall" rows are not evidence of a gain, since the baseline itself sweeps minimax; only
the baseline column compares like with like. Null move was best of three on it and tied
futility on Sunfish, so it went to the re-run.

### Re-run of the best, and a baseline reference

Selecting the best of three noisy measurements is biased upward, so null move was re-run from
scratch against the baseline over all 56 paired openings. A baseline-only run against Sunfish
and minimax was added so the Sunfish column has a same-bench reference.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst |
|---|---|---|---|---|---|---|---|---|---|
| null-move-rerun | baseline | 10s+0.1s | 56 | +16 =28 -12 | 53.6% | +25 | -40 to +91 | 0 / 0 / 0 / 0 | 1.38s |
| baseline-ref | sunfish | 10s+0.1s | 16 | +9 =7 -0 | 78.1% | +221 | +112 to +395 | 0 / 0 / 0 / 0 | 1.29s |
| baseline-ref | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.26s |

The re-run reproduced the direction (53.1% became 53.6%) but not a gain: the lower bound is
-40. Pooling both null-move runs, 88 games at +28 =38 -22, is 53.4%, Elo +24 with an interval
of about -31 to +80, and pooling flatters it since the first run was selected as the maximum.
The step 7 gate (lower bound above zero) was not met, so no 120 s confirmation was run.

**The Sunfish column cannot rank candidates.** The same baseline scored 25% on 4 Sunfish games
in the sanity run and 78.1% on these 16. The Winawer game as white diverged at Sunfish's 10th
move between the two runs: Sunfish's deepening is clock-driven, so its moves vary with timing
jitter, and 16 games is a ±13 to ±19% interval. All three candidates scored under the
baseline's 78.1% on Sunfish, and that means nothing either.

**Verdict: ship nothing.** No candidate has a lower bound above zero against the baseline. Null
move is the only one with a consistent positive direction over 88 games, and at 4 games in
parallel the bench plays about 450 games an hour, so a next cycle spent on 300 to 400 games of
null move against the baseline, plus the 120 s confirmation, would settle it. Nothing was
merged to `main` and nothing was uploaded. Branches: `cand/null-move` (132a964), `cand/lmr`
(5990f59), `cand/futility` (4d8f361), tooling on `tooling/bench`. PGNs under `../pgn/bench-*`.

## Phase 0, 2026-09-08: spend the clock (phase0/spend-the-clock, b12c8f0)

Not a bench run: self-play verification at 120 s + 0.5 s of SOFT_DIVISOR 16 and
TABLE_MAX_ENTRIES 1M. 20 plies clean. Full game 130 moves, draw by repetition, 0 exceptions,
0 illegal, slowest 10.5 s vs hard 11.9 s, 22 moves 3 to 49 ms over hard (clock-check slice),
peak RSS 514 MB at 996k table entries. Clock: 12.8 s after move 47, 8.7 s after move 60,
4.4 s minimum. Failed the 10 s floor; not tagged, not shipped. The bench now reports peak RSS.

## Cycle 2, 2026-09-08, time management. v2.3 shipped from `time/gate-hard`.

Baseline is v2.2 (a6c1fa6) throughout this cycle. Three candidates off `prod`, one change
each: `time/gate-hard` (8cc4670, gate on the hard budget, became **v2.3**), `time/growth-cap`
(7a5d60d, projection growth cap 8 to 4), `time/reserve-floor` (ace9ebc, soft budget from
clock minus 10 s). Fast gauntlet 64 games each; from-scratch re-runs of 56 games; the 45 s +
0.2 s rows are the platform proxy (the match machine runs at 0.38x our speed, so 45 s here is
120 s there in nodes per game). Peak RSS is now reported.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| gate-hard | baseline | 10s+0.1s | 32 | +12 =10 -10 | 53.1% | +22 | -81 to +128 | 0 / 0 / 0 / 0 | 1.27s | 83 MB |
| gate-hard | sunfish | 10s+0.1s | 16 | +5 =5 -6 | 46.9% | -22 | -182 to +129 | 0 / 0 / 0 / 0 | 1.26s | 73 MB |
| gate-hard | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.25s | 49 MB |
| growth-cap | baseline | 10s+0.1s | 32 | +9 =9 -14 | 42.2% | -55 | -168 to +48 | 0 / 0 / 0 / 0 | 1.28s | 82 MB |
| growth-cap | sunfish | 10s+0.1s | 16 | +6 =7 -3 | 59.4% | +66 | -63 to +217 | 0 / 0 / 0 / 0 | 1.27s | 74 MB |
| growth-cap | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.28s | 47 MB |
| reserve-floor | baseline | 10s+0.1s | 32 | +13 =13 -6 | 60.9% | +77 | -14 to +181 | 0 / 0 / 0 / 0 | 1.26s | 140 MB |
| reserve-floor | sunfish | 10s+0.1s | 16 | +3 =7 -6 | 40.6% | -66 | -217 to +63 | 0 / 0 / 0 / 0 | 1.27s | 94 MB |
| reserve-floor | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.16s | 49 MB |
| gate-hard-rerun | baseline | 10s+0.1s | 56 | +18 =22 -16 | 51.8% | +12 | -60 to +86 | 0 / 0 / 0 / 0 | 1.27s | 115 MB |
| gate-hard-proxy45 | baseline | 45s+0.2s | 24 | +7 =12 -5 | 54.2% | +29 | -72 to +135 | 0 / 0 / 0 / 0 | 5.64s | 277 MB |
| growth-cap-proxy45 | baseline | 45s+0.2s | 24 | +11 =7 -6 | 60.4% | +73 | -44 to +211 | 0 / 0 / 0 / 0 | 5.64s | 256 MB |

120 s + 0.5 s games against v2.2, one per colour, first 40 moves: gate-hard spent 131 to 136%
of its soft budget at depth 6.33 / 5.83 vs v2.2's 6.00 / 5.55 on the other side of the same
boards (1 loss, 1 draw); growth-cap 100 to 120% at 5.47 / 6.08 vs 5.78 / 5.92 (1 win, 1 draw);
reserve-floor 84 to 86% at 5.30 / 5.47 vs 5.80 / 4.97 (1 win, 1 loss). Clock floors in long
games: gate-hard 5.1 s (v2.2 4.8 s on the same board), reserve-floor 7.0 s (v2.2 4.7 s). No
exceptions, illegal moves or flags in any of the six games; the largest hard-budget overshoot
was 32 ms, the clock-check slice.

**Shipped v2.3 (gate-hard) on the depth evidence, not on Elo**: three runs vs v2.2 at 53.1%,
51.8% and 54.2% all point the same way and none clears zero on the lower bound. Growth-cap's
proxy result (60.4%) is above v2.3's (54.2%) with overlapping intervals; a 100+ game match
between them at the proxy control is the cycle 3 tie-break. Reserve-floor does not add depth
and is the v2.4 companion; its re-run row follows.

Reserve-floor re-run from scratch:

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| reserve-floor-rerun | baseline | 10s+0.1s | 56 | +14 =22 -20 | 44.6% | -37 | -112 to +34 | 0 / 0 / 0 / 0 | 1.27s | 101 MB |

The 60.9% did not reproduce: 44.6% over 56 games. Best-of-three selection at 32 games is worth
about a hundred Elo of optimism, which is why the re-run step exists. Cycle 2 closed.

## Cycle 3 opening, 2026-09-08: the merged prod (v2.3 + PR #5 evaluation) against v2.3

Prod moved under cycle 2: a teammate merged the evaluation PR on top of v2.3. Benched before
anything else; baseline v2.3.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| prod-eval | baseline | 10s+0.1s | 32 | +25 =3 -4 | 82.8% | +273 | +153 to +510 | 0 / 0 / 0 / 0 | 1.27s | 74 MB |
| prod-eval | sunfish | 10s+0.1s | 16 | +10 =3 -3 | 71.9% | +163 | +13 to +420 | 0 / 0 / 0 / 0 | 1.27s | 59 MB |
| prod-eval | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.26s | 42 MB |

Same eight opening positions, one move each at a 75 s clock: v2.3 mean depth 5.25 at median
50k nps, prod 5.38 at 62k nps. 120 s self-play: 51 moves, clean, peak RSS 243 MB, clock 17 to
21 s at the end. Regression suite 3 of 12 (v2.2 also 3 of 12, a different three). Built as
`submission-v2.4.zip`; the 45 s proxy match follows.

Proxy control, 24 games vs v2.3:

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| prod-eval-proxy45 | baseline | 45s+0.2s | 24 | +17 =3 -4 | 77.1% | +211 | +81 to +441 | 0 / 0 / 0 / 0 | 5.65s | 247 MB |

The gain holds at platform-like depth. v2.4 is the bench baseline from here.

## Cycle 3, 2026-09-08, dynamic time and checks in quiescence. Nothing shipped.

Baseline v2.4 (1077652). Three candidates off `prod`, one change each: `time/unstable-extend`
(de956ad), `time/growth-cap-4` (e27abf8), `search/qs-checks` (bb0690e). Fast gauntlet 64
games each; 45 s + 0.2 s platform proxy for the two timing candidates; two 120 s games each
against v2.4 for the timing candidates; regression suite for all three (3 of 12 each, run
before round 75's positions were added).

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| search-qs-checks | baseline | 10s+0.1s | 32 | +11 =4 -17 | 40.6% | -66 | -196 to +47 | 0 / 0 / 0 / 0 | 1.29s | 46 MB |
| search-qs-checks | sunfish | 10s+0.1s | 16 | +11 =2 -3 | 75.0% | +191 | +35 to +512 | 0 / 0 / 0 / 0 | 1.29s | 46 MB |
| search-qs-checks | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.28s | 31 MB |
| time-unstable-extend | baseline | 10s+0.1s | 32 | +14 =5 -13 | 51.6% | +11 | -104 to +129 | 0 / 0 / 0 / 0 | 1.27s | 54 MB |
| time-unstable-extend | sunfish | 10s+0.1s | 16 | +8 =6 -2 | 68.8% | +137 | +8 to +321 | 0 / 0 / 0 / 0 | 1.26s | 68 MB |
| time-unstable-extend | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.26s | 39 MB |
| time-growth-cap-4 | baseline | 10s+0.1s | 32 | +13 =5 -14 | 48.4% | -11 | -129 to +104 | 0 / 0 / 0 / 0 | 1.27s | 92 MB |
| time-growth-cap-4 | sunfish | 10s+0.1s | 16 | +5 =7 -4 | 53.1% | +22 | -114 to +164 | 0 / 0 / 0 / 0 | 1.26s | 60 MB |
| time-growth-cap-4 | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.26s | 45 MB |
| time-unstable-extend-proxy45 | baseline | 45s+0.2s | 24 | +11 =2 -11 | 50.0% | +0 | -144 to +144 | 0 / 0 / 0 / 0 | 5.64s | 177 MB |
| time-growth-cap-4-proxy45 | baseline | 45s+0.2s | 24 | +10 =5 -9 | 52.1% | +14 | -116 to +149 | 0 / 0 / 0 / 0 | 5.64s | 192 MB |

120 s games, first 40 moves, candidate vs v2.4 on the other side of the same board:
unstable-extend depth 5.00 / 5.95 vs 5.90 / 4.95 (the colour swap accounts for the whole
difference), spend 81 s / 96 s vs 98 s / 82 s, 1 loss 1 win; growth-cap-4 depth 5.43 / 5.92 vs
5.96 / 6.05, spend 109 s / 130 s vs 104 s / 123 s, minimum clock 9.3 s, 2 wins. No
disqualifiers anywhere. No re-run: the best baseline-column score is 51.6%, nothing to
reproduce. Cycle 3 closed.

## Phase 1, 2026-09-08: the numba engine (phase1/search)

`fasteval.py` and `fastsearch.py` are `agent.py`'s evaluation and search compiled by numba over
the `fastboard` mailbox. `get_move` runs them, validates the move against
`chess.Board(fen).legal_moves`, and falls back to `_think_python` on any exception or illegal
move. It is a port, not a redesign, and it is held to that: `tests/test_fasteval.py` proves the
two evaluations return the same integer on 10,000 positions, and `tests/test_fastsearch.py`
proves the two searches return the same root score at the same depth on 188 fixed-depth
searches at depths 2 to 5, at 1.03x the nodes.

Node rate 1.2 to 3.3 M/s against the python-chess engine's 49 to 69 k/s on the same positions,
25 to 30x, measured under a competing benchmark at load 4 to 6. Depth at 10 s + 0.1 s: median 6,
range 6 to 7, against the python engine's median 4, range 4 to 5. At 120 s + 0.5 s: median 8,
range 7 to 8, against median 6, range 4 to 6. Two plies at both controls. Import with warm-up
3.7 to 4.4 s, RSS after import 215 to 225 MB, peak RSS 224 to 226 MB across every gauntlet run
here. The `make zip` smoke game has been seen anywhere between 202 and 244 MB, because numba's
compilation allocations live in the same process and are not returned tidily; the search's own
footprint is what does not move, since the table is a fixed 33 MB array rather than a dict that
grows with the game.

Opponents: `../eval-agent` is prod with the evaluation (v2.4), `../memory-agent` is prod before
it (v2.1). Two games at a time, load 4 to 6 from another benchmark on the same machine.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| v3.0 shipped | eval-agent v2.4 | 10s+0.1s | 32 | +29 =2 -1 | 93.8% | +470 | +322 to +inf | 0 / 0 / 0 / 0 | 1.25s | 225 MB |
| v3.0 shipped | sunfish | 10s+0.1s | 16 | +14 =2 -0 | 93.8% | +470 | +307 to +inf | 0 / 0 / 0 / 0 | 1.25s | 225 MB |
| v3.0 shipped | memory-agent v2.1 | 10s+0.1s | 32 | +30 =0 -2 | 93.8% | +470 | +304 to +inf | 0 / 0 / 0 / 0 | 1.25s | 226 MB |
| v3.0 null move on | eval-agent v2.4 | 10s+0.1s | 32 | +31 =1 -0 | 98.4% | +720 | +526 to +inf | 0 / 0 / 0 / 0 | 1.25s | 226 MB |
| v3.0 null move on | sunfish | 10s+0.1s | 16 | +15 =1 -0 | 96.9% | +597 | +397 to +inf | 0 / 0 / 0 / 0 | 1.25s | 226 MB |
| v3.0 null move on | v3.0 null move off | 10s+0.1s | 32 | +13 =7 -12 | 51.6% | +11 | -100 to +124 | 0 / 0 / 0 / 0 | 1.26s | 225 MB |

| v3.0 shipped | sunfish | 120s+0.5s | 2 | +2 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 13.89s | 224 MB |
| v3.0 after audit fixes | eval-agent v2.4 | 10s+0.1s | 32 | +31 =1 -0 | 98.4% | +720 | +526 to +inf | 0 / 0 / 0 / 0 | 1.25s | 226 MB |

The last row is the confirmation run after the audit fixes. It is nominally better than the
93.8% above it, but the two intervals overlap almost entirely and 32 games cannot tell them
apart; what it establishes is that nothing regressed, not that anything improved. The fix that
matters cannot show up here at all: the fallback never fired in any of these games, and the
history loss it caused only bites in a game where it does.

Also 16 games at 2 s + 0.1 s and 60 at 3 s + 0.1 s against `baselines/random`: 76 wins, 76 by
checkmate, no failed terminations.

At the real control the two games against Sunfish, one each colour, were both won with no flag
and no fallback: `exceptions 0` is the fallback count, since `_think_fast` raises whenever the
numba engine returns anything `chess.Board(fen).legal_moves` does not contain. The slowest move
was 13.89 s against a hard budget of 13.89 s at that clock, so the deadline binds to within a
clock-check slice, which is what reading the real clock through `objmode` every 1024 nodes
buys over estimating a node budget.

**Null-move pruning is off in what ships, and the two gauntlet rows above are why it is a close
call rather than a decision.** Head to head against exactly this engine with it on, which is the
sensitive comparison because everything else is identical, 32 games came back 51.6%, Elo +11,
interval -100 to +124. The gauntlet rows differ by one loss and one draw out of 32, which is
inside that interval. It also solved 2 of the 12 regression positions against 3 with it off. So
the measurement says nothing, and the tie-break is that null move is unsound about quiet lines
by construction while with it off this search returns `agent.py`'s score at every depth, which
is the property the whole port is verified against. The code and the constant
`NULL_MOVE_PRUNING` stay; measure it again once there is a PVS to reduce around, and with the
300 to 400 games cycle 1 said it would take.

**The regression suite did not improve: 3/12 for the python engine, 3/12 for the numba engine
with null move off, 2/12 with it on.** Depth reached went from d5-d6 to d7-d9, so the extra
plies are real and they are not what those twelve positions need. Read that as evidence about
the evaluation rather than the search: `r73 m11` and `r73 m40` swap places between the two
engines, and the rest are missed at every depth either engine reaches.

## v3.1, 2026-09-08 late: the timer-thread backstop (search/clock-backstop)

The one idea carried over from the parallel v3.0 port in PR #11. v3.0 reads the clock every
1024 nodes through `objmode`, so it stops as regularly as nodes come; a thread now sleeps until
the hard deadline and sets `STATS[EXPIRED]`, which every node reads before the clock check, and
the search functions release the interpreter lock so the thread can run. The search tree is
unchanged, so the fast bench is a disqualifier check, not an Elo claim; the test that matters is
`tests.test_fastsearch`'s `backstop`, which disables the clock read and asks for depth 40: six
searches all stopped on the thread within 5 ms of the deadline.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| backstop | v3.0 | 10s+0.1s | 16 | +6 =3 -7 | 46.9% | -22 | -199 to +144 | 0 / 0 / 0 / 0 | 1.25s | 226 MB |

120 s + 0.5 s against v3.0, one game per colour: won as White by checkmate (54 moves), lost as
Black by checkmate (29 moves); the tree is identical, so the split is the coin toss it looks
like. Depth over the first 40 moves 8.55 / 8.83 against v3.0's 8.70 / 7.62 on the other side
of the same boards, so the flag read per node and `nogil` cost nothing measurable. Worst
overshoot of the hard budget 1 ms on both sides (the clock-check slice, as before); slowest
move 13.9 s at a 13.9 s hard budget; clock minima 15.5 s and 28.4 s. The thread never had to
fire in play, which is the expected case; the test is where it is exercised.


## Book, 2026-09-09: a polyglot opening book (book/opening)

`weights/book.bin`, 24,479 entries over 15,530 positions, 391,664 bytes, read by
`chess.polyglot` at import and consulted before the search up to ply 20. The site's rules were
re-read for this: "Opening books and endgame tablebases are permitted as shipped data, and
`chess.polyglot` and `chess.syzygy` are in the base image", inside "model weights, opening
books and any other files, total <= 50 MB unzipped", and no limit on book moves or plies.

**The Lichess masters explorer is gone.** `https://explorer.lichess.ovh/masters` and
`/lichess` answer `401 Authorization Required` (nginx, no `WWW-Authenticate`) to every request,
from this machine and from a second network, with and without a token, on both
`explorer.lichess.ovh` and `explorer.lichess.org`; `/master` and
`lichess.org/api/opening-explorer` are 404, and `lichess.org` itself answers 200, so this is
that service and not our network. The corpus instead is PGN Mentor's 233 opening collections,
**1,133,198 over-the-board master games**, 879 MB of zips cached under `tools/book/cache/`
(gitignored). Provenance is the reason it is PGN files and not an engine: every move in the
book was played by a human master in a real game.

**Four of the eight sample openings occur in no master game at all.** Games in the corpus that
pass through each root, which is what the book's depth per opening follows from:

| root | master games through it | plies of book |
|---|---|---|
| standard start | 1,133,202 | 16 |
| English Opening | 3 | 0 |
| French Winawer | 127 | 3 |
| Petroff Defence | 0 | 0 |
| Scotch Game | 0 | 0 |
| Grunfeld Defence | 651 | 9 |
| French Classical | 0 | 0 |
| Sicilian Closed | 44 | 5 |
| Sicilian Sveshnikov | 1,373 | 9 |

The ladder's curated positions are picked to be close to level, not to be theory, so a book of
human games cannot answer them, and no threshold changes that: at `--min-games 100` (the number
this was briefed with) the book is 6,600 entries and answers only Grunfeld and Sveshnikov; at
12 it is 24,479 entries and answers four roots; at 4 it is 46,109 entries and 738 KB and still
leaves the same four openings at zero. 12 ships. **Expect the book to fire in a minority of
rated games** — it fires from the standard position, which rated games never use.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| book | v3.1 | 10s+0.1s | 32 | +11 =5 -16 | 42.2% | -55 | -179 to +57 | 0 / 0 / 0 / 0 | 1.26s | 227 MB |

The score interval is 26.3% to 58.1%, so it includes 50%, and the disqualifiers are all zero,
which is the pass criterion: a book that changes at most a handful of opening moves cannot show
Elo over 32 games. Nine of the sixteen bench openings are the ladder's, where the book is
mostly silent.

Lookup cost, measured over the nine roots and an off-book middlegame, worst 0.096 ms against a
5 ms budget. The book rebuilds byte-identically from the cache
(`8e8bf2b0...`), and `tools/book/build.py` reads every entry back through
`chess.polyglot.open_reader` and proves it legal before the file is accepted.

**A defect this found in `harness/readlog.py`, which this branch may not edit:** the numba
search's log line has not matched `OUTPUT_LINE` since v3.0. It prints `tt 31%` where the reader
wants an integer and an extra `null 0` field before `contempt`, so the reader files every
searched move under "other output" and reports a total log gap on every v3.0+ game. The book's
line is written in the shape the reader still parses (`d0 move <uci> nodes 0 <n>ms soft 0
hard 0 clock <ms> tt <n> cut 0 contempt +0 peakrss <n>MB book <k> of <total>`), and
`tests/test_book.py` asserts that through `readlog.parse`. Worth a one-line fix on a harness
branch.

**120 s + 0.5 s against v3.1, one game per colour from the harness's first opening (English),
plus one from the Sveshnikov where the book has something to say.** Everything but the book is
byte-identical to v3.1, so where the book does not fire the two engines replay the same game.

| game | our colour | book moves | our clock after move 10 | their clock after move 10 | first searched depth | result |
|---|---|---|---|---|---|---|
| English | white | 0 | 75.3 s | 74.4 s | 6 | won by checkmate |
| English | black | 0 | 74.5 s | 75.4 s | 6 | lost by checkmate |
| Sveshnikov | black | 1 (`c6b8`, 2 entries, 1,052 master games) | **85.8 s** | 75.4 s | 6 | won by checkmate |

The two English games are the same board from both sides, which is what identical engines do:
the book is silent there (3 master games through that position), so nothing differed. The
Sveshnikov game is the measurement that matters: one book move at ply 15 cost 0 ms instead of
about 11 s, and the clock after move 10 is 85.8 s against the opponent's 75.4 s. That is the
whole mechanism - the book buys clock in the openings it knows, and is silent in the rest.
