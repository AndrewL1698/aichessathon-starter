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

## v3.0, 2026-09-08: the compiled search (`v3/compiled-search`)

Baseline v2.4 (1077652, `local-opponents/v2.4`). One candidate: `fastsearch.py` and the
wrapper in `agent.py`, described in `docs/LOGBOOK.md`. The correctness gates before any game
was played: compiled evaluation equal to `agent.evaluate` on 10,045 positions; compiled search
equal in score to `agent._root` at depth 3 on 334 positions and depth 4 on 35 with the table
and killers off, every move difference a verified tie; `tests.test_fastboard` clean.

Speed and depth, one move per harness opening at a 75 s clock (soft 3.4 s, hard 9.4 s), each
engine in its own process on an idle machine:

| opening | v3.0 depth | v3.0 nps | v3.0 ms | v2.4 depth | v2.4 nps | v2.4 ms |
|---|---|---|---|---|---|---|
| English Opening | 8 | 2,684,930 | 3153 | 6 | 75,951 | 4361 |
| French Winawer | 8 | 2,450,331 | 5682 | 6 | 73,025 | 7034 |
| Petroff Defence | 8 | 2,525,329 | 2868 | 6 | 84,436 | 4418 |
| Scotch Game | 8 | 2,536,612 | 5209 | 5 | 96,017 | 2324 |
| Grunfeld Defence | 8 | 2,069,009 | 8097 | 5 partial | 81,749 | 9382 |
| French Classical | 7 partial | 2,363,860 | 9374 | 5 partial | 78,616 | 9378 |
| Sicilian Closed | 8 | 2,391,859 | 5470 | 6 | 74,461 | 7160 |
| Sicilian Sveshnikov | 8 partial | 2,161,653 | 9289 | 5 | 80,005 | 1430 |
| **mean / median** | **7.88** | **2,421,095** | 6143 | **5.50** | **79,310** | 5686 |

The same eight in one process through `tests.test_fastsearch --speed`, table on, fresh per
position: median 2,397k nps at depth 6 against 70k at depth 4, 34x. Import with the warm-up
search 2.4 s. Peak RSS 264 to 282 MB on 120 s moves.

The first gate with games: 200 against v2.4 at the fast control, four at a time, 22 minutes.
Terminations: 185 checkmates, 11 threefold repetitions, 4 insufficient material.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| v3-200 | baseline | 10s+0.1s | 200 | +181 =15 -4 | 94.2% | +486 | +416 to +596 | 0 / 0 / 0 / 0 | 1.28s | 266 MB |

The worst move is the 1.25 s hard budget plus the slice the node budget lands in, as it was for
every python-chess version.

120 s + 0.5 s games against v2.4 from the English Opening, one per colour, both won by v3.0 by
checkmate (57 and 54 moves). First 40 moves, v3.0 against v2.4 on the other side of the same
board: depth 8.05 / 8.93 against 5.85 / 6.33; spend 116% / 124% of the soft budget against
v2.4's 116% / 134%; 7 and 5 partial iterations. Slowest moves 12.9 s at a 13.0 s hard budget
and 13.8 s at 13.8 s: the node budget lands the abort inside the hard budget, and the largest
overshoot past it was 25 ms, the root-move granularity, against the 32 to 49 ms clock-check
slices of the python-chess versions. Clock minimum 12.3 s and 14.7 s (v2.4 sank to 9.3 s in
the first game). Median 2.35M and 2.40M nps over the game against v2.4's 92k and 94k. Peak RSS
263 MB, both games; no fallback to the Python engine on any move.

The gauntlet, two games at a time alongside the 120 s games:

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| v3-gauntlet | baseline | 10s+0.1s | 32 | +28 =4 -0 | 93.8% | +470 | +345 to +946 | 0 / 0 / 0 / 0 | 1.22s | 266 MB |
| v3-gauntlet | sunfish | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.23s | 263 MB |
| v3-gauntlet | minimax | 10s+0.1s | 16 | +16 =0 -0 | 100.0% | +inf | +inf to +inf | 0 / 0 / 0 / 0 | 1.35s | 263 MB |
| v3-gauntlet | overall | 10s+0.1s | 64 | +60 =4 -0 | 96.9% | +597 | +475 to +1146 | 0 / 0 / 0 / 0 | 1.35s | 266 MB |

v2.4 scored 71.9% against Sunfish on this bench; v3.0 wins all sixteen. The minimax column's
worst move, 1.35 s at a 7.3 s clock, is not over budget by the bench's rule (a quarter of the
clock) but it is 48% past the 0.91 s hard budget, where the python-chess versions overshot by
one clock-check slice of tens of milliseconds. The node budget for a root move is set from the
node rate measured so far in the move, so a rate that sags inside that one subtree runs past
the deadline; see the logbook for the backstop this led to and the re-run on the final code.

Regression suite, 17 positions, 20 s each, one process alone: **v3.0 solves 5 of 17** reaching
depth 7 to 9 (r73 m34 at d4, r73 m37 at d5, r75 m23 at d8, r75 m25 at d7, r75 m27 at d1);
**v2.4 solves 4 of 17** reaching depth 5 to 7 (r73 m34, m37, m40, r75 m27). Three in common;
v3.0 finds the round 75 Ba4 idea twice where v2.4 never does, and drops r73 m40, which it
chose at depth 5 and leaves at depth 9. Both still play the round 74 queen move that walks into
mate, at depth 9 with a score of -815: the mate is beyond the horizon and the evaluation is
v2.4's, so deeper search converges on the evaluation's preference, not Stockfish's. The suite
measures sharpness; the games above decide.
