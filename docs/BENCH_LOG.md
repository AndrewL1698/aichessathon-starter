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

## v3.2 candidate, 2026-09-09: the learned evaluation at runtime (nnue/runtime)

`fastnnue.py` is the shipped half of the NNUE: numba inference of the integer weights
`tools/nnue/export.py` writes, with two perspective accumulators kept per ply. This section is
**mechanics and speed**. The evaluation changed, so the search tree changed, and nothing here
is an Elo claim about the network beyond the rows measured below.

### What was verified rather than assumed

`uv run python -m tests.test_nnue [--full]`, against every `weights/nnue*.npz` present. The
three files on `nnue/weights-v1` deliberately use three different scale combinations — h128 at
`qa512 qb512`, h256 at `qa512 qb512`, h256-e87 at `qa256 qb1024` — and every scale and the
hidden width is read from the file, so a scale assumption cannot hide:

- **Exact equality with `tools/nnue/nnue_ref.py`**, 2,000 positions per file (943 with Black to
  move, 60 with a live en passant, 198 with a promotion available, 985 endgames): **0
  mismatches**, all three files. Not a tolerance — the reference is the specification the
  export was verified against, and the two routes to the answer are independent, a mailbox scan
  with a precomputed index table here against `board.piece_map()` and the index formula there.
- **Incremental accumulators equal a from-scratch build**, 10,000 make/unmake sequences per
  file driven through the calls the search makes in the order it makes them, both perspectives
  compared in full at every ply (362 captures, 4 en passant, 22 promotions, 6 castles, 363 null
  moves): **0 mismatches**. `acc[ply]` is also snapshotted and re-checked after each unmake,
  which is the property the design rests on: unmaking is free because `acc[ply]` is never
  written.
- **Seven broken weight files all refused** and a missing one reported as missing: a future
  scheme version, a hidden size the arrays contradict, a zero scale, a transposed first layer,
  a bias at the wrong width, a bias at the wrong dtype, and an `l1_weight` whose int16
  accumulator can overflow. A refused file is the hand evaluation and a line in the init log,
  never a crash and never a net read with the wrong shapes.
- **Floor division**, 65 cases including negative numerators at every scale a shipped file
  uses. C truncation would put every negative evaluation one centipawn high, which is small
  enough to pass a tolerance and large enough to break the equality above.
- **`tests.test_fastsearch` with `USE_NNUE` off** still scores exactly what `agent.py` scores,
  30 searches, unchanged. That is what says this branch changed the evaluation and nothing else.
  Its legality, timeout and backstop checks pass again with the network on.

### The evaluation policy, and the one thing the brief got wrong

The plan was "network centipawns plus `fasteval`'s mop-up term in mop-up positions". Measured,
that is not enough. Played out at a fixed depth with the network scoring the leaves and the
mop-up term added on top:

| ending | net + mop-up | hand evaluation |
|---|---|---|
| KRvK | mate | mate |
| KQvK | mate | mate |
| KRRvK | **draw by repetition** | mate in 9 plies |
| KPvK | **draw by repetition** | mate |

The reason is structural rather than a tuning miss. The network's training set is positions
real games reached, so KRRvK and KPvK are a vanishing fraction of it, and it scores every legal
move in KRRvK within a few centipawns of every other — every one of them leaves the same men on
the board. The mop-up term is worth at most 120 cp by construction and the network's own
variation across those positions is larger than that. KPvK is worse still: it is a hundred
centipawns ahead, not four hundred, so `fasteval`'s mop-up condition does not even fire and the
network was scoring it alone.

So the policy shipped is a handover, not an addition: **leaves past `fastnnue.bare_endgame` —
either side down to a king and at most two other men, which is `fasteval`'s own
`MOP_UP_MAX_WEAK_PIECES` — are scored by the hand tables outright**, mop-up, drawish scaling and
bare-minor zero included. There is nothing to lose by it: everything a learned evaluation knows
is about positions with men on the board, and what decides a bare endgame is geometry and the
fifty-move clock, which is what those three terms were written and tested for. The test is that
all five bare endgames convert *and* that the playouts are identical with the network on and
off, which cannot pass if a single leaf past the line is still scored by the network:

```
KRvK mate in 33 plies, KQvK mate in 19, KRRvK mate in 11, KPvK mate in 41, KBBvK mate in 47
```

The gate costs a dozen loads on a full board, because it counts each side's men from that
side's own end and stops at the limit.

### Speed

The cost of a learned evaluation is paid twice per node: an accumulator update on the way down
and an inner product at the leaf. Depth 7 over the six `tests.test_fastsearch` positions, h128:

| position | network on | hand evaluation | ratio |
|---|---|---|---|
| start | 2.02M nodes/s | 2.97M | 68% |
| kiwipete | 1.78M | 2.01M | 88% |
| italian | 1.81M | 2.17M | 83% |
| queens gambit | 2.01M | 2.56M | 78% |
| sicilian | 1.83M | 2.62M | 70% |
| endgame | 1.62M | 2.14M | 76% |
| **mean** | **1.85M nodes/s** | **2.41M** | **77%** |

Comfortably past the 1.0M target, and it took two fixes to get there. The first version ran
**0.56M nodes/s**: the second layer's inner product accumulated into int64 and clipped the
accumulator with a branch, and neither vectorises, so 4,096 multiply-accumulates cost 2,815 ns
a leaf. int32 accumulation of an int16 by int16 product is one SIMD instruction and `min`/`max`
clips without a branch — the same integers, 610 ns. Fusing the accumulator copy with the moving
piece's delta saved another pass. Hoisting the clip into a scratch row was measured too and
saved 29 ns of 610, which is not worth another array threaded through every frame of the search,
so it was not done.

Import with warm-up **5.2 s** against v3.1's 4.1 s (the target was 6 s; the 90 s the platform
allows is not close to binding), peak RSS **251 MB** against v3.1's 220 MB, and the 2 GB cap is
not close either.

### Gauntlet

Every 64-game row is at 10 s + 0.1 s. `nnue-h128` and `nnue-blend-e60-v31` ran one game at a
time; the two `-v32` rows ran two at a time, so their worst move times are taken under load and
the timing measurement to trust is the single-game rows'. Weights come from `nnue/weights-v1`;
the smoke net was used only while the mechanics were being written and no number here is its.

**Read the baseline column before the score column.** `local-opponents/v3.1` has no opening
book and `local-opponents/v3.2` does. A post-merge candidate has one, so a row against v3.1 is
measuring the book as well as the evaluation, and the v3.2 rows are the ones where the
evaluation is the only difference between candidate and baseline.

| run | net | policy | val loss | baseline | games | +=- | score | Elo | 95% | ill/exc/tmo/over |
|---|---|---|---|---|---|---|---|---|---|---|
| nnue-h128-disq | h128 21M | absolute | 0.01654 | v3.1 (no book either side) | 16 | +8 =1 -7 | 53.1% | +22 | -158 to +216 | 0 / 0 / 0 / 0 |
| nnue-h128 | h128 21M | absolute | 0.01654 | v3.1 (no book either side) | 64 | +20 =5 -39 | 35.2% | -106 | -201 to -25 | 0 / 0 / 0 / 0 |
| nnue-h256-e87 | h256 21M e87 | absolute | 0.01541 | v3.2 | 64 | +26 =3 -35 | 43.0% | -49 | -139 to +34 | 0 / 0 / 0 / 0 |
| nnue-h256-52m | h256 52M | absolute | 0.01489 | v3.2 | 64 | +28 =6 -30 | 48.4% | -11 | -95 to +72 | 0 / 0 / 0 / 0 |
| nnue-residual-v32 | h256 52M res | residual | 0.01450 | v3.2 | 64 | +28 =5 -31 | 47.7% | -16 | -101 to +67 | 0 / 0 / 0 / 0 |
| nnue-h512-res-v32 | h512 52M res | residual | 0.01350 | v3.2 | 64 | +30 =6 -28 | 51.6% | +11 | -72 to +95 | 0 / 0 / 0 / 0 |
| nnue-blend-e60-v31 | h256 52M e60 | blend | 0.01460 | v3.1 (**book only on our side**) | 64 | +53 =5 -6 | 86.7% | +326 | +231 to +489 | 0 / 0 / 0 / 0 |
| **nnue-blend-e60-v32** | **h256 52M e60** | **blend** | **0.01460** | **v3.2** | **64** | **+44 =8 -12** | **75.0%** | **+191** | **+109 to +298** | 0 / 0 / 0 / 0 |

### What the rows say

**The blend wins, and it wins by a lot.** Averaging the network with the hand evaluation,
`(hand + net) // 2`, scored **75.0%, Elo +191, interval +109 to +298** against v3.2 with the
book on both sides. The *same weight file* used alone is worth 48.4%. That is not a tuning
gain; it is the difference between a network that costs Elo and one that pays about 200 of it,
from one line in `leaf`.

The hypothesis the blend was built to test was exactly right: a network that is noisy but
carries real signal should beat both halves when averaged with a solid evaluation, and a
network that carries nothing new should land between them. It landed far above both, so the
network knows things `fasteval` does not; what it could not do alone was keep its material
sanity, and halving it against the hand tables supplies that.

**The 86.7% row is the one not to quote.** Its candidate had the opening book and its baseline
did not, so it measures the book too. It is kept here because the honest version of "we got
+326" is "+326 was the confounded number, +191 is the controlled one", and because the gap
between the two is a reasonable estimate of what the book is worth from those openings.

**The absolute rows are monotone in validation loss** -- 0.01654 / 0.01541 / 0.01489 giving
-106 / -49 / -11 Elo -- so the offline metric's *ordering* is worth trusting even though its
level says nothing about board strength.

**The residual net is the surprise, and it is a negative one -- at two widths.** Training the
network on Stockfish's centipawns *minus* `fasteval`'s, and scoring leaves as `hand + net`, is
the principled version of the blend. It reaches the best validation losses of anything measured
here -- 0.01450 at 256 wide and **0.01350 at 512**, against the hand evaluation's 0.0329 on the
same split -- and it played at **47.7%** and **51.6%**. Both are parity. The 512-wide file is
the best net on paper by a clear margin and it is 180 Elo behind averaging a *worse* net with
the hand evaluation.

That is the most useful thing on this page, because it says the offline metric stops ordering
things once the composition changes: within the absolute nets, validation loss ranks them
correctly; across policies it does not rank them at all. Worth being clear about the mechanism,
because it points somewhere cheap: the residual is added at *full* weight, so the network's
noise arrives at full weight with it, while the blend halves that noise relative to material.
If that reading is right the thing to try is a residual at half weight -- `hand + net // 2` --
which is one more branch in `leaf` and no new training.

Across 464 benched games and six weight files there were **zero illegal moves, zero exceptions,
zero flag falls and zero over-budget moves**. `exceptions` is also the fallback count, since
`_think_fast` raises on any move python-chess will not accept, so the numba engine never handed
out a move the board did not believe in.

### Speed by hidden width

Depth 7 over the six `tests.test_fastsearch` positions, against the hand evaluation's 2.4-2.5M:

| net | width | policy | nodes/s | of hand |
|---|---|---|---|---|
| h128 21M | 128 | absolute | 1.85M | 77% |
| h256 52M e60 | 256 | blend | 1.30M | 53% |
| h256 52M res | 256 | residual | 1.35M | 55% |
| h512 52M res | 512 | residual | **0.94M** | 39% |

The accumulator copy and the second layer both scale with the width, so doubling it costs about
a third of the node rate -- roughly half a ply. **h512 is the only file to miss the 1.0M
target**, at 0.94M, and its row above says it did not buy anything with the width: 51.6% at
0.94M against the blend's 75.0% at 1.30M. So the price is real and, for the residual policy at
least, it was not worth paying.

### Why, as far as this branch can tell

Over 1,393 positions from random playouts, with the bare endgames excluded so the handover is
not what is being measured:

| | slope against material | r against material | sd |
|---|---|---|---|
| network | 0.64 | 0.706 | 498 cp |
| hand evaluation | 1.00 | 0.983 | 559 cp |

The network knows what the pieces are worth — from the start position, removing a Black pawn is
+76 cp, a knight +337, a rook +483, the queen +857, all close enough — but across a sample it
tracks material at r = 0.706 against the hand evaluation's 0.983, on a scale 36% flat, and its
mean absolute disagreement with the hand evaluation is 288 cp. That is a couple of hundred
centipawns of positional opinion swinging around on top of material, and a search whose leaves
disagree with each other by that much will trade a pawn for nothing.

Read the correlation figures with one caveat: random playouts are not the distribution the net
was trained on, so they measure its behaviour *off* distribution rather than its quality. The
64-game result is on real games and is the verdict; the correlation is a hypothesis about the
mechanism, and the actionable version of it is that the loss (MSE on `sigmoid(cp / 400)`) buys
very little accuracy per centipawn once a position is lopsided, which is exactly where a search
needs it. Worth trying before another bench: more data, a wider net, or a loss with a material
term in it.

The 23% node-rate cost is not the explanation. Three quarters of the nodes is about a third of a
ply, which is worth tens of Elo at these depths, not a hundred.

### The side-to-move offset, and the one line that read it

The 52M-position nets carry a tempo bonus: they score a dead-equal position at **+46 cp for
whoever is to move** — the same +46 with either side to move, in the start position and in bare
kings alike, against the hand evaluation's 0. That is not a bug in the export and not a
perspective error; the mirror identity still holds exactly, which is what the parity test
proves. It is a term the network learned, and a moderate one is normal in an engine.

It cancels in negamax at even depths and does not cancel where a centipawn figure is compared
against a constant. There is exactly one such place, and it was found by looking rather than by
losing games to it: `contempt_for`'s `CONTEMPT_THRESHOLD`, 150 cp, calibrated against
`fasteval`. Over 760 root positions, feeding the network's score to it fired contempt in **69%**
of them against the hand evaluation's **58%**, agreeing on the sign only **62%** of the time.
What that buys is a draw refused in positions that are not actually won.

So contempt reads the hand evaluation whatever scores the leaves (`fastsearch.root_contempt`).
Nothing else in the draw handling reads an evaluation at all — repetition, the fifty-move rule
and insufficient material return `draw_score` directly — and the `fastnnue.bare_endgame`
handover is a count of men on the board, not a score, so a cp offset cannot move it either. The
h128 and h256-e87 rows above were measured *before* that change, with the network feeding
contempt; the fix can only have helped, and it is one line if it needs re-measuring.

## v4.0, 2026-09-09: the learned evaluation switched on (nnue/v4.0)

The runtime is PR #14's; this is the file and the switch. Weights `nnue-h256-52m-e60.npz`
(sha ed74493b) as `weights/nnue.npz`, leaf = (hand + net) / 2. Re-run from scratch by the
orchestrator, then the platform proxy and the 120 s games, all against `local-opponents/v3.2`.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| nnue-blend-confirm96 | v3.2 | 10s+0.1s | 96 | +68 =18 -10 | 80.2% | +243 | +177 to +329 | 0 / 0 / 0 / 0 | 1.26s | 256 MB |
| nnue-blend-proxy45 | v3.2 | 45s+0.2s | 24 | +19 =3 -2 | 85.4% | +307 | +172 to +668 | 0 / 0 / 0 / 0 | 5.65s | 256 MB |

120 s + 0.5 s, one game per colour: both won by checkmate (32 and 30 moves). Depth over the
first 40 moves 6.69 / 6.20 against v3.2's 7.97 / 8.07 on the other side of the same boards: the
net costs about a ply and a half at 1.3M nodes/s against 2.4M, and wins anyway. Worst overshoot
of the hard budget 1 ms; slowest moves 11.9 s at a 13.4 s hard budget and 9.8 s at 10.6 s;
clock minima 24.4 s and 41.0 s. Pooled with the doer's 64-game row the blend is +112 =26 -22
over 160 games, 78.1%.


## Cycle 4, small fixes, 2026-09-09: what the rated-game reviews pointed at

Baseline `local-opponents/v4.0`, 96 games at 10 s + 0.1 s per candidate, four at a time, the
disqualifier counts and peak RSS from `harness.bench`. These are the three "small" items from the
reviews of rounds 76 to 86 in `docs/LOGBOOK.md`: a rule the evaluation lacks, one time constant,
one blend weight. One change per branch off `prod`.

### `eval/kpk-rook-pawn-draw`: king and rook pawn against a bare king is a draw

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| kpk-rook-pawn-draw | baseline | 10s+0.1s | 96 | +28 =32 -36 | 45.8% | -29 | -88 to +28 | 0 / 0 / 0 / 0 | 1.29s | 256 MB |

**The bench is not the instrument for this one, and the row says why.** The rule can only fire
in king-and-one-pawn positions with the pawn on a rook file and the defender in front; scanning
the 96 PGNs, 7 games reached a king-and-pawn ending of any kind and **3 reached the rule's
position, all three with the candidate defending, all three drawn**, which is what those
positions are. The other 93 games ran code that is byte-for-byte v4.0's apart from a handful of
integer compares at the leaf, so the -29 is the timing noise of two identical engines playing
four at a time on a laptop; the interval is the honest statement and it includes zero. Node
rate in `tests.test_fastsearch` is unchanged (3.85M / 2.59M nodes/s on start / kiwipete against
3.93M / 2.61M on the sibling branch). What the change is measured by is `tests.test_fasteval`'s
`check_kpk` and the round 82 positions at depth 10: the three game positions score 0 with
contempt 0 where v4.0 scored +182 to +292 with contempt -50; the won rook-pawn ending with the
attacking king on g7 (+928), the centre-pawn KPvK (+162) and KRvK (+614) are unchanged.

## Cycle 4 (M5), 2026-09-09: the round 85 fixes

Two branches off `prod` (v4.0) from the M5, one change each, against `local-opponents/v4.0`.
A training job (`tools.nnue.train`, one core plus the GPU) was running on the same machine
throughout both benches; games ran one at a time.

### `search/promo-tiebreak`: an equal-scoring promotion resolves to the queen

Round 85 move 56 (`8/6R1/3P4/4K3/Pp6/1P5k/6p1/5r2 b - - 0 56`): `g2g1q`, `g2g1r` and `f1e1`
all scored +21 at depth 9 and the engine played the rook. The mechanism is ordering, not
evaluation: `think` hands the previous iteration's answer to `search_root` as `first`, which is
ranked `TABLE_BONUS` and searched before everything, and `search_root` improves on it only with
a strict `>`. A depth-2 iteration returned the rook, every deeper iteration inherited it, and no
equal score could take it back. `queen_first` swaps an under-promotion out of that slot for the
queen promotion of the same move when both are legal; the under-promotion stays in the list at
its own rank, so Saavedra's `c7c8r` is still found (it is strictly better), which the new test
asserts alongside the round 85 position at four clocks. The search and the evaluation are
otherwise byte-identical to v4.0; the scores are unchanged, only the tie changes hands.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| search-promo-tiebreak | v4.0 | 10s+0.1s | 64 | +25 =21 -18 | 55.5% | +38 | -32 to +111 | 0 / 0 / 0 / 0 | 1.26s | 257 MB |

A disqualifier run, not a strength claim: the change fires only on a promotion tie, so the
interval is v4.0 against itself plus noise. `uv run ruff check .`, `uv run mypy`,
`tests.test_fastsearch` (with the new `promotion tie-break` check) and `tests.test_nnue` all
green on the rebased branch.

## Cycle 5, 2026-09-09 night: the two candidates from the rounds 73 to 90 review

Two branches off `prod` at 2354ff5 (v4.1 plus the cross-game review), one change each, against
`local-opponents/v4.1`, 96 games at 10 s + 0.1 s four at a time with nothing else running; the
check extension also at the 45 s + 0.2 s proxy and re-run from scratch. Regression suite: prod's
47 positions, `--engine fast`, depth 11, 20 s. `ruff`, `mypy` and `tests.test_fastsearch` (plus
`tests.test_nnue` for the contempt branch) green on both.

### `search/check-extension`: a node in check is searched one ply deeper

Round 90's ending, a rook against two rooks and a pawn, was drawn by perpetual as long as the
checks came from the h-file; the engine played the f-file check at moves 69 and 70 and was mated,
because every check cost a full ply and the perpetual never repeated inside the horizon. With the
extension the drawing `Rh8+` is chosen from depth 8 instead of 12 at move 69 and `Rh7+` from
depth 6 instead of 9 at move 70, still scored as a rook down at every depth to 12 either way:
what changes is that the checks are searched to their replies. Cost: 4.6 to 7.4x the nodes in
those endings, about 6% of all nodes extended in a middlegame, and half a ply of nominal depth in
20 s on the suite (7.89 against 8.40).

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| search-check-extension | v4.1 | 10s+0.1s | 96 | +42 =23 -31 | 55.7% | +40 | -21 to +103 | 0 / 0 / 0 / 0 | 1.28s | 257 MB |
| search-check-extension | v4.1 | 45s+0.2s | 48 | +19 =17 -12 | 57.3% | +51 | -28 to +136 | 0 / 0 / 0 / 0 | 5.65s | 253 MB |
| search-check-extension, re-run from scratch | v4.1 | 10s+0.1s | 96 | +35 =24 -37 | 49.0% | -7 | -69 to +54 | 0 / 0 / 0 / 0 | 1.29s | 256 MB |
| pooled, all three | v4.1 | both | 240 | +96 =64 -80 | 53.3% | +23 | -14 to +61 | 0 / 0 / 0 / 0 | | |

Suite 26/47 against v4.1's 27/47: solves `r74 m21` (a mate horizon) and `r90 m69`, which v4.1
misses, and misses `r77 m14`, `r78 m16` and `r84 m23`, which v4.1 solved at exactly the depth the
extension no longer reaches in 20 s; nine other positions are solved at a lower nominal depth
(`r90 m39` at d6 against d9). Clock minimum at the proxy 2.5 s on both sides. **Not proven**: the
first run's +40 did not reproduce from scratch, the pool's lower bound is -14, and the suite says
it buys one class at the price of half a ply everywhere else. Branch pushed for a PR.

### `eval/contempt-quiescence`: contempt from the hand evaluation resolved through quiescence

`root_contempt` fed the raw static evaluation to the threshold, so a pending recapture read a
piece down and set contempt to +50 (a draw worth half a pawn to us) in level positions: round 90's
11...Nxa6, 19...Qxe4 and 20...Rxe4 read -359, -381 and -947 against a search score of -35, and
v4.0's rated games did that on 22 level moves. The hand evaluation is now read through
`quiescence` with the leaf policy forced to `HAND`: same threshold, same scale, one quiescence
search per move.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-contempt-quiescence | v4.1 | 10s+0.1s | 96 | +41 =15 -40 | 50.5% | +4 | -61 to +69 | 0 / 0 / 0 / 0 | 1.29s | 258 MB |

A disqualifier run, like PR #17's: the change only matters when a draw is on offer with a
capture pending. Measured directly over the 24,127 positions of two benches: contempt changes on
8.4%, draw-seeking dropped on 1,313 (5.4%), draw-averse dropped on 58, and 713 positions where we
are about to win material now refuse a draw. Suite unchanged at 27/47. Branch pushed for a PR, to
be judged on the mechanism.


## Cycle 6, 2026-09-10: mobility in the hand evaluation. Rejected over 128 games

Branch `eval/mobility`, PR #19, **not merged and not shipped**. The engine change is not in
this commit; only the record is. The branch keeps the code if the evaluation ever changes
enough to make the term worth retesting.

The term: for each knight, bishop, rook and queen, the squares it attacks that none of our own
men stand on and no enemy pawn covers, scored against a typical count for that piece so a full
board sits near zero rather than paying a bonus for owning pieces. Kings and pawns excluded.
Weights per square, tapered mg/eg: knight 4/4, bishop 3/4, rook 2/4, queen 1/2, against
baselines 4/6/7/13. Written into both evaluations, because `tests/test_fasteval.py` holds them
to the same integer: `agent.py` by `attacks_mask` and a popcount, `fasteval.py` by walking the
move generator's rays with two mailbox reads per target for the pawn cover.

The note in `agent.py`'s `_pieces` saying mobility was "deliberately absent" because
python-chess has to generate moves to count it was written when `agent.py` was the engine. It
is the fallback now, so that reasoning had expired and the term was worth measuring. It was
measured, and it does not pay.

### The three runs

| run | opponent | openings | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over |
|---|---|---|---|---|---|---|---|---|---|
| arena | `prod` df8f1fb | 8 | 10s+0.1s | 64 | +27 =14 -23 | 53.1% | +22 | -54 to +100 | 0 / 0 / 0 / 0 |
| arena | `prod` df8f1fb | 8 | 45s+0.2s | 32 | +7 =11 -14 | 39.1% | -77 | -188 to +19 | 0 / 0 / 0 / 0 |
| bench h2h | `prod` b508b37 | 28 | 10s+0.1s | 32 | +10 =8 -14 | 43.8% | -44 | -158 to +62 | 0 / 0 / 0 / 0 |
| **pooled** | | | | **128** | **+44 =33 -51** | **47.3%** | **-19** | **-72 to +33** | **0 / 0 / 0 / 0** |

Both baselines predate v4.1's promotion tie-break (PR #20): df8f1fb is v4.0 plus the rook-pawn
rule, b508b37 is the same plus docs. The candidate is v4.0-plus-rook-pawn with mobility added,
so each row is still one variable against its own baseline, which is what the rows claim. None
of them is a comparison against v4.1 as shipped.

Only the first row is above 50%, and it is the one on the narrowest opening set. Pooling across
two controls and two opening sets is not clean arithmetic; read the direction, which is
consistent and downward, rather than the third digit.

### The head-to-head, and the finding that closes the question

32 games, both engines pinned to exact commits in separate detached worktrees so neither could
change mid-run, one game at a time, paired openings with colours reversed, nothing else on the
machine. `fastsearch.py` byte-identical between the two trees, so the only difference was the
evaluation. PGNs and both agents' logs kept.

| metric | candidate | baseline `prod` b508b37 | delta |
|---|---|---|---|
| depth, real searches, mean | 7.14 | 7.18 | **-0.04** |
| depth, real searches, median | 7.0 | 7.0 | **0** |
| aggregate nodes/s | 1.405M | 1.519M | **-7.5%** |
| nodes searched | 574.0M | 614.1M | -6.5% |
| total think time | 408.7 s | 404.4 s | +1.1% |
| slowest move | 1.26 s | 1.27 s | - |
| clock floor | 1.00 s of 10 s | 1.00 s of 10 s | - |
| peak RSS | 255 MB | 254 MB | - |

Failures on the candidate side across all 128 games: **illegal 0, exception 0, timeout 0,
over-budget 0, tracebacks 0.** The rejection is on strength alone; the code is sound.

**There is no depth loss, and that is what settles it.** The term costs 7.5% of the node rate
and hands back a 6.5% smaller tree, landing on the same median depth 7.0 and within 0.04 ply on
the mean. So the loss cannot be attributed to the cost of computing it -- in real games the
better ordering very nearly pays for the extra work. What is left is the evaluation term
itself, and it is not adding anything.

The mechanism that fits: a 256-wide network trained on 52M evaluated positions already encodes
mobility. It is one of the easiest things for a piece-square net to learn, and the leaf is
`(hand + net) // 2`, so the hand term re-states at half weight what the other half of the leaf
already knows, while the node-rate cost is paid in full. **A hand-authored term that duplicates
something the net has learned is not free; it is a tax.** That is the transferable result here,
and it applies to the next hand-authored term someone proposes on top of a learned evaluation.

Statistically, no interval excludes zero, the pooled one included (-72 to +33). This is not
proof the term is harmful. It is firm evidence it is not an improvement, which is the question
that was asked. Resolving a genuinely small effect would need 200+ games at one control, and
there is no case for spending them after three runs failed to find a positive signal.

### A flag in the 45 s run was an artifact, not the engine

The 45 s proxy run was suspended with `SIGSTOP` partway through to free the machine, then
resumed. Game 14 was in flight. The referee measures every move against the wall clock, so the
suspension was billed to our agent as thinking time, and the PGN shows it:

```
11... Bxb2 { [%clk 0:00:43.432] }  12. Rb1 { [%clk 0:00:40.272] }  1-0  [Termination "flag"]
```

An agent with 43.4 s of a 45 s clock does not flag on the next move. The first scoring of that
run reported 35.9% **with one flag** -- the termination class these rules call priority zero --
and it was not real. Game 14 was replayed from the same index, which is the same pairing
because the arena's game order is a pure function of the index, and it is a win. The row in the
table above is the clean 32. The rule this produced is in `docs/TEAM.md`.

## Cycle 4 (M5), 2026-09-10: the over-budget moves were the laptop, not the engine

Cycle 4's late move reductions measured +74 Elo on the laptop and were disqualified anyway, by
three moves of 396 to 758 ms against hard budgets of 191 to 229 ms. Every run below is on the
M5, one bench at a time, machine otherwise idle. The baseline is `local-opponents/v4.1`.

### The probe: `think` and `get_move` against the hard budget, in process

50 positions (26 rated-game blunders from `tests/positions/positions.epd`, 6 from this repo's
bench and endgame sets, 18 from the round 73-90 PGNs at plies 24/40/56/72), none of them a book
position, at four short clocks, with reductions on and off, measuring both `fastsearch.think` and
the `agent.get_move` the harness actually calls. `think` was not edited: an instrumented
line-for-line copy stamped every stage of its python driver and was installed over
`fastsearch.think`, which `get_move` calls too. The table clears before every measurement, so a
cold table is what the projection has to work with. 800 measurements.

| clock | LMR | entry | median over hard | p95 | max | over 50 ms |
|---|---|---|---|---|---|---|
| 1200 ms | on | think | -64.9 ms | +0.1 ms | +0.2 ms | 0/50 |
| 1200 ms | on | get_move | -62.6 ms | +0.5 ms | +0.5 ms | 0/50 |
| 1200 ms | off | think | -73.2 ms | +0.1 ms | +0.1 ms | 0/50 |
| 1200 ms | off | get_move | -73.5 ms | +0.5 ms | +0.5 ms | 0/50 |
| 1500 ms | on | think | -69.2 ms | +0.1 ms | +0.2 ms | 0/50 |
| 1500 ms | on | get_move | -69.6 ms | +0.5 ms | +0.5 ms | 0/50 |
| 1500 ms | off | think | -85.2 ms | +0.1 ms | +0.2 ms | 0/50 |
| 1500 ms | off | get_move | -86.5 ms | +0.5 ms | +0.5 ms | 0/50 |
| 2000 ms | on | think | -92.9 ms | +0.2 ms | +0.2 ms | 0/50 |
| 2000 ms | on | get_move | -94.4 ms | +0.5 ms | +0.6 ms | 0/50 |
| 2000 ms | off | think | -100.5 ms | +0.2 ms | +0.2 ms | 0/50 |
| 2000 ms | off | get_move | -100.0 ms | +0.5 ms | +0.9 ms | 0/50 |
| 2500 ms | on | think | -140.8 ms | +0.6 ms | +0.9 ms | 0/50 |
| 2500 ms | on | get_move | -143.7 ms | +0.6 ms | +1.2 ms | 0/50 |
| 2500 ms | off | think | -132.0 ms | +0.7 ms | +0.9 ms | 0/50 |
| 2500 ms | off | get_move | -132.5 ms | +1.2 ms | +1.3 ms | 0/50 |

The worst of the 800 is `get_move` at **+1.27 ms** past the hard budget. Where the milliseconds
go, as medians: the setup before the loop (`from_fen`, `legal_moves`, `observe`, the contempt
evaluation, the from-scratch accumulator) 0.32 ms, the backstop's cancel and join 0.03 ms, the
log line 0.02 ms, and the whole `agent.py` wrapper around `think` (the python-chess board, the
fallback memory's reachable scan, the book probe, the legality check, `remember_played`) 0.34 ms.
62 of the 400 moves ended on an abort, and those are the tight ones: median +0.13 ms, p95
+0.68 ms, max +0.90 ms past the deadline. Quiescence nodes count toward the clock-check mask
exactly as negamax's do, so there is no unchecked region of the tree. The engine's problem at
these clocks is the opposite of an overshoot: the median move spends 51-63% of its hard budget.

Repeating the 1500 ms probe with 36 spinning processes on 18 cores, about twice oversubscribed,
moved the maximum from +0.2 ms to **+6.2 ms with reductions on and +22.9 ms with them off**, and
still put no move past a quarter of its clock. So contention is the mechanism and it points the
right way, but twice oversubscribed does not reach 396 ms; the laptop's runs were two four-job
benches on ten cores, with a 260 MB engine per game and a five-second numba compile as each new
game's process imports. **No time-management fix was made.** Every cause the probe could have
found is under a millisecond, and the 200-game row below is the same measurement that
disqualified the cycle, run on an idle machine.

### The rows

`time/platform-spend-v41` is v4.1 plus 515cc62 alone (the iteration gate capped at 1.5x the soft
budget). `stack/v42` is `search/lmr` (PVS, late move reductions, and "never reduce at a principal
variation node") plus that same commit. The middle row is the intermediate build the probe was
run against, kept because its 200 games are the direct answer to the disqualifier.

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| platform-spend + PVS, LMR off | v4.1 | 10s+0.1s | 200 | +84 =44 -72 | 53.0% | +21 | -22 to +64 | 0 / 0 / 0 / 0 | 1.26s | 262 MB |
| A `time/platform-spend-v41` | v4.1 | 10s+0.1s | 200 | +68 =59 -73 | 48.8% | -9 | -50 to +32 | 0 / 0 / 0 / 0 | 1.29s | 261 MB |
| A `time/platform-spend-v41` | v4.1 | 45s+0.2s | 48 | +12 =13 -23 | 38.5% | -81 | -175 to +2 | 0 / 0 / 0 / 0 | 5.65s | 255 MB |
| B `stack/v42` | v4.1 | 10s+0.1s | 200 | +115 =48 -37 | 69.5% | +143 | +101 to +190 | 0 / 0 / 0 / 0 | 1.28s | 263 MB |
| B `stack/v42` | v4.1 | 45s+0.2s | 48 | +26 =15 -7 | 69.8% | +145 | +66 to +243 | 0 / 0 / 0 / 0 | 5.63s | 257 MB |
| C `stack/noceiling` | v4.1 | 10s+0.1s | 200 | +121 =41 -38 | 70.8% | +153 | +110 to +202 | 0 / 0 / 0 / 0 | 1.29s | 263 MB |
| C `stack/noceiling` | v4.1 | 45s+0.2s | 48 | +27 =16 -5 | 72.9% | +172 | +94 to +270 | 0 / 0 / 0 / 0 | 5.68s | 257 MB |

**Not one over-budget move in 1,096 games**, the 10 s rows at `--jobs 9`, the 45 s rows at
`--jobs 4`. The worst move of every run is its hard budget plus a clock-check slice. That closes
the disqualifier: it was the laptop.

**The search is the gain and the ceiling is not.** B and C both clear zero at both controls and
are the same measurement within noise at 10 s (+143 and +153), which is the check that the
ceiling really is inert below a ~9.2 s clock, since the two builds differ only by 515cc62. At
45 s, where it does bind, C without it scores 72.9% against B's 69.8%, a point estimate 27 Elo
apart on heavily overlapping intervals. That is not significant on 48 games a side, but it is the
same sign as the ceiling measured alone: A is -9 [-50, +32] fast, which says little for the same
inertness reason, and **-81 [-175, +2] at 45 s**. Two independent measurements of one commit,
both negative on strength.

### 120 s + 0.5 s, one game per colour

`harness/sandbox.py` keeps a head and a tail of each game's stderr with `[N bytes dropped]`
between them, mirroring the platform's 4 KB + 4 KB, so a long game's middle is never logged. The
clocks below therefore come from the PGN, which is complete: `%clk` is the clock after the move
including the increment, and it matches the engine's own logged `clock` for the next move to a
millisecond, so a spend derived from it is the engine's spend plus the harness round trip. Depth
can only come from the log, so its column says how many of our opening moves it covers.

| cand | game | result | our moves | depth over our first N logged | clock minimum | slowest move | worst spend minus hard |
|---|---|---|---|---|---|---|---|
| A | white | draw, insufficient material | 72 (to move 78) | 7 / 8 / 9 over 27 | 17.24 s at move 77 | 8430 ms | +1 ms |
| A | black | win by checkmate | 50 (to move 55) | 6 / 7 / 9 over 40 | 40.26 s at move 52 | 7351 ms | -1967 ms |
| B | white | draw, fifty moves | 166 (to move 172) | 9 / 10 / 12 over 25 | 5.06 s at move 116 | 7785 ms | +1 ms |
| B | black | win by checkmate | 56 (to move 61) | 8 / 11 / 14 over 25 | 24.98 s at move 56 | 6078 ms | -210 ms |
| C | white | win by checkmate | 39 (to move 45) | 10 / 10 / 13 over 39 | 21.43 s at move 40 | 10694 ms | +1 ms |
| C | black | draw, insufficient material | 80 (to move 85) | 8 / 10 / 11 over 25 | 7.16 s at move 81 | 9719 ms | +1 ms |

All three candidates scored 1.5/2. **Over all 463 of our moves in the six games the worst
exceeded its hard budget by 1 ms, and none reached a quarter of its clock**; the handful of moves
that read as "past hard" are 0.4 to 1.4 ms over on a figure derived from the PGN, which includes
the harness round trip. A holds v4.1's depth, which is what a pure time change should do; B and C
are two to three plies deeper over the same opening moves, which is the reductions.

All six games start from `harness.play`'s default opening, so they are matched, and the clock
trajectory separates the ceiling where the Elo cannot. Our clock in seconds after our move N:

| cand | side | our moves | m20 | m30 | m40 | m60 | m80 | minimum | at move |
|---|---|---|---|---|---|---|---|---|---|
| A | white | 72 | 75.0 | 59.0 | 48.1 | 28.5 | end | 17.24 | 77 |
| B | white | 166 | 85.6 | 64.1 | 43.1 | 26.6 | 11.6 | 5.06 | 116 |
| C | white | 39 | 58.0 | 36.3 | 21.4 | end | end | 21.43 | 40 |
| A | black | 50 | 91.4 | 67.9 | 51.0 | end | end | 40.26 | 52 |
| B | black | 56 | 84.8 | 63.7 | 44.1 | 26.7 | end | 24.98 | 56 |
| C | black | 80 | 54.3 | 35.3 | 25.7 | 11.8 | 7.2 | 7.16 | 81 |

**At matched move numbers C spends far faster than B, in both colours and from move 20 on**: by
move 40 C holds 21.4 s and 25.7 s where B holds 43.1 s and 44.1 s. The ceiling is worth about
20 s of clock by move 40, which is what it was written to do. B's 5.06 s minimum is not B
spending faster; it is a 166-move game, and at every matched move number B is the most
conservative build in the table.

B's white game is the one to read: 172 moves, terminated by the fifty-move rule in a dead-drawn
same-colour bishop ending that both sides shuffle from about move 85. Its clock reads 43.1 s after
move 40, 26.6 s after move 60, 11.6 s after move 80 and 6.4 s after move 100, bottoms out at
5.06 s on move 116 and recovers to 6.9 s by the end. The 1.5x ceiling bound on 78 of our 166
moves, every one down to a clock of about 9.1 s, below which `budgets` clamps soft to hard and the
ceiling is inert. What drains the clock is not the ceiling but that `soft = clock/25 + 400` has no
floor against the increment: at a 6 s clock the soft budget is 640 ms against a 500 ms increment,
so a long game settles at a fixed point near 5-7 s. v4.1 does the same in a 170-move game;
`time/reserve-floor` in `docs/VERSIONS.md` is the candidate that addresses it.

## Cycle 5 (M5), 2026-09-10: a floor under the soft budget (time/soft-floor)

Off `prod` at v4.2, one change against `local-opponents/v4.2`. The previous section ends on the
problem: `soft = clock/25 + 400` has no floor against the 0.5 s increment, so at a 6 s clock the
engine plans 640 ms for a move the increment pays 500 ms of, and a long game settles at 5-7 s
whatever the build. Rated games have run to 90 moves, so this is insurance against a flag, not
Elo; it ships only if it costs no strength.

The change: the soft budget is taken from the clock above a reserve of `RESERVE_DIVISOR` (8) of
the **first clock of the game** -- 15 s at the platform's 120 s, 5.6 s at the 45 s proxy, 1.25 s
at 10 s. A fraction, not a fixed 10 s, because one build plays every control and a fixed reserve
that is 8% of the platform's clock is 22% of the proxy's (that is what `time/reserve-floor`
ace9ebc did). The first clock is inferred, because `get_move(fen, time_left_ms)` is told neither
the base clock nor the increment; `budgets` records it and `reset` clears it, and `observe`
already calls `reset` on any position that is not one legal move on from the one we handed back,
so a second game in the same process cannot inherit a stale reserve. Below the reserve a move
plans `SOFT_BONUS_MS` alone, 400 ms, under the increment, so the clock climbs back out. The hard
budget and the abort path are untouched.

### The replay, before any games were played

`pgn/stack-120s/B-white.pgn`, the 172-move drawn game v4.2 played as White, walked through
`fastsearch.think` position by position with the clock starting at 120 s, decremented by what
each move actually spent on this machine and credited 0.5 s per move. The moves played are the
game's, so every variant sees the same 166 positions and only the clock differs. The M5 is about
2.5x the platform's core, so the depths are higher than a platform game's and the clock curve is
the shape to read, not the absolute depth.

| budgets | m40 | m60 | m80 | m100 | m120 | m160 | minimum | depth first 40 min/med/max | spend first 40 |
|---|---|---|---|---|---|---|---|---|---|
| prod (v4.2) | 49.3 | 28.1 | 14.8 | 8.8 | 6.5 | 5.6 | **5.30 s @m149** | 9 / 10 / 14 | 104.4 s |
| reserve = first/12 | 60.6 | 39.3 | 24.7 | 17.2 | 13.8 | 12.5 | **11.64 s @m152** | 9 / 11 / 14 | 89.1 s |
| reserve = first/8 | 56.8 | 37.7 | 27.7 | 23.8 | 19.7 | 17.5 | **16.91 s @m156** | 8 / 10 / 14 | 90.1 s |

The prod row reproduces the real game (5.06 s at move 116 there, 5.30 s at move 149 here), which
is what says the method is sound. /12 floors at 11.6 s, under the 15 s target, so the fraction was
tuned once to /8: 3.2x prod's minimum at prod's median depth over the first 40 moves, for 14% less
time spent there. That is the trade -- the reserve is depth given up early, and early depth is
worth less than a move that is not made on a 5 s clock.

### The rows

| run | opponent | control | games | +=- | score | Elo | 95% | ill/exc/tmo/over | worst | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| time-soft-floor | v4.2 | 45s+0.2s | 48 | +19 =13 -16 | 53.1% | +22 | -64 to +110 | 0 / 0 / 0 / 0 | 5.63s | 258 MB |

Four 120 s + 0.5 s games against v4.2, one at a time, clocks read from the PGN rather than the
log. Two openings: the English the replayed game came from, and the Petroff, picked as a
symmetrical line likely to run long.

| game | our colour | result | moves | our clock m40/60/80/100 | our minimum | past hard | our depth first 40 | theirs |
|---|---|---|---|---|---|---|---|---|
| English | white | 1-0 us, mate | 102 | 53.0 / 37.4 / 26.4 / 22.9 | 22.05 s @m98 | 0 | 7 / 10 / 14 | 7 / 10 / 14 |
| English | black | 1-0 them, mate | 86 | 51.7 / 37.1 / 26.3 / end | 25.22 s @m82 | 0 | 7 / 12 / 16 | 7 / 12 / 15 |
| Petroff | white | 1/2-1/2 threefold | 49 | 57.2 / end / end / end | 48.29 s @m47 | 0 | 7 / 10 / 14 | 7 / 10 / 14 |
| Petroff | black | 1-0 them, mate | 75 | 57.7 / 39.0 / end / end | 33.86 s @m67 | 0 | 7 / 12 / 16 | 7 / 11 / 16 |

One win, one draw and two losses over four games is noise; the 48-game row is the strength
signal. The clock columns are the point. No move anywhere past the hard budget, and the
comparable number is the English as White: 22.9 s at move 100, where v4.2's own 120 s game in the
same opening and colour held 6.4 s and bottomed at 5.06 s.

**What these four games do not show.** None of them ran long enough to reach the reserve -- 49 to
102 moves against the replayed game's 172, and the lowest clock in the set is 22 s. So they say
the change costs no depth (the depth distributions are v4.2's on the same board) and no
over-budget move; the offline replay above is what says the floor holds in a game that actually
gets long. The 48-game proxy row's lower bound is -64, below the -40 the brief asked for, on a
positive centre: a 48-game row at exactly 50.0% has a lower bound near -86, so that is the
interval's width and not a measured loss.
