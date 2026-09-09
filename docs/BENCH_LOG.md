# Bench log

Every improvement cycle appends here. The frozen reference is `local-opponents/baseline`, a
byte-identical copy of `agent.py` at `a6c1fa6` (tag `build-20260908-a6c1fa6`), the build that
went to the rated round on 2026-09-08. Never edit it; freeze a new directory instead.

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
