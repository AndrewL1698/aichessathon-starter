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
