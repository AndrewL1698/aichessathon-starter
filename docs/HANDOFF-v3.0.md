# Prompt for the v3.0 session

Paste everything below this line into a fresh Claude Code session opened in
`/Users/valstrm/aichessathon-starter`.

---

You are continuing work on our AI Chessathon entry, team "2 Pawns and a Queen". Read these
before doing anything, in this order, and keep them open: `CLAUDE.md` (the platform contract:
what breaks agents, what is banned), `docs/BRIEF.md` (where we are and why, written for a
non-expert; section 8 is the plan you are executing), `docs/VERSIONS.md` (what each shipped
version is), `docs/LOGBOOK.md` Part 1 (how the engine works, function by function) and Part 2
(every change tried, including the rejected ones and why), `docs/BENCH_LOG.md` (every
benchmark table), and `docs/TEAM.md` (team conventions). Then `git fetch origin` and look at
`git log --oneline v2.3..origin/prod`: two teammates push to `prod` without notice, and the
last time that happened it was a 520-line evaluation change. Base everything on the fetched
`origin/prod`.

## Where things stand

- Shipped: v2.3 (8cc4670, time manager fix) won round 75. v2.4 (1077652, the tapered
  evaluation) is benched at +273 Elo over v2.3, lower bound +153, built as
  `submission-v2.4.zip`; check `docs/VERSIONS.md` for whether it has been uploaded. Rated
  games are hourly 08:00 to 22:00 UK; the upload deadline is Thursday 2026-09-11 11:00 UK,
  and a 13-round Swiss on the locked builds follows that afternoon.
- The engine is `agent.py`: python-chess objects, 27k nodes/s on the platform (0.38x this
  laptop), depth 4 to 6. The blunders in our rated games (8, 4, 5 per game) are 2-to-4-ply
  tactics beyond that depth. Cycles 1 to 3 measured null move, LMR, futility, three time
  changes and checks in quiescence: all within noise of zero. The python-chess engine is
  squeezed. Depth is speed, and speed is the compiled board.
- `fastboard.py` is our numba mailbox board: move generation, make/unmake, attacks,
  Zobrist keys, eager `@njit` signatures so it compiles at import (1.8 s locally), verified
  against python-chess on 12,000 positions and every published perft. It ships in every zip
  unused. Its docstring documents the array layout and the two invariants callers own.
- Tooling: `harness/bench.py` (the gauntlet: paired openings vs the frozen baseline,
  Sunfish and minimax; Elo with 95% interval; illegal/exceptions/timeouts/over-budget
  counts, any nonzero disqualifies; peak RSS; `--jobs 4`; `--base-ms 45000 --increment-ms
  200` is the platform-speed proxy for 120 s), `harness/readlog.py` (reads a competition
  log), `tools/analyse_game.py` (local Stockfish review, offline only, never shipped),
  `tests/positions/run.py` (17 real blunders; measures sharpness, not strength; decide on
  Elo). Frozen builds live in `local-opponents/vX.Y/`; the bench baseline is the newest
  shipped version. Sunfish needs `local-opponents/fetch_sunfish.sh` once per machine.
- Worktrees from earlier cycles may exist beside the repo (`../cand-*`, `../time-*`,
  `../search-*`, `../phase0-clock`, `../prod-1077652`). They are finished; ignore or remove.
  PR #9 (`tooling/bench` into `prod`, docs and tooling only) may still be open.

## Rules, all of them non-negotiable

- Anything that reaches GitHub and could ship goes through a pull request into `prod` with
  a description that says **what** the change is, **why** (the evidence), and **how** it works,
  plus the bench numbers (Elo, interval, game count, the four disqualifier counts, peak RSS).
  Comment on the PR as later results land. Never merge it yourself; the user merges. Never
  touch `main`, never open a PR against `main`, never force-push, rewrite or squash. If a
  merge into `prod` would conflict, stop and show the conflict.
- Rejected candidates get no PR: write them into `docs/LOGBOOK.md` Part 2 with the numbers
  and the reason. Competition logs dropped into `logs/` get committed and pushed straight
  to `prod` with no PR, renamed `<round>-<opponent>.log/.pgn`, then read with `readlog`,
  reviewed with `analyse_game`, real blunders appended to `tests/positions/positions.epd`.
- One isolated change per candidate branch, named for the change. Bench every candidate
  against the frozen shipped build; re-run the best from scratch; promote only when the
  Elo lower bound is above zero and the disqualifier counts are zero. Shipping nothing is
  normal. Report losses plainly; negative results are the most useful part of the log.
- Versions: every uploaded build is `vX.Y` (v3.0 is the compiled search), an annotated tag
  on the exact commit, a frozen copy in `local-opponents/vX.Y/agent.py`, a row in
  `docs/VERSIONS.md`, and a zip named `submission-vX.Y.zip`. The user uploads, not you,
  unless they ask for a zip path under deadline; then build it, inspect it (agent.py at the
  zip root, under 50 MB, no .so/.pyd/.dll/.dylib, no tools/tests/logs/local-opponents),
  smoke it with `make zip`, and hand over the path.
- Do not edit `harness/` (new files there are fine), do not read `HARNESS_SEED` in the
  agent, no new dependencies, nothing that is not in the platform's stack (torch, numpy,
  python-chess, onnxruntime, numba). Stockfish is never imported, invoked or referenced by
  anything in the zip. Keep `agent.py` readable: a judge reads it.
- Before any major change or analysis, fetch and re-read `origin/prod`. Keep
  `docs/LOGBOOK.md`, `docs/BENCH_LOG.md` and `docs/VERSIONS.md` current as you go, not at
  the end.

## The job: v3.0, the search on the compiled board

Goal: the v2.4 engine's behaviour on `fastboard.py` at 5 to 20 times the node rate, so
depth 7 to 8 at 120 s locally and one ply less on the platform. It ships only if it beats
v2.4 on the bench with the lower bound above zero and zero disqualifiers. v2.4 stays the
shipped build until then. Work on a branch `v3/compiled-search` off `origin/prod`; commit as
you go. Build in this order and do not skip a gate:

1. **Evaluation on the compiled board.** Port v2.4's `evaluate` (tapered tables, pawn
   structure, king shield, mop-up) to a `@njit` function with an eager signature over the
   `board`/`st` arrays. Gate: exactly v2.4's integer on 10,000 random positions, generated
   the way `tests/test_fastboard.py` generates them, compared against the Python `evaluate`.
2. **Search on the compiled board.** Negamax alpha-beta, quiescence on captures and queen
   promotions, MVV-LVA, killers, history, a transposition table as fixed numpy arrays
   (keys, depth, bound, score, move; a few million entries, around 64 MB) indexed by `st[7]`,
   repetition detection from an array of the game's keys plus the current path, fifty-move
   and insufficient-material draws with contempt, mate scores by ply. All `@njit`, all
   buffers preallocated (see `MAX_MOVES`, `MAX_PLY`, `UNDO_SIZE`), nothing allocated per
   node. The search stops on a node count, not the clock: numba cannot read time. Gate: at
   equal fixed depth with the table and killers disabled, the same best move and score as
   the Python search on a few hundred positions (use `_root` from `agent.py` the way
   `tests/positions/run.py` does); that proves the search is correct rather than fast.
3. **The Python wrapper in `agent.py`.** Keep v2.3's time management exactly (`_budgets`, the
   gate in `_think`, `PANIC_MS`, the safety margin) and turn the soft and hard budgets into
   node budgets from the node rate measured over the previous iterations; iterative
   deepening stays in Python so the previous depth's move is always available. Before
   returning, verify the move is legal with `python-chess`. Keep the entire v2.4 search in
   the file as the fallback: if anything in the compiled path raises or returns an illegal
   move, play the v2.4 move instead, and print a line saying so. Print the same per-move
   log line format as v2.4 plus a `nps` that the log reader already parses.
4. **Warm-up at import.** Every jitted function compiles at import, with a short search
   from the start position so no signature compiles on the clock. Measure the import time
   locally and treble it: it must stay well inside 90 s. If it does not, reduce what is
   jitted rather than lazily compiling.
5. **Verification, stopping at the first failure:** evaluation parity; search parity;
   `tests/test_fastboard.py`; 200 fast games against v2.4 with zero illegal, crash, flag or
   over-budget counts (`harness.bench --candidate . --baseline-games 200 --sunfish-games 0
   --minimax-games 0 --jobs 4`); two full 120 s + 0.5 s games (`harness.play`) for clock
   trajectory, peak RSS under 1 GB, and the log line; the gauntlet with intervals; the
   45 s + 0.2 s proxy; the regression suite; `make zip` and its smoke; then the PR into
   `prod` with all of it in the description, and the logbook entry.

Expect the search to be around ten times faster, not a hundred: a node still costs the
evaluation, and python-chess's 40 µs was never all move generation. Report the measured node
rate and the depth reached at 120 s against v2.4's 5 to 6 on the same positions
(`harness.rules.OPENINGS`, one move each at a 75 s clock, is the comparison already used).

## After v3.0, in priority order

Only after v3.0 has passed its gates or been honestly abandoned with the reason logged:

1. 3- and 4-man Syzygy tablebases, read with `chess.syzygy` (in the platform stack), within
   the 50 MB cap. Measure the draw and win rates in endings.
2. A larger blunder suite from the Lichess puzzle database (measurement only).
3. **Dynamic draw weighting (low priority, the user's idea).** Today `_contempt` scores a
   draw as -50 for us when the root evaluation is above +150 and +50 when below -150, and 0
   in between. Make it a function of how far ahead or behind we are (for example scaling
   with the score up to a cap of about a pawn and a half), and test whether it should also
   depend on the game phase or the clock. A draw is worth more the further behind we are and
   the fewer ways the opponent has to convert; the 600-ply cap, the fifty-move rule and
   stalemate are all draws that simple bots fall into. Measure it two ways: the bench Elo
   against v2.4 or v3.0, and the draw rate from losing positions (start games from
   positions the engine evaluates at -300 or worse, against the baseline and Sunfish). Do
   not try to detect the opponent's strength: the agent sees only a FEN and a clock. One
   change per candidate; this is an evaluation-of-draws change and nothing else.

Start by reading the documents, fetching `prod`, and writing a one-paragraph statement of
what you are about to do. Then build step 1.
