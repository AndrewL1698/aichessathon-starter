# How we work on this entry

The repo's `AGENTS.md` is the platform contract. `docs/STRATEGY.md` is the research brief and
phased plan. This file is the team's working conventions and the live status. It is in the repo
so every clone, machine and Claude session gets the same context; keep it current here, not in
a local file.

## Hard dates

- Upload deadline **Fri 2026-09-11 11:00 UK (verified against aichessathon.com/docs on 2026-09-10: "Uploads close 11 September 11:00")**. The latest validated upload plays.
- Final qualification: 13-round Swiss on locked builds that afternoon. The hourly rated rounds
  (08:00–22:00) only seed it.
- London final Sep 12 needs a UK university student on the team. Eligibility unconfirmed.

## Set up on a new machine

```
git clone https://github.com/AndrewL1698/aichessathon-starter.git
cd aichessathon-starter
git checkout prod
uv sync
local-opponents/fetch_sunfish.sh     # GPL, never committed, never shipped
mkdir -p ../pgn
```

Until PR #1 merges, `local-opponents/` lives on branch `tooling/local-opponents`:
`git worktree add ../opponents-tooling tooling/local-opponents` and run the fetch script there.
The frozen 2-ply agent is `local-opponents/v1.0` (29e6dc1; 1151aa3 on `prod` changed docstrings
only). One worktree per branch being worked on; branch names are the PR list below.

The directory is hyphenated on purpose: `harness/package.py` zips any root directory a root-level
module imports by name, and no `import` statement can name `local-opponents`, so the GPL engine
can never reach `submission.zip`. `local-opponents/` is outside the mypy gate.

Branches: `prod` is what the platform plays; `main` is the untouched starter. The remote is a
fork of `advitrocks9/aichessathon-starter`, so `gh pr create` needs
`-R AndrewL1698/aichessathon-starter --base prod` or it opens PRs upstream.

## Workflow

- **Orchestrator and doers.** One session writes briefs with numeric success criteria, reviews
  diffs, runs the final benchmark pass, and opens PRs. Implementation runs in subagents
  (`model: opus`), one per PR, each in **its own worktree**. Two agents never edit the same file
  at once; PRs that touch `agent.py` run one after another.
- **One variable per PR.** Search, then memory, then evaluation, then numba. A search PR leaves
  `evaluate` and the tables byte-identical, and vice versa, so the arena delta is attributable.
- **Subagents commit locally and never push or open PRs.** The orchestrator does, after review.
- **No AI attribution in git history:** no `Co-Authored-By` trailers, no "Generated with
  Claude" footers, on commits or PR bodies. Commit style is the repo's: `feat: ...`,
  `fix(harness): ...`, imperative, lowercase.
- **Do not edit `harness/`.** It mirrors the platform; editing it makes local results meaningless.
- **Do not read `HARNESS_SEED` in `agent.py`.** The platform does not set it.
- Keep `agent.py` readable. A judge reads it if games are flagged.

## Benchmarking rules

- Arena jobs run **one at a time**; concurrent games break time measurement. Always pass
  `--pgn-dir ../pgn/<run>` and break terminations down by side before trusting a score.
  `flag`, `illegal` or `crash` on our side is priority zero regardless of score.
- Two deterministic engines replay identical games across 8 openings × 2 colours, so 32 games
  is 16 unique. Run such matchups at two controls (`--increment-ms 100` and
  `--base-ms 20000 --increment-ms 200`) to double the sample.
- Ladder anchors (Round 60, 2026-09-08): house Random 802, Greedy 939, Minimax Two
  (= `baselines/minimax`) 1064, Sunfish 1465 (rank 241 of 377). Rank 50 cutoff ~1983,
  median 1576. Local scores against these convert to ladder Elo.
- Reference opponents by path from any worktree: `local-opponents/v1.0`, `local-opponents/v2.2`, `local-opponents/v2.3`, the previous version's
  worktree, `local-opponents/sunfish` (or `../opponents-tooling/local-opponents/sunfish` before
  PR #1 merges), `baselines/minimax`, `baselines/random`.
- A laptop core is ~1.5–2× faster than the platform's EPYC core. Budgets are clock-relative;
  after the first upload, scale by the validation log's real slowest move.
- Before any upload: `make gate`, 60+ games vs random at 3 s plus 16 at 2 s with zero failed
  terminations, `make zip` smoke passes.
- **Never `SIGSTOP` a running arena, and never resume one.** The referee times every move
  against the wall clock, so a suspended process is charged the whole pause and the in-flight
  game records a `flag` -- the termination this list calls priority zero, manufactured out of
  nothing. It cost cycle 6 a fake disqualifier that survived into a scored row until the PGN
  clocks were read. To pause a run, kill it: the completed PGNs are the state, and a run
  resumes from its PGN directory by index because the arena's game order is a pure function of
  the index. Resuming from disk is safe; suspending the process is not.
- Pinning both sides matters when the difference is small. Resolve each side to a commit and
  play out of separate **detached** worktrees, so neither can move mid-run and the diff between
  them is auditable afterwards. `harness.bench` plays 28 openings (`BENCH_OPENINGS`) against
  `harness.arena`'s 8, so a 56-game bench run is the full set at both colours and is the wider
  sample when a result on 8 openings looks like it might be an opening-set artifact.

## Status (newest first; update this in the same PR or a docs commit)

- 2026-09-10 evening · **Rejected** `time/v4.3-soft-overrun-2` (7b0fd82): `SOFT_OVERRUN`
  1.5 -> 2.0, the iteration gate's ceiling. It was built and benched on **v4.3**, which was
  `prod` at the time; `prod` has since moved to v4.4 (`eval/bare-endgame-blend`), and the
  rejection stands either way because v4.4 changed the leaf past the bare-endgame line and not
  the clock. **41.2% over 40 games at the 45 s + 0.2 s proxy, Elo -61, interval -158 to +27**
  against frozen `local-opponents/v4.3`. **That 41.2% is the 40-game proxy result and not a
  pooled 80-game number**: the same candidate scored 51.2% (+9, -79 to +97) over 40 games at
  10 s + 0.1 s, and the two are deliberately not pooled, because v4.3's reserve leaves this
  ceiling binding only above a 10.0 s clock at the fast control -- which is where that control
  starts. 0 illegal, 0 exceptions, 0 timeouts, 0 over-budget moves in all 80 games; calibration
  1.204M nodes/s before the proxy arena and 1.211M after, so the arenas are comparable.
  **The 200-game proxy extension was declined on purpose**, and that is a decision rather than
  an omission: the interval does not prove the candidate weaker, but the sign is negative at the
  only control where the change acts, the diagnostics confirm the change did take effect
  (utilisation 0.65 -> 0.85 of the soft budget, +0.6 ply of mean depth), and eight more hours of
  arena the night before an upload deadline was the wrong trade. **`SOFT_OVERRUN` stays 1.5 in
  prod and no version or tag was created for a rejected code change.** The branch is pushed and
  kept, as are every PGN and log; cycle 9 in `docs/BENCH_LOG.md` has the full experiment,
  including the clue for anyone who revisits it -- the candidate ends proxy games on a mean
  5.77 s clock against v4.3's 9.19 s, so it buys its depth out of the endgame.
- 2026-09-10 afternoon · **v4.4** = v4.3 + PR #25 (`eval/bare-endgame-blend`, from the M5 session, merged
  on Neil's instruction): past the 3-man line the leaf is the blend plus one mop-up term, except pure pawn
  endings, which stay hand-only; the round-102 seam (a mate-in-14 shuffled for 47 moves) is closed and all
  five conversion endgames still mate. 45 s proxy vs v4.2 60.4% (+73, -1 to +155), 0 disqualifiers; two
  120 s games clean. Frozen at `local-opponents/v4.4`, tag `v4.4`; **this is the upload**. Still open on
  the laptop: `eval/blend-weight` (net 2:1) benching vs v4.3; on the M5: the e36 net's pooled 10 s run.
  Neither holds the submission.
- 2026-09-10 midday · **v4.3** = v4.2 + PR #24 (`time/soft-floor`, from the M5 session): a reserve of
  the first clock / 8 kept out of the soft budget so long games floor near 17 s instead of 5 s. Proxy
  within noise (+22), four 120 s games with minima 22 to 48 s, 0 disqualifiers. Frozen at
  `local-opponents/v4.3`, tag `v4.3`; **upload candidate**. In flight on the M5: `eval/bare-endgame-blend`
  (round 102: the 3-man handover seam stalled a mate-in-14 for 47 moves; policy: blend + mop-up past the
  seam except pure pawn endings; five conversions green) and the e36 net (+55 at 10 s, +51 at 45 s,
  neither lower bound above zero; 200 more games pending).
- 2026-09-10 · **Built, not benched** `nnue/king-buckets-4` (011e75c, worktree
  `../nnue-king-buckets-4`, branched from `prod` afb34f4): four king buckets, 3,072 inputs, the
  experiment cycle 7's rejection pointed at. **Deliberately not HalfKAv2_hm** -- four buckets,
  not 32, no king-file mirroring -- because a 32-bucket scheme divides the same corpus 32 ways
  and a bucket with too few positions learns noise. `docs/NNUE_KING_BUCKETS.md` is the spec;
  cycle 8 in `docs/BENCH_LOG.md` has the numbers. `fastsearch.py`, `fasteval.py` and `agent.py`
  are byte-identical on the branch, so search, blend, time management, the UCI reply and the
  python-chess fallback are v4.1's exactly. **Speed gate passed: 3.4% of the node rate**
  (1.341M against 1.390M at depth 7) at **identical node counts**, which is what makes it a
  clean A/B -- a warm-started net evaluates identically, so the tree is the same and only the
  time differs. Exactness: 11,000 positions return the 768 net's integer, 11,000 exact against
  the reference, 30,000 randomised make/unmake sequences with 3,175 bucket crossings and 54
  crossing castles, and all of it repeated on a net whose four blocks differ, because a
  warm-started file cannot tell a bucketing bug from a correct bucketing. **No strength claim
  exists and none can until a net is fine-tuned on bucketed shards** -- the blocks are copies,
  so a bench today scores 50% by construction, and the init line says so out loud to stop a
  stray row being read as a result. Next, in order: rebuild shards (old ones carry 768 indices
  and are now refused by design), warm start with `tools/nnue/bucketize.py`, fine-tune, export,
  bench against `local-opponents/v4.2` at both controls. **The branch is based on afb34f4,
  which is v4.1**: v4.2 (PVS, late move reductions, the spend ceiling) landed on `prod` after
  it and changed only `fastsearch.py`, which this branch does not touch, so it merges up
  cleanly -- but merge it up before benching, and re-take the 3.4% on v4.2's search, because
  reductions change how often a king move is searched at all. Whoever trains it should know
  `densify` now builds a 201 MB dense batch at batch 16384, against 50 MB before; drop the batch
  size before anything else. **The 32-bucket version does not start unless that bench shows a
  credible gain that outweighs the 3.4%.**
- 2026-09-10 · **Rejected** the 163M-position 768-input net (`nnue-h256-100m-e71.npz` on
  `nnue/weights-v1`), benched as the last read on the 768 architecture before anyone starts on
  king-relative features. Code identical to `prod` afb34f4 = v4.1; the weight file is the only
  variable. **44.8% pooled over 144 games** vs `local-opponents/v4.1` (43.8% over 96 at
  10 s + 0.1 s, Elo -44, -106 to +16; 46.9% over 48 at 20 s + 0.2 s, -22, -95 to +50), both
  controls below 50%, **0 / 0 / 0 / 0 across all 214 games played in the cycle**. It is not a
  speed question: 1.367M nodes/s at depth 7 against 1.363M, suite 26/47 against 27/47, import
  4.24 s, peak RSS 254 MB. **The finding to carry forward: validation loss did not order strength
  even within one policy and one architecture.** The candidate wins every offline number (-6.1e-4
  validation loss, qa=512, the most precise export the project has made) and loses on the board;
  the epoch-11 file from the same run had already benched at parity over 96 games, so that is two
  files and 240 games from the 163M run with neither ahead. The pooled upper bound of +10 Elo is
  what settles it against spending an upload. **The blend was re-confirmed on the new weights:**
  net alone 28.1% and hand alone 21.9% over 32 games each, about +119 and +177 Elo for the blend.
  Method note worth keeping: both builds were frozen into `chmod a-w` directories under
  `~/Documents/bench-snapshots/2026-09-10-nnue768/` and their checksums re-read after the last
  game, so what was measured is provably what was snapshotted, and every game ran out of a
  read-only directory, which is the read-only-filesystem check for free. PGNs under
  `~/Documents/pgn/2026-09-10-*`; the numbers are cycle 7 in `docs/BENCH_LOG.md`. **What this
  leaves for the king-relative work:** more data on the same 768 features has now failed twice,
  which is the argument that the feature set is the binding constraint. v4.2 landed after this
  run and left `weights/nnue.npz` byte-identical, so the verdict still names the file that
  ships.
- 2026-09-10 afternoon · **rounds 91 to 96 were v4.1-contempt** (the 01:23 zip), not the PVS/LMR
  `v4.2`: 2 W 1 D 3 L, the three losses to opponents at ACPL 17 to 20 on depth-6/7 moves with 25 to
  98 s on the clock; the contempt change decided no move (checked at fixed depth on the two it
  coincided with). Do not go back to v4.0; upload the `v4.2` tag (PVS/LMR, +143 vs v4.1). The
  `submission-v4.2.zip` in the repo root is NOT that build. Eleven positions into the suite.
- 2026-09-10 morning · **v4.2** = v4.1 + PR #22 (`stack/v42`: PVS, LMR with no reduction at a PV node,
  iteration gate capped at 1.5x soft). Benched on the idle M5 vs v4.1: 69.5% at 10 s over 200, 69.8% at
  the 45 s proxy over 48, 0 disqualifiers in 696 games; audited (PVS exact over 1,120 searches; the cap
  raises the 120 s clock minimum from 9.3 s to 16.7 s in the round-85 replay). Frozen at
  `local-opponents/v4.2`, tag `v4.2`; **this is the upload for the Friday 11:00 UK cutoff**. Bench
  baseline is v4.2. Held: PR #16 null move, PR #18 hard divisor. Next candidates: a soft-budget floor
  against the increment (long games settle at 5 to 7 s on every build), the warm/cold root tie
  instability, the LMR minimum-depth knob.
- 2026-09-10 01:30 · **v4.1-contempt** (built as `submission-v4.2.zip` before the tag went to PR #22; played rounds 91 to 96) = v4.1 + `eval/contempt-quiescence` (contempt read through quiescence,
  so a pending recapture no longer sets draw-seeking contempt in a level position; round 90). Built as
  `submission-v4.2.zip` from the branch (f05f0dd) because the merge into prod is the team's to make:
  PR open, tag `v4.2` goes on the merge commit. Proof: 96 vs v4.1 50.5%; vs v4.0 96 fast 52.1% and 48
  proxy 46.9% (50.3% pooled), two 120 s games 1-1 with worst move 12.5 s and clocks over 15 s, 76 vs
  random all mates, 0 disqualifiers, suite 27/47, smoke clean. Frozen at `local-opponents/v4.2`;
  bench baseline v4.2. Every change since v4.0 fixes a rated-game situation and is Elo-neutral by
  design; nothing on prod has an Elo lower bound above v4.0.
- 2026-09-10 · **Rejected, PR #19 closed unmerged** `eval/mobility`: knight, bishop, rook and
  queen mobility in both evaluations. **47.3% pooled over 128 games** against v4.0-plus-rook-pawn
  baselines (53.1% on 64 at 10 s, 39.1% on 32 at the 45 s proxy, 43.8% on 32 at the 28-opening
  bench head-to-head); only the narrowest opening set was above 50%. **Zero disqualifiers in all
  128 games** -- the code is sound and exact against `agent.py` on 10,000 positions, so this is a
  strength rejection, not a robustness one. The head-to-head is what closed it: the term costs
  7.5% of the node rate and gives back a 6.5% smaller tree, so **it reaches the same median depth
  7.0 and still loses ground**, which rules out the cost as the explanation and leaves the term
  itself. The reading is that the net already encodes mobility -- easy for a piece-square net to
  learn -- so a hand term duplicating it re-states at half weight what `(hand + net) // 2`
  already has, and pays full price. Worth remembering before the next hand-authored term on top
  of a learned evaluation. Branch kept, code not merged; `docs/BENCH_LOG.md` cycle 6 has the
  rows, the telemetry and the `SIGSTOP` artifact that produced a fake flag.

- 2026-09-09 night · **cycle 5**, from the rounds 73 to 90 review (draws from won positions: 1 in
  17; no repetition ever taken while ahead; contempt read a pre-quiescence static). Off prod
  2354ff5, bench baseline v4.1. `search/check-extension`: 55.7% fast, 57.3% proxy, 49.0% re-run,
  pooled 53.3% (+23, -14 to +61), suite 26/47 vs 27/47, **not proven**, pushed for a PR.
  `eval/contempt-quiescence`: contempt read through quiescence, so a pending recapture no longer
  sets draw-seeking contempt in a level position; 50.5%, a correctness fix judged on the
  mechanism, pushed for a PR.
- 2026-09-09 evening · **v4.1** = v4.0 + PR #17 (KPK rook-pawn draw) + PR #20 (queen-first promotion
  tie-break, from the M5 session). Frozen at `local-opponents/v4.1`, tag `v4.1`. Bench baseline stays
  v4.0 for candidates already in flight; new ones use v4.1. Held: PR #16 (null move, proxy says no),
  PR #18 (hard divisor 6) until `time/platform-spend` reports its proxy rows, then one combined time
  candidate. In flight: cycle 4 PVS/LMR/futility; the 100M-position net (`nnue-h256-100m-*` on
  `nnue/weights-v1`), whose epoch-11 file is benching vs v4.0 now.
- 2026-09-09 morning · **v4.0**: PR #14 (`nnue/runtime`, the compiled learned evaluation, merged with
  the switch off after an independent audit; the one CRITICAL, a damaged weight file failing the
  import, fixed before merge) plus `nnue/v4.0`, which commits `weights/nnue.npz` and sets
  `USE_NNUE = True`. Leaf = (hand + net) / 2; residual nets and the net alone both measured worse.
  vs v3.2: 80.2% over 96 games (+243, +177 to +329), 85.4% at the 45 s proxy, two 120 s wins, no
  disqualifiers. Bench baseline is v4.0 (`local-opponents/v4.0`). Weights come from the M5 session
  (branch `nnue/weights-v1`, run notes in `tools/nnue/runs/`); the M5 is training a 256 net on 100M
  positions exported at qa=512 as the next candidate. Deferred audit findings, as candidates: the
  3-man handover is a step of up to ~880 cp in the leaf score; the HAND path costs ~9% more per
  node than v3.2. Next: search cycle 4 (null move, PVS, LMR, futility) on v4.0.
- 2026-09-09 · **In progress** `nnue/runtime`: `fastnnue.py`, the shipped half of the learned
  evaluation. numba inference of the exported integer weights, exact against
  `tools/nnue/nnue_ref.py` on 2,000 positions for all three weight files on `nnue/weights-v1`
  (h128 at qa512/qb512, h256 at qa512/qb512, h256-e87 at qa256/qb1024 — every scale and the
  hidden size are read from the file, none is hardcoded). Two perspective accumulators per ply,
  `push` before every `make_move` and nothing on the way back, checked against a from-scratch
  build over 10,000 make/unmake sequences including castling, en passant, promotions and null
  moves: zero mismatches. `uv run python -m tests.test_nnue [--full]`.
  **One deviation from the brief, on measurement.** The policy was to be "net plus `fasteval`'s
  mop-up term in mop-up positions". Played out that way, KRRvK drew by repetition and KPvK never
  promoted: the net scores every move in those within a few centipawns of every other (they all
  leave the same men on the board) and the mop-up term is worth at most 120 cp. So leaves past
  `fastnnue.bare_endgame` — either side down to a king and at most two other men, `fasteval`'s
  own bound — are scored by the hand tables outright, mop-up, drawish scaling and bare-minor
  zero included. All five bare endgames now convert and the playouts are byte-identical with the
  network on and off, which is the test.
  1.85M nodes/s at depth 7 against the hand evaluation's 2.41M, import 5.2 s with warm-up,
  peak RSS 251 MB. `USE_NNUE` switches evaluations, and with it off `tests.test_fastsearch`'s
  score equality against `agent.py` still holds exactly, so this PR changed the evaluation and
  nothing else. Missing or malformed weights fall back to v3.1's hand evaluation and the init
  log line names whichever is active. `torch` moved from `[project] dependencies` to an
  optional `nnue` extra (`uv sync --extra nnue`); nothing that ships imports it.
  **Weights are not committed on this branch** — `weights/*.npz` is gitignored with a comment
  saying why, and the orchestrator brings `nnue/weights-v1` in at merge time.
  **The result: blending the network with the hand evaluation is worth about 200 Elo.**
  `(hand + net) // 2` scored **75.0%, Elo +191, interval +109 to +298** over 64 games vs v3.2 at
  10 s + 0.1 s with the book on both sides. The *same weight file used alone* is 48.4%. The
  absolute rows are monotone in validation loss (0.01654 / 0.01541 / 0.01489 giving -106 / -49 /
  -11 Elo), so the offline metric's ordering is real. **The residual nets are a negative result at both
  widths:** best validation losses measured (0.01450 at h256, **0.01350 at h512**) and they
  played at 47.7% and **51.6%** — parity, ~180 Elo behind averaging a *worse* net with the hand
  evaluation. So validation loss orders the absolute nets correctly and does not order policies
  at all. The likely mechanism is that a residual is added at full weight so the net's noise
  comes with it, while the blend halves that noise against material; the cheap thing to try is
  `hand + net // 2`, one more branch in `leaf` and no retraining.
  Zero illegal, zero exceptions, zero timeouts, zero over-budget across 464 games and six weight
  files. Depth-7 nodes/s by width: h128 1.85M, h256 1.30-1.35M, **h512 0.94M — the only file to
  miss the 1.0M target**, about half a ply, and its row says the width bought nothing.
  **`USE_NNUE` still ships off**, so what plays is v3.2 exactly; flipping it selects the blend,
  which is the configuration that measured +191. Turning it on is the orchestrator's call
  together with which weight file ships. Ignore any 53.1% figure — that is the old 16-game
  disqualifier row, not a strength number; `docs/BENCH_LOG.md` has the seven real rows and
  reads the baseline column, since a row against v3.1 from a post-merge candidate measures the
  opening book as well as the evaluation (that is the 86.7% row, kept only as the confounded
  counterpart of the controlled 75.0%).
  Also here: four leaf policies behind one stats slot, weight files that declare `target='cp'`
  or `'residual'` (an unknown marker is refused, absent means absolute), and contempt reading
  the hand evaluation whatever scores the leaves — the 52M nets read a dead-equal position as
  +46 cp for whoever is to move, which cancels in negamax and did not cancel against
  `CONTEMPT_THRESHOLD`. The `bare_endgame` handover is a count of men, so no cp offset moves it.

- 2026-09-09 · **In review** `book/opening` (PR: opening book): `weights/book.bin`, a 24,479
  entry polyglot book (391,664 bytes) read by `chess.polyglot` before the search up to ply 20,
  weighted by master game counts, legality-checked, and committed to both engines' history
  through `fastsearch.remember_played`. Built offline by `tools/book/build.py` from 1,133,198
  over-the-board master games in PGN Mentor's 233 opening collections. **The Lichess masters
  explorer this was briefed on is dead** (401 on every route, two networks), which is why the
  corpus is PGN files; provenance still human games only, never our engine, as `AGENTS.md`
  requires. **Four of the eight sample openings the ladder publishes occur in no master game at
  all** (Petroff, Scotch, French Classical 0 games; English 3), so the book answers the start
  position (16 plies), Sveshnikov (9), Grunfeld (9), Sicilian Closed (5) and Winawer (3) and
  nothing else: expect it to fire in a minority of rated games. 32 games vs v3.1 42.2%,
  interval 26.3-58.1% so it includes 50%, 0/0/0/0 disqualifiers; the book move in a 120 s
  Sveshnikov game left the clock at 85.8 s against 75.4 s after move 10. Lookup 0.096 ms worst.
  Also found, and **not** fixed here because `harness/` is off limits: the numba search's log
  line has not matched `harness/readlog.py`'s `OUTPUT_LINE` since v3.0 (`tt 31%` and an extra
  `null` field), so the reader reports a total log gap on every v3.0+ game; the book's own line
  is written in the shape that still parses.

- 2026-09-08 late · **v3.1** `search/clock-backstop`: the timer-thread backstop from PR #11 on
  top of v3.0, with `STATS[EXPIRED]` read at every node, `nogil` on the three search functions,
  and the thread joined on exit (the PR #11 audit found `Timer.cancel()` cannot stop a callback
  whose sleep has ended). Tree unchanged; 16 fast games vs v3.0 clean of disqualifiers; the new
  `backstop` test stops depth-40 searches with the clock read disabled within 5 ms. PR #11 is
  superseded by v3.0 + v3.1 and can be closed.
- 2026-09-08 late · **PR #10 merged** to `prod` = **v3.0**: `fasteval.py` and `fastsearch.py`, the
  v2.4 evaluation and search compiled by numba over `fastboard.py`; `agent.py` plays the numba move
  after a python-chess legality check and keeps the python engine as the fallback. Equality-tested
  (same integer evaluation on 10,000 positions, same root score on 188 fixed-depth searches).
  1.2–3.3M nps, depth 6 at 10 s and 8 at 120 s; 93.8% vs v2.4 and vs Sunfish. Null move ships off.
  Audit HIGH fixed before merge (fallback resync via `fastsearch.remember_played`). Head-to-head
  vs the parallel port in PR #11: 44.9% for #11 over 144 games, interval −88 to +15, i.e. equal
  within noise; #11 is now conflicting and its extras (timer-thread clock backstop) are for a
  follow-up. Bench baseline is now v3.0; freeze `local-opponents/v3.0` from the merge commit.
- 2026-09-08 · **In progress** `phase1/movegen` (PR 6): `fastboard.py`, the numba mailbox core
  (10x12 int8 board, packed int32 moves, pseudo-legal generation, make/unmake, attack detection,
  incremental Zobrist in `st[7]`). No search and no evaluation yet. All 30 published perft counts
  match, plus start depth 6 and Kiwipete depth 5; 12,000 random positions agree with python-chess
  on legal moves, fen and key. 23.6M perft nodes/s is **bulk counting**; the search-shaped figure
  is 0.53M gen_legal + make/unmake rounds/s (25.6M legal moves/s), so expect low single-digit
  Mnps in a real search against python-chess's 85k. Import with warm-up 1.3 s, RSS 150 MB.
  `uv run python -m tests.test_fastboard [--full]`. Audited: the HIGH was two Zobrist keys for
  one position from an en passant criterion that differed between `make_move` and `to_fen`.
- 2026-09-08 · **PR #4 merged** to `prod`: table keyed on the transposition tuple (500k cap,
  ~270 MB), killers, history, repetition/fifty-move draws with contempt. Post-merge: 100% vs
  old prod (32–0), 84.4% vs the #2 search, KQvK/KRRvK/KPvK convert, 120 s opening moves 2–7 s.
  **Upload `submission-v2-memory.zip`** (supersedes v1) for the calibration read.
  Worktree layout on Neil's machine: main clone on `prod`; `../memory-agent` = frozen prod
  (previous version); `../prod-agent` = 29e6dc1; `../phase0-eval` = evaluation PR in progress.
- 2026-09-08 · **In progress** `phase0/eval` (PR 5): tapered hand-authored PSTs, mop-up
  (KRvK must convert), passed/isolated/doubled pawns, rook files, bishop pair, king shield,
  endgame stalemate check in quiescence. Target ≥ 55% vs Sunfish, ≥ 60% vs previous version.

- 2026-09-08 · **PR #2 merged** to `prod` (merge commit, since #4 is stacked on it) with the audit
  fixes: soft budget binds (120 s opening spend 15/13/11 s → 2.1/1.7/2.8 s, depth unchanged),
  sub-310 ms clocks no longer flag, queen-only quiescence promotions. 92.2% vs minimax, 89.1%
  vs prod, 43.8% vs Sunfish. **First calibration upload built:** `submission-v1-search.zip`
  (agent.py only, smoke games pass). Upload it, then read init time and slowest move off the
  validation log and scale budgets by the platform/local ratio.
- 2026-09-08 · **PR #4 audited**: no critical; HIGH fixed before merge: the table key was
  `hash()` of the transposition tuple, and CPython folds bits 61–63 of every bitboard onto
  bits 0–2 (f8/g8/h8 alias a1/b1/c1, ~41 effective bits). Now keyed on the tuple, cap 500k.

- 2026-09-08 · **PR #4 open** `phase0/memory` (stacked on #2): transposition table, killers,
  history, repetition + fifty-move as draws, contempt ±50 past ±150 cp. 96.9% vs prod with
  **zero** threefold draws (was 10/32); 75% vs the #2 search over 64 games; KQvK, KRRvK, KPvK
  now convert in self-play; KRvK still draws by fifty moves (needs a mop-up eval term). Sunfish
  flat at 46.9% ± 13.9%: the next Elo is in evaluation. Peak RSS 141 MB. Under audit.
- 2026-09-08 · **PR #2 audited**: no critical findings; search proven exact vs an unpruned
  reference. Fixing before merge: soft budget never binds (first three 120 s moves spent
  15/13/11 s), sub-310 ms clock clamp order, insufficient-material shortcut unreachable at
  depth 0, diagnostic accuracy. Branch `phase0/search-fixes`, will fast-forward #2.
- 2026-09-08 · **PRs #1 and #3 merged** to `prod` after audit. `opponents/` renamed
  `local-opponents/` so `harness/package.py` can never zip GPL Sunfish (an `import opponents`
  in any root module would have). `prod` also gained 1151aa3 (docstrings + .vscode).
- 2026-09-08 · **Next:** evaluation PR off `phase0/memory` once #2's fixes are merged into it:
  tapered PST, passed/isolated/doubled pawns, rook on open file, king shield, **mop-up**
  (king-to-edge + king proximity when the loser has only a king), stalemate check in the
  endgame. Then first platform upload from `prod` for timing calibration. Then numba, where
  the TT probe needs an incremental Zobrist key or it becomes the bottleneck.
- 2026-09-08 · Team's ladder bot name and UK-student eligibility still unconfirmed.
