# How we work on this entry

The repo's `AGENTS.md` is the platform contract. `docs/STRATEGY.md` is the research brief and
phased plan. This file is the team's working conventions and the live status. It is in the repo
so every clone, machine and Claude session gets the same context; keep it current here, not in
a local file.

## Hard dates

- Upload deadline **Thu 2026-09-11 11:00 UK**. The latest validated upload plays.
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
The frozen 2-ply agent is `local-opponents/prod` (29e6dc1; 1151aa3 on `prod` changed docstrings
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
- Reference opponents by path from any worktree: `local-opponents/prod`, the previous version's
  worktree, `local-opponents/sunfish` (or `../opponents-tooling/local-opponents/sunfish` before
  PR #1 merges), `baselines/minimax`, `baselines/random`.
- A laptop core is ~1.5–2× faster than the platform's EPYC core. Budgets are clock-relative;
  after the first upload, scale by the validation log's real slowest move.
- Before any upload: `make gate`, 60+ games vs random at 3 s plus 16 at 2 s with zero failed
  terminations, `make zip` smoke passes.

## Status (newest first; update this in the same PR or a docs commit)

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
