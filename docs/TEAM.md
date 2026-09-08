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
git worktree add --detach ../prod-agent 29e6dc1            # frozen 2-ply agent, opponent
git worktree add ../opponents-tooling tooling/local-opponents   # until PR #1 merges
../opponents-tooling/opponents/fetch_sunfish.sh              # GPL, never committed, never shipped
mkdir -p ../pgn
```

After PR #1 merges, `opponents/` is on `prod` and `opponents/fetch_sunfish.sh` runs from the
main clone. One worktree per branch being worked on; branch names are the PR list below.

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
- Reference opponents by path from any worktree: `../prod-agent`, the previous version's
  worktree, `opponents/sunfish` (or `../opponents-tooling/opponents/sunfish` before PR #1
  merges), `baselines/minimax`, `baselines/random`.
- A laptop core is ~1.5–2× faster than the platform's EPYC core. Budgets are clock-relative;
  after the first upload, scale by the validation log's real slowest move.
- Before any upload: `make gate`, 60+ games vs random at 3 s with zero failed terminations,
  `make zip` smoke passes.

## Status (newest first; update this in the same PR or a docs commit)

- 2026-09-08 · **PR #1 updated** (`50daf57`): Sunfish wrapper no longer flags at fast controls
  (budget bonus proportional to the clock, 1 s reserve, deadline handed to Sunfish early). Zero
  flags on either side across 64 games; 96.9% vs minimax unchanged.
- 2026-09-08 · **PR #2 open** `phase0/search`: iterative deepening, alpha-beta, quiescence,
  MVV-LVA, time management; eval unchanged. 95.3% vs minimax, 84.4% vs prod, 0 failed
  terminations in 172 games. **43.8% ± 13.0% vs Sunfish** (clean run, no flags): at the 1465
  anchor, not past it. 13 of 14 draws were threefold repetitions.
- 2026-09-08 · **PR #1 open** `tooling/local-opponents`: Sunfish wrapper, fetch script, frozen
  prod. Sunfish 96.9% vs minimax confirms calibration.
- 2026-09-08 · **PR #3 open** `docs/team-workflow`: this file and `docs/STRATEGY.md`.
- 2026-09-08 · **In progress** `phase0/memory`: transposition table, killers, history,
  repetition with contempt. Interim: threefold draws vs prod 10 → 0.
- 2026-09-08 · **Not started:** evaluation PR (tapered PST, pawn structure, mop-up), numba
  rewrite, first platform upload for timing calibration. Team's ladder bot name unknown.
