# AI Chessathon — research and strategy brief

Written 2026-09-08 (Mon). Upload deadline **Thu 2026-09-11 11:00 UK**. Roughly 3 days of work remain.
Repo: https://github.com/AndrewL1698/aichessathon-starter (`main` = clean starter, `prod` = 2-ply agent, commit 29e6dc1; 1151aa3 later changed docstrings only).

---

## 1. The competition, as it actually is

| Item | Value | Source |
|---|---|---|
| Qualifier ladder | Sep 4–11, rated round **every hour 08:00–22:00** (15/day) | rules.md |
| What the ladder does | **Only seeds the final Swiss.** It is not the ranking that decides anything. | leaderboard page |
| Upload deadline | Sep 11 11:00; latest validated upload plays | rules.md |
| Final qualification | **13-round Swiss on locked builds, Sep 11 afternoon** → top 50 go to London | rules.md |
| London final | Sep 12, Encode Club. 50 seats, one per **UK university student**, max 2/team, filled in seed order | rules.md |
| Prizes | £1,000 / £500 / £250 | rules.md |
| Time control | 120 s + 0.5 s/move per side, wall time. Increment lands **after** the move. | agent-contract |
| Init budget | 90 s before the clock starts (import time) | agent-contract |
| Hardware | **One core** AMD EPYC 9V74 @ 2.60 GHz, 2 GB RAM, no network, no GPU | agent-contract |
| Stack | Python 3.12, torch 2.13 cpu, numpy 2.5, python-chess 1.11, onnxruntime 1.29, numba 0.67. **Nothing else installs.** | agent-contract |
| Zip | ≤ 50 MB unzipped, `agent.py` at root, source only (weights `.onnx/.pt/.safetensors` ok, no native binaries) | agent-contract |
| Uploads | 10 per team per day; each validated by two smoke games | rules.md |
| Losses for free | illegal move, malformed output, crash, OOM, missed init, **flag** (draw only if opponent has insufficient material) | agent-contract |
| Draws | FIDE rules + **600-ply cap** counted from the opening FEN | agent-contract |
| Openings | Curated, **unpublished** set; the eight in `harness/rules.py` are a sample. Repetition/50-move counts start at the first FEN. | README |
| Process model | One process per game, **SIGSTOP'd while the opponent thinks** (no pondering), module state survives between moves | AGENTS.md |
| Banned | Stockfish/Lc0/Maia or ports, published nets (even fine-tuned), lookup tables of engine moves/evals, obfuscation. Checked after games. | AGENTS.md |
| Allowed | Own search + hand eval, own trained net (labelling with an engine is fine), opening books, syzygy 3–4 man | AGENTS.md |

**Eligibility flag.** The London seats and prizes require a UK university student on the team. The `prod` commit is authored from an NYU address. If nobody on the team is at a UK university, the ceiling is ladder rank and Swiss placement, not London. Worth confirming before deciding how hard to grind.

### The ladder, Round 60 (377 teams, scraped 2026-09-08)

| Rank | Rating | Who |
|---|---|---|
| 1 | 2641 | AlphaFish (47 games, 31-12-4) |
| 10 | 2407 | |
| 25 | 2158 | |
| **50** | **1983** | **London cutoff, roughly** |
| 100 | 1772 | |
| median (188) | 1576 | |
| 241 | 1465 | **Sunfish** (house bot: pure-Python MTD-bi + PST, ~depth 5–6) |
| 359 | 1153 | **Minimax Three** (house) |
| 365 | 1064 | **Minimax Two** (house, = `baselines/minimax`) |
| 370 | 939 | **Greedy Material** (house, = `baselines/greedy`) |
| 377 | 802 | **Random Mover** (house) |

Distribution: 1200–1700 holds 260 of 377 teams. Above 2000 there are only 48.

What this says:
- The house bots anchor local results to ladder Elo. Beating `baselines/minimax` at 65% ≈ 1170 ladder.
- A sunfish-class pure-Python engine (iterative deepening, alpha-beta, PST, no quiescence) is only **rank 241**. Pure Python done well maybe reaches 1600–1750 (rank ~100–150).
- Top 50 (1983) is ~500 Elo above sunfish. That is the gap a **numba-compiled search reaching depth 6–8 with quiescence and a tapered eval** closes. It is not a gap that eval tweaks on a 2-ply search close.
- Top games (FableEngine vs APEX, both ~2500+): ACPL 18–41, accuracy 90–95%, and both bots spent **~1–2 s per move**, finishing with 60–80 s on the clock. The leaders are conservative on time; there is headroom to use ~2.5–3 s/move safely.

---

## 2. Where `prod` (29e6dc1) stands

What it is: fixed-depth-2 negamax, no alpha-beta, no ordering (moves sorted alphabetically by UCI), no quiescence, no TT, ignores `time_left_ms`, no repetition awareness, material + classic Simplified-Evaluation-Function PSTs, deterministic.

Measured on M2 Pro (platform core will be ~1.5–2× slower):

| Matchup | Games | TC | Result |
|---|---|---|---|
| prod vs greedy | 16 | 10s+0.1 | **100%** (+16 =0 -0, all checkmate) |
| prod vs minimax | 16 | 10s+0.5 | **65.6% ± 14.8%** (+6 **=9** -1), Elo +112 [+6, +245] |
| prod vs random | 8 | 10s+0.1 | 100% |

| Metric | Value |
|---|---|
| Time per move at depth 2 | **25–45 ms of 120,000 ms** (0.03% of the clock) |
| Time per move at depth 3, same code | 900–1700 ms |
| `evaluate()` throughput | ~55k evals/s |
| python-chess movegen+push/pop | ~320k leaf nodes/s (perft), i.e. real search runs ~10–20k nps in pure Python |

Estimated ladder rating: **~1100–1200, rank ~350 of 377.**

The three things costing the most Elo, in order:
1. **It leaves 99.97% of its time unused.** Depth 4–5 with alpha-beta is affordable in pure Python today.
2. **No quiescence.** Leaf evals are taken mid-exchange; depth-2 material counting hangs pieces to any 3-ply tactic.
3. **9 of 16 games vs minimax were threefold repetitions** in positions it was winning. It has no idea a position has occurred before, and the referee claims threefold automatically.

---

## 3. Strategy

**Goal ordering:** (a) never lose a game for free, (b) maximise depth per second, (c) then eval quality. This is the order the docs give and the ladder confirms it.

**One-line plan:** ship a robust pure-Python alpha-beta engine today (→ ~1500–1700), then spend the remaining two days moving movegen+eval into numba (→ 1900–2200), and only then touch learned eval or tablebases.

### Phase 0 — today (Sep 8): a real engine in pure Python. Target ≥ 1500 ladder (beat local Sunfish ≥ 50%).

Build in this order; each step is independently testable against the previous version.

1. **Iterative deepening + alpha-beta negamax + time management.**
   - Soft budget ≈ `time_left/25 + 400 ms`, hard cap ≈ `min(3×soft, time_left/8)`. Don't start iteration N+1 if elapsed > ~0.5×soft (it won't finish). Check the clock every ~1k nodes inside the search; on timeout return the best move from the last **completed** iteration.
   - Emergency: if `time_left < 5 s`, depth 1 + quiescence only. Keep a fixed ~300 ms margin: the harness measures wall time from send to receive and the 500 ms watchdog grace is not slack you can spend.
2. **Quiescence search** at the leaves: captures + promotions, stand-pat, MVV-LVA ordering, optional delta pruning. Biggest single strength gain after alpha-beta.
3. **Move ordering:** TT move → captures by MVV-LVA → promotions → killer moves (2 per ply) → history heuristic → rest. Avoid `board.gives_check()` in ordering (expensive in python-chess).
4. **Transposition table** (plain dict keyed on `board._transposition_key()`): depth, bound type, score, best move. Keep across moves within the game; size-cap and clear when it grows past ~2M entries (2 GB RAM).
5. **Repetition and draw awareness.** Track every position the game has passed through (start from the first FEN we're given; record our own replies too). In search, score a repetition of a game position as a draw. Add **contempt**: when our eval is > +100, score draws as −50 so we don't shuffle a won game; when behind, take them.
6. **Eval:** tapered middlegame/endgame PSTs, passed/doubled/isolated pawns, rook on open file, bishop pair, king pawn shield, and a **mop-up term** (drive the lone king to the edge and bring ours close) so K+R/K+Q vs K converts inside the 600-ply cap. Keep every number in `agent.py` readable: a judge reads it if games are flagged.
7. **Mate handling:** mate scores adjusted by ply so shorter mates are preferred; only detect mate when `legal_moves` is empty.
8. **Robustness gate before every upload:** 200 games vs `baselines/random` at 5 s (flushes promotions, en passant, no-legal-move, insufficient-material paths), `make gate`, `make zip` smoke. Zero crash/illegal/flag terminations or it does not ship.

Upload one version **today** even if unfinished, purely to read the validation log's real init time and slowest move, and scale local time budgets by that ratio.

### Phase 1 — Sep 9–10: numba engine. Target 1900–2200 (top 50–100).

The docs are explicit: jitting a shallow search buys nothing; the gain is the depth the speed affords. python-chess tops out around 10–20k nps; a numba mailbox engine should reach 0.5–2M nps, i.e. +2–3 plies.

- Board: 10×12 mailbox or 0x88 in `np.int8`, piece lists, side/castling/ep/halfmove in a small int array. Undo via an explicit history stack.
- Movegen: **pseudo-legal**, legality by "did we leave our king capturable" (check `is_square_attacked` after make). Full legal movegen is not needed and doubles the work.
- Search + quiescence + TT (numpy arrays, Zobrist keys) + eval all inside `@njit`. Python only at the root: parse FEN with python-chess, hand the array to numba, convert the result back to UCI with python-chess, and **validate the move is in `board.legal_moves` before returning** (a fallback to the pure-Python engine if not). This turns any numba bug into a lost tempo rather than a lost game.
- **Correctness:** perft to depth 3–4 against python-chess on 100+ random positions plus the classic perft suites (Kiwipete, position 4/5 with promotions/ep/castling). No engine ships until perft matches.
- **Compile budget:** warm every jitted function at import with the real argument dtypes. Measure compile time locally, expect ~2× on the platform, keep it well under 90 s. `cache=True` does nothing on the platform.
- Keep the Phase 0 engine in the zip as the fallback path.

### Phase 2 — only if Phase 1 is done and tested (Sep 10): eval quality

Pick one, not both:
- **Texel-tune the hand eval.** Label ~1–2M positions from public PGNs (lichess elite database) with Stockfish locally, fit PST/term weights by minimising logistic loss. Cheap, zero runtime cost. **Needs a ruling first:** AGENTS.md explicitly allows engine-labelled data for training a *network*; it says nothing about fitting hand-eval weights to engine labels, and the adjacent ban on shipping engine evaluations is the rule a judge would reach for. Email hello@aichessathon.com before spending a day on it.
- **Tiny NNUE-style net.** Piece-square inputs (768) → 32–64 hidden → 1, int16 weights, inference inside numba as a dot product. Trained on the same labelled data. Higher ceiling, high risk in the time left.

Skip: opening books (games start out of book from unpublished positions), 5-man syzygy (too big), anything with torch at runtime (start-up and per-call cost dwarf a hand eval on one core).

### Things to not do

- Do not tune on the eight sample openings. Test from positions you have not seen (`make play FEN=...`).
- Do not read `HARNESS_SEED` in `agent.py`; the platform does not set it.
- Do not ship any file named `chess.py`, `types.py`, `random.py`, etc.
- Do not use threads or `multiprocessing`; set `torch.set_num_threads(1)` if torch is ever imported.
- Do not write anywhere but `/tmp`, and don't rely on `/tmp` surviving.

---

## 4. Evaluation and benchmarking protocol

**Anchors.** The house bots sit on the ladder, so local results convert to ladder Elo:

| Local opponent | Ladder rating |
|---|---|
| `baselines/random` | 802 |
| `baselines/greedy` | 939 |
| `baselines/minimax` | 1064 |
| Sunfish (`local-opponents/sunfish`, GPL source fetched by script, gitignored; **local only, never ship**) | 1465 |

Beating minimax 100% tells you nothing past ~1400. Once Phase 0 is done, **Sunfish becomes the reference opponent**, and after that the previous version of our own agent does ("better than my last one" is the only comparison that matters).

**How to run.**
- Fast TC `10s + 0.1s` for iteration, 100+ games (interval must exclude 0 before a change is accepted). Arena jobs run **one at a time**: concurrent games break time measurement, and Sunfish is wall-clock budgeted so its strength moves under CPU contention. Parallel shells are for crash-hunting only, never for a number anyone quotes.
- `--pgn-dir` on every run; grep terminations. Any `flag`, `illegal`, `crash` is a P0 regardless of score.
- The interval assumes independence and there are only eight openings, so read long runs as optimistic. Add our own FENs: the platform PGNs are downloadable from game pages (`[FEN "..."]` header) and give a much larger opening sample than the eight.
- Slow the local clock to mimic the EPYC core: after the first upload, compute `platform_slowest_move / local_slowest_move` from the validation log and multiply time budgets by it (expect ~1.5–2×).
- Ladder games themselves (~15/day per version) are too few and too confounded by upload changes to evaluate anything. Use them only to catch failures (terminations) and init/move-time reality.

**Definition of done for an upload:** ruff+mypy clean; 60+ games vs random at 3 s with no failed termination; `make zip` smoke passes; local Sunfish score with lower interval bound > previous version's.

---

## 5. Immediate next actions

1. Confirm team composition / UK-student eligibility (decides whether top-50 is a real target).
2. Tell me the bot name on the ladder so its rating can be tracked (I could not identify it among 377 entries).
3. Start Phase 0 on a branch off `prod`; upload once today for the timing log.
4. Done: Sunfish wrapper in `local-opponents/sunfish` (PR #1); run `local-opponents/fetch_sunfish.sh`.
5. Begin the numba movegen in parallel if two people are available; perft harness first.

Sources: aichessathon.com/docs/rules.md, /docs/agent-contract.md, /docs, /leaderboard (Round 60), game 847b8a6a (FableEngine vs APEX), repo README/AGENTS/IDEAS, local arena runs in `harness/arena.py`.
