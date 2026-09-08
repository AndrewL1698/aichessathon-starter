# Logbook

Two parts. Part 1 explains the engine that won round 73, v2.2 (commit `a6c1fa6`, tag `v2.2`;
see `docs/VERSIONS.md` for the numbering), as a story of one move. Part 2 is the running record of every change
we tried afterwards, including the ones we threw away.

Everything the platform runs is in `agent.py`. `fastboard.py` is also in the zip, a numba board
written for a later phase, but nothing in `agent.py` imports it, so it plays no part in any game
yet.

## Part 1: how the shipped engine works

### The library underneath

The engine does not implement chess itself. It uses `python-chess`, which the platform
preinstalls. The pieces of it you will see in `agent.py`:

- `chess.Board(fen)` builds a position from a FEN string (the standard text notation for a
  chess position: piece placement, side to move, castling rights, en passant square, and the
  two move counters).
- `board.legal_moves` lists the legal moves. `board.push(move)` plays one, `board.pop()` takes
  it back. The search walks the tree by pushing and popping on a single board.
- `board.turn` is the side to move, `board.is_check()`, `board.is_capture(move)` and
  `board.halfmove_clock` (moves since the last capture or pawn move, for the fifty move rule)
  do what they say.
- `board._transposition_key()` returns a tuple that identifies the position for repetition
  purposes: the piece bitboards, side to move, castling rights and en passant square, without
  the move counters. The engine uses this tuple as the identity of a position everywhere.

Scores are in centipawns: 100 is one pawn. Positive is good for the side to move.

### The story of one move

The platform calls `get_move(fen, time_left_ms)` and wants a move string back like `e2e4`.

**1. The safety net.** `get_move` does nothing but call `_think` inside a `try`. If anything
inside raises, it prints the traceback to the log and returns the first legal move
`python-chess` lists, and if even that fails it returns `0000`, the null move. This exists
because a crash forfeits the game while a bad move only risks it. It has never fired in a
rated game. If you ever see `Traceback` in a competition log, this is what happened, and the
move played on that turn was essentially random.

**2. Remembering the game.** `_think` builds the board and calls `_observe`. The process lives
for one game, so a few things survive from move to move in a module-level `_Memory` object:
the transposition table, the history counters, and the set of positions the game has stood in.
`_observe` checks that the position it was handed is one legal move on from the position it
handed back last time (`_reachable`). If it is not, this must be a different game, and it
wipes everything. If it is, it halves every history counter (below) and adds the new position
to the set of seen positions.

**3. Setting the budgets.** `_budgets` turns the clock into two numbers.

- The *soft* budget is `time_left / 25 + 400 ms`. This is the target: the search stops
  starting new work once it expects to pass it.
- The *hard* budget is `time_left / 8`, but never more than `time_left - 300 ms`. This is the
  deadline. The search is aborted the moment it passes it, whatever it is doing.

At 120 s the soft budget is 5.2 s and the hard budget 15 s. Both shrink with the clock, so the
engine can never spend more than an eighth of what it has, and a flag needs eight consecutive
worst cases in a row. Under one second (`PANIC_MS`) it searches one ply only. Under 300 ms it
plays the first move in its ordering without searching at all.

The clock is read every 1024 nodes rather than every node, because reading it costs more
than the nodes it would save. On a budget under 300 ms it reads every 128 nodes instead, so
the overshoot is bounded by a smaller slice.

**4. Iterative deepening.** This is the standard technique of searching to depth 1, then 2,
then 3, and so on, keeping the best move from the last finished depth. It exists for two
reasons. It makes the search an *anytime* algorithm: when the deadline arrives there is
always a complete answer from the previous depth. And each depth's result orders the next
depth's search, which is what makes alpha-beta fast. The loop lives in `_think`.

Before each depth the loop asks whether to start it: `elapsed + projected / 2 > soft` means
stop. `projected` (`_projected`) is what the next depth is expected to cost, taken as the
last depth's cost times the observed growth between the last two depths, clamped between 2
and 8, and assumed to be 5 before there are two to compare. Halving it means "start if the
budget falls inside the iteration, not only if the whole iteration fits". This gate is the
reason round 73 ended with 40 s unspent: an iteration that is projected to overrun the soft
budget is never started, even when the hard budget has plenty of room. (Part 2 has the
follow-up.)

A depth that runs past the hard budget raises `_Timeout`, caught here. The move returned is
then the best of the previous depth, unless the aborted depth had already proven a better one
(`search.root_best`, published by `_root` as soon as any move beats the previous best). The
loop also stops early when it finds a forced mate, since deeper search cannot shorten one.

**5. The root.** `_root` searches every legal move at the current depth. It tries the best
move from the previous depth first, then the rest in the order described below, and calls
`_negamax` on each with the window `(-INFINITY, -best_so_far)`, which is alpha-beta's way of
saying "only tell me if this beats the best move I already have".

**6. The search proper: negamax with alpha-beta.** `_negamax` is a *negamax* search: the same
function scores every node from the point of view of whoever is to move there, and a child's
score is negated on the way up. *Alpha-beta* is the pruning that makes it affordable: alpha is
the score the side to move is already guaranteed, beta is the score the opponent will not
allow, and once a move proves the node is worth at least beta the remaining moves need not be
looked at (a *cutoff*). The implementation is *fail-soft*, meaning it returns the actual
score it found even when that is outside the window, which gives the caller a tighter bound.

In order, a node:

1. Returns a draw score if this position has already occurred in the game or on the current
   line (repetition, below), or the fifty move counter has reached 100, or the material on the
   board cannot deliver mate.
2. At depth 0 hands over to quiescence (below).
3. Probes the transposition table (below), and returns immediately if the table already
   knows the answer to at least this depth.
4. Generates the legal moves. No moves means mate (scored `-MATE + ply`, so a shorter mate
   scores higher and the engine converts instead of shuffling) or stalemate (a draw).
5. Orders the moves, searches them, and on a cutoff records the move as a killer and in the
   history table.
6. Stores what it learned in the transposition table.

**7. Move ordering.** Alpha-beta only pays off when good moves are searched first, so
`_order_fully` sorts by a score with these tiers, highest first: the transposition table's
move for this position; captures, ordered by *MVV-LVA* (most valuable victim, least valuable
attacker: take the queen with a pawn before taking a pawn with a queen), computed in
`_move_score`; promotions; the two killer moves for this ply; and finally the quiet moves by
their history count. The tiers are separated by constants (`TABLE_BONUS` and friends) large
enough that no history count can ever outrank a killer.

**8. Killer moves.** A standard trick (the *killer heuristic*). A quiet move that caused a
cutoff at some ply is likely to cause a cutoff at the same ply in a sibling position, so two
such moves per ply are remembered in `search.killers` and tried right after the captures.
They are reset every move, since they belong to one search. Recorded in `_remember_cutoff`.

**9. The history heuristic.** Also standard. `_MEMORY.history` counts, per side and per
from-square/to-square pair, how often a quiet move has caused a cutoff anywhere in the tree,
weighted by depth squared because a cutoff high in the tree is stronger evidence. It survives
between moves, halved each time by `_observe` so old evidence fades. It is the tie-breaker
among quiet moves and is capped so it can never outrank a killer.

**10. The transposition table.** The same position is reached by many move orders, so a
*transposition table* remembers what the search learned about each position: the depth it
was searched to, the score, whether that score is exact or only a bound (`EXACT`, `LOWER`,
`UPPER`), and the best move found. It lives in `_MEMORY.table`, a plain dictionary keyed on
the position tuple from `_transposition_key`. Most engines hash the position to a 64-bit
number (Zobrist hashing); this one keys on the tuple itself, because Python's hash of a
large integer folds its top bits, and the docstring of `_key` works through why that would
let two different positions share a slot. The table persists for the whole game, which is
why the second search of a position is so much faster than the first. It is capped at
500,000 entries and simply cleared when full (`_store`).

One subtlety: a mate score is stored relative to the node rather than the root
(`_to_table`, `_from_table`), so a mate found at ply 3 does not look like a different-length
mate when the same position is met at ply 5. Another: a score that came from a repetition
belongs to the line, not to the position, so nodes whose score passed through a repetition
are not stored.

**11. Quiescence.** Stopping the search at a fixed depth and evaluating mid-exchange is the
classic mistake: the evaluation sees a queen that is about to be recaptured. *Quiescence
search* (`_quiescence`) continues past depth 0 but looks only at captures and queen
promotions until the position is quiet. The side to move may also *stand pat*, taking the
static evaluation instead of capturing, which is what makes it terminate. In check every
evasion is searched, which is also where quiescence can find a mate. It is capped at 8
plies so a long exchange cannot explode. It does not look at checking moves; Part 2 will.

**12. Evaluation.** `evaluate` is material plus *piece-square tables*: each piece type has an
8x8 table of bonuses for standing on each square (knights like the centre, kings like the
corner in the middlegame). The tables are Tomasz Michniewski's Simplified Evaluation
Function, a well known starting point. Black's pieces look up the vertically mirrored
square. That is the whole evaluation: no pawn structure, no king safety, no mobility, no
endgame knowledge.

**13. Contempt.** A draw is not always worth zero. When the root position is better than
+150 for us, a draw is scored as -50, so the search avoids repetitions and would rather play
on; when it is worse than -150, a draw scores +50 and the engine steers into one. This is a
*contempt factor*, set once per move in `_contempt` from the static evaluation, and applied
by `_draw_score`, which flips the sign at odd plies because those nodes are scored from the
opponent's side.

**14. Repetition tracking.** The referee declares a draw at the third occurrence of a
position, and the second occurrence is the move that offers it. `_MEMORY.seen` holds every
position the game has stood in, including the one after our own reply, and `search.path`
holds the positions on the line currently being searched. A node that meets a position in
either set is scored as a draw straight away. Counting the second occurrence rather than the
third is the usual simplification, and the comment in `_negamax` explains its bias: when
winning it makes us shy of positions that are not yet drawn, which is what we want; when
losing it can bank a half point the opponent has not agreed to.

**15. The log line.** Every move prints one line, and this is what the competition log
keeps. From round 73:

```
d5 score +185 move h7h6 nodes 89711 nps 26558 3378ms soft 4437 hard 12617 clock 100936 tt 21675 cut 3461 contempt +0 peakrss 28MB
```

`d5` is the last depth that finished; `from partial d6` would mean the move came out of an
aborted depth 6. `score` is that depth's score from our side. `nodes` and `nps` are the
search size and speed. `3378ms` is what the move took, against the `soft` and `hard`
budgets and the `clock` we were handed. `tt` is the table's size, `cut` the number of
cutoffs, `contempt` the draw value in force, and `peakrss` the process's peak memory.

### Where things live

| Component | Where |
|---|---|
| Fallback | `get_move` |
| Game memory and the new-game check | `_Memory`, `_observe`, `_reachable` |
| Time budgets | `_budgets`, constants `SOFT_DIVISOR`, `SOFT_BONUS_MS`, `HARD_DIVISOR`, `SAFETY_MARGIN_MS`, `PANIC_MS` |
| Iterative deepening and the iteration gate | `_think`, `_projected` |
| Root search | `_root` |
| Alpha-beta | `_negamax` |
| Quiescence | `_quiescence`, `QUIESCENCE_MAX_PLY` |
| Evaluation | `evaluate`, `PIECE_VALUES`, `*_TABLE` |
| Move ordering | `_order_fully`, `_order`, `_move_score` |
| Killers and history | `_Search.killers`, `_Memory.history`, `_remember_cutoff` |
| Transposition table | `_Memory.table`, `_key`, `_store`, `_to_table`, `_from_table`, `TABLE_MAX_ENTRIES` |
| Contempt | `_contempt`, `_draw_score`, `CONTEMPT`, `CONTEMPT_THRESHOLD` |
| Repetition | `_Memory.seen`, `_Search.path`, the top of `_negamax` |

### What it does not have

Naming these so the improvement log reads in context: no null move pruning, no late move
reductions, no futility pruning, no principal variation search or aspiration windows, no
checks in quiescence, no check extensions, no pawn structure or king safety in the
evaluation, no endgame knowledge, no opening book, no tablebases, and the numba board is not
wired in. Every one of these is a standard technique with a Chess Programming Wiki page.

### v3.0: the same engine, compiled

Everything above is still true of how the engine thinks; v3.0 changes what it is written in.
Three files ship. `fastboard.py` is the board: a 10x12 array of piece codes, move generation,
make and unmake, attack detection and a Zobrist key kept up to date by every move, compiled
by numba at import. `fastsearch.py` is the evaluation and the search from this file, ported
term for term and step for step onto that board, also compiled. `agent.py` keeps what both
engines share and what a judge needs to read: the evaluation's numbers, the time management,
the game memory, the log line and the safety net, plus the whole python-chess engine, now as
the fallback.

A move now goes like this. `_think` calls `_think_compiled`, which runs the same deepening
loop as `_think_python` with the same `_budgets`, the same gate and the same panic rule, but
each depth is searched by `fastsearch.root`: the root loop in Python, every root move's
subtree in compiled code. The compiled search cannot read the clock, so it stops on a node
count instead. Before each root move the wrapper converts what is left of the hard budget
into nodes at the rate it has measured so far this move (before the first node, at the
previous move's rate, or the warm-up's), and the search abandons the iteration when it gets
there. Between root moves the wrapper reads the real clock. The move that comes back is
checked legal with python-chess before it leaves; if the compiled path raises or returns
anything illegal, `_think_python` plays on what is left of the clock and the log says so.

Two things changed shape because arrays are not dictionaries. The transposition table is one
fixed array of two million rows (key, depth, bound, score, move), 84 MB allocated once at
import and indexed by the low bits of the Zobrist key, replaced on every store rather than
cleared when full. Repetition detection scans the game's positions and the current line
backwards, but only as far as the halfmove clock says: a capture or a pawn move changes the
board for good, so nothing before the last one can recur, and the scan is a handful of
compares instead of a set lookup over the whole game. Neither changes an answer; the parity
test in `tests/test_fastsearch.py` proves the compiled evaluation returns v2.4's exact integer
on ten thousand positions and the compiled search returns v2.4's score at fixed depth on a few
hundred, with every differing move checked to be a tie.

| Component | Compiled | Wrapper |
|---|---|---|
| Board, moves, keys | `fastboard.py` | |
| Evaluation | `fastsearch.evaluate`, tables packed by `pack_tables`, `pack_weights` | constants in `agent.py`, `_load_compiled` |
| Alpha-beta, quiescence | `fastsearch.negamax`, `fastsearch.quiescence` | `fastsearch.root` (Python, per depth) |
| Ordering, killers, history | `rank_moves`, `pick`, `SearchState.killers`, `.history` | |
| Transposition table | `SearchState.tt`, `TT_SIZE`, `TT_*` columns | `SearchState.table_move` |
| Repetition, fifty moves | the top of `negamax`, `SearchState.hist`, `path` | `SearchState.remember`, `_observe` |
| Node budget and abort | `C_MAX_NODES`, `C_ABORT` in `SearchState.ctl` | `budget` in `_think_compiled`, `NODE_BUDGET_SAFETY` |
| Fallback | | `_think`, `_think_python`, `COMPILED_IMPORT_ERROR` |
| Warm-up | `fastsearch.warm` | `WARM_UP_DEPTH`, `_load_compiled` |

## Part 2: improvement log

Every change we try gets an entry: what, why it should help, where, what the bench said, and
whether it shipped. Rejections stay in. Full tables are in `docs/BENCH_LOG.md`; the bench is
`harness/bench.py` and its disqualifier rules are described there.

### 2026-09-08, cycle 1: forward pruning. Nothing shipped.

Three candidates off `a6c1fa6`, one change each, 64 games each at 10 s + 0.1 s (32 vs the
frozen baseline, 16 vs Sunfish, 16 vs minimax). No candidate had an illegal move, exception,
timeout or over-budget move; peak RSS was not yet measured by the bench.

**Null move pruning** (`cand/null-move`, 132a964). Theory: give the opponent a free move at a
node; if a search two plies shallower still cannot get them under beta, the real moves would
not either, so skip the node. Standard technique, usually the single biggest depth gain in an
alpha-beta engine. In `_negamax`, before move generation, guarded against check, king-and-pawn
endings (zugzwang), consecutive passes and mate scores. Result vs baseline: 53.1%, Elo +22,
interval -81 to +128 over 32 games. Re-run from scratch over 56 games: 53.6%, Elo +25,
interval -40 to +91. **Rejected**: the direction reproduced but the lower bound never cleared
zero. It is the only one of the three with a consistent positive sign over 88 games, and the
open question is whether 300 to 400 games would resolve it.

**Late move reductions** (`cand/lmr`, 5990f59). Theory: the ordering is good enough that
quiet moves ranked fourth and later rarely matter, so search them a ply shallower and only
re-search at full depth if they surprise. Standard technique. In the move loop of `_negamax`,
from depth 3, exempting captures, promotions, checks and killers. Result vs baseline: 48.4%,
Elo -11, interval -114 to +90. **Rejected**: flat to slightly negative. Note this engine has no
principal variation search, which is what makes LMR pay in most engines; a zero-window
re-search structure would be a prerequisite to trying it again.

**Frontier futility pruning** (`cand/futility`, 4d8f361). Theory: one ply from the leaves,
if the static evaluation plus a margin (150 cp) cannot reach alpha, a quiet move cannot
rescue the node, so skip quiet moves and search only captures, promotions and checks.
Standard technique. In `_negamax` at depth 1. Result vs baseline: 50.0%, Elo 0, interval -114
to +114. **Rejected**: no effect measurable at 32 games.

**What the cycle taught.** At 32 games the interval is about ±120 Elo and these techniques
are worth tens of Elo in an engine this size, so the bench as run could not have found a
real gain. The Sunfish column is worse: the same baseline scored 25% over 4 Sunfish games and
78% over 16, because Sunfish's clock-driven deepening makes its moves vary with timing
jitter. Ranking candidates needs several hundred games against the baseline, which the bench
can play at about 450 an hour.

### 2026-09-08, Phase 0: spend the clock. Verified, not shipped.

Round 73 ended with 40.4 s of 143.5 s unspent, slowest move 6.3 s against a 13.3 s hard
budget, depth 4 to 6, peak RSS 73 MB of 2 GB, and the transposition table at 114k of a
500k cap. Two constants changed on `phase0/spend-the-clock` (b12c8f0, off `prod`):
`SOFT_DIVISOR` 25 to 16 (soft budget at 120 s from 5.2 s to 7.9 s; hard budget and abort
path untouched) and `TABLE_MAX_ENTRIES` 500k to 1M. Not shipped for the 16:00 round: the
full-game clock check could not run inside the 5 minute window.

Verification at 120 s + 0.5 s self-play afterwards: 20 plies clean; a 130-move full game with
no exceptions, no illegal moves, peak RSS 514 MB with the table at 996k entries (so 1M costs
about 500 MB when it fills, which a long game does); `make zip` smoke clean. The slowest move
was 10.5 s against an 11.9 s hard budget, and 22 moves overshot their hard budget by 3 to
49 ms, which is the clock-check slice and happens on the baseline too. **Failed the clock
floor**: 12.8 s left after move 47, 8.7 s after move 60, 4.4 s at the lowest, against the
"not below 10 s" asked for. A model fitted to the observed spend (84% of the soft budget)
says the old divisor sinks the same way, only later: 12.5 s at move 80, 8.8 s at move 100.
The budget formula `clock / N + 0.4 s` has no floor, so every divisor converges on a few
seconds in a long game; the 500 ms increment is what stops it at 4 to 6 s. **Not tagged,
recommend not uploading.** The lesson goes into the loop's priority A: the fix is a reserve
the budget never spends, not a divisor.

### 2026-09-08, Phase 2 setup: what round 73 says

Stockfish 19 at depth 18 over the round 73 PGN (`tools/analyse_game.py`): 47 of our moves,
ACPL 87, 8 real blunders at the 150 cp threshold (4 at 300 cp), 0 cosmetic, 1 opening and
7 middlegame, none in the endgame because the game was over by then. The platform's review
said 5 blunders and ACPL 62; different engine, depth and threshold. **Blunders per game,
the target metric: 8 (150 cp) / 4 (300 cp) for round 73.** All eight are in
`tests/positions/positions.epd`; the baseline solves 2 of 8 in 20 s per position at depth
5 to 7. The competition machine runs the engine at 25k nps against 58k to 80k here in self-
play, a ratio of 0.38 (`harness/readlog.py`), so local depth-6 games overstate what the
platform sees; on the platform the same search reaches depth 4 to 5.

### 2026-09-08, round 74: lost to Makina in 19 moves. The time manager is implicated.

Stockfish 19 at depth 18 over the round 74 PGN: 19 of our moves, ACPL 123, **4 real blunders
at 150 cp, 0 cosmetic** (moves 14, 17, 20 and 21; the last, Qg3 instead of Bh5, walked into a
forced mate). Blunders per game: 4, which over 19 moves is a worse rate than round 73's 8
over 47. The four are in `tests/positions`; the baseline solves 1 of them in 20 s.

Every one of the four was a depth-4 or depth-5 move that stopped after 1.3 to 2.7 s with
about 4 s of soft budget still available. The game ended with 74.4 s of 129.5 s unspent (57%)
and an average spend of 68% of the soft budget, the same pattern as round 73 and the same
mechanism: the iteration gate refusing depth 5 or 6 because its projection of the next
iteration's cost is capped at eight times the last one. Competition nps was 20k to 52k,
median 29k.

Also from the log: the average move spent 66% of its soft budget, and 30 of 47 moves
stopped at depth 4 in about a second with 3 to 5 s of soft budget left. The iteration gate
projects the next depth at up to 8 times the last one, and when the table has made depth
3 nearly free the ratio between depths 3 and 4 hits that cap, so depth 5 is refused. That
is the cycle 2 diagnosis.

### 2026-09-08, cycle 2: time management. PR #7 opened for gate-hard.

Three candidates off `prod` (a6c1fa6), one change each, 64 fast games each plus two 120 s
games against the baseline. No disqualifiers anywhere. The regression suite bypasses the
time manager, so it is unchanged for all three (baseline: 3 of 12).

**Gate on the hard budget** (`time/gate-hard`, 8cc4670, **PR #7**). Theory: the gate
refused any iteration projected to end past the soft budget, and the projection is capped
at 8x the last iteration, which the table-warmed early depths trip constantly; rounds 73 and
74 both show depth-4 moves stopping in a second with 4 s of budget unspent, and all four of
round 74's blunders are such moves. New rule in `_think`: start while the soft budget is
unspent and the whole projected iteration fits the hard budget. Fast bench vs baseline:
53.1%, Elo +22, interval -81 to +128 (32 games). 120 s: spent 131 to 136% of soft over the
first 40 moves, depth 6.33 and 5.83 vs the baseline's 6.00 and 5.55 on the other side of the
same boards, 1 loss 1 draw. Clock 12 s at move 47, 6 s at move 80 in a 120-move game (the
baseline reached 4.8 s in the same game). **Chosen for the PR** because it is the change
that acts on the diagnosed mechanism; the Elo lower bound is not above zero, and the fast
control cannot show a 120 s time-management change, so this is a judgment on the 120 s
depth evidence and two rated games, stated as such.

**Growth cap 4** (`time/growth-cap`, 7a5d60d). Theory: same diagnosis, minimal remedy, cap
the projected growth at 4 instead of 8. Fast bench vs baseline: 42.2%, Elo -55, interval
-168 to +48. 120 s: spent 100 to 120% of soft, depth 5.47 and 6.08 vs 5.78 and 5.92, 1 win
1 draw. **Not promoted**: the fast result points the wrong way, and gate-hard reaches deeper.

**Reserve floor** (`time/reserve-floor`, ace9ebc). Theory: Phase 0 showed the budget formula
sinks the clock to 4 s in long games; computing the soft budget from `clock - 10 s` makes
it floor there, since the 400 ms bonus is under the 500 ms increment. Fast bench vs
baseline: 60.9%, Elo +77, interval -14 to +181, the best of the three, but at a 10 s clock
the reserve makes it play a flat 400 ms budget, so its edge there is holding more clock into
the endgame, not the platform's regime. 120 s: spent 84 to 86% of soft (it spends slightly
less, by design), floor 7.0 s in a 144-move game where the baseline sank to 4.7 s, 1 win 1
loss. Re-run from scratch: **44.6%**, Elo -37, interval -112 to +34 over 56 games. The
60.9% was selection noise. **Rejected as a strength change**; still the natural way to put a
floor under the clock, but it has to be measured as a safety change at 120 s, not sold on Elo.

Closing numbers. Gate-hard (now **v2.3**) re-run from scratch: 51.8%, Elo +12, interval -60 to
+86 over 56 games. At the 45 s + 0.2 s platform proxy (0.38x speed makes it the platform's 120 s
in nodes per game): v2.3 54.2% (Elo +29, -72 to +135) and growth-cap 60.4% (Elo +73, -44 to
+211), 24 games each, overlapping intervals. v2.3 was uploaded for round 75 at about 17:00 on
the depth evidence and the two rated games, and every run of it was clean on the
disqualifiers. **Blunders per game so far: v2.2 played 8 (round 73) and 4 (round 74); v2.3's
first rated game will be the first data point for it.** Cycle 3's first job is a 100+ game
match of v2.3 against growth-cap at the proxy control, then a dynamic time extension and
checks in quiescence as candidates.

**On dynamic allocation** (asked this cycle): standard engines do vary time by position,
and it works: spend less when the move is forced or the table's move has held across
iterations, spend more when the best move changed in the last iteration or the score fell.
It is a good strategy, but it is the second fix, not the first. Right now the engine refuses
depth it has time for on nearly every move; a dynamic rule sitting on top of a gate that
refuses iterations would still be refused. Gate first, measure, then "extend when unstable"
as a cycle 3 candidate.

### 2026-09-08, cycle 3 opening: the evaluation merge is v2.4. Shipped on the numbers.

While cycle 2 was closing, a teammate merged PR #5 into `prod` on top of v2.3: a tapered
evaluation (separate middlegame and endgame piece-square tables blended by how much material
is left), mop-up terms that drive a won ending to mate, passed, isolated and doubled pawn
terms, and a king shield. Standard techniques all; "tapered evaluation" and "mop-up" are the
names to look up. It had not been benchmarked against v2.3 when it landed.

Benched first: **82.8% against v2.3** (+25 =3 -4 over 32 games), Elo +273, interval +153 to
+510, the first lower bound above zero in this log by a wide margin; 71.9% against Sunfish;
no illegal moves, exceptions, timeouts or over-budget moves; peak RSS 243 MB in a 120 s game.
On the same eight positions it searches to the same depth (5.38 vs 5.25) at higher speed
(62k vs 50k nps), so the gain is evaluation quality, not depth. The regression suite stays at
3 of 12: those positions are tactical and this change is positional. Built as
`submission-v2.4.zip` from 1077652 and handed over for the next round; the 45 s platform
proxy match is running and goes into the bench log when done.

Blunders per game: still 8 and 4 from v2.2's two rated games; v2.3 and v2.4 have not played
a rated game yet as this is written.

### 2026-09-08, round 75: v2.3's first rated game. Won vs Brokefish in 82 moves.

The time change did what it was for: 149 s of 161 s used (7% unspent, against 28% in round
73 and 57% in round 74), average spend 89% of the soft budget (was 66% and 68%), slowest move
9.3 s against a 9.3 s hard budget (the abort path, as designed), and still 12 s on the clock
at the end of an 82-move game. Depth 4 to 6 as before: the platform's 27k nps median is what
bounds depth now, not the gate. Peak RSS 119 MB. 62 of 82 move lines survive; 2.4 KB of the
middle is gone and nothing is inferred from it.

Stockfish 19 at depth 18: ACPL 72, **5 real blunders and 4 cosmetic** (the cosmetic four are
in an ending we were winning by a queen; the tool separates them as asked). All five real
ones are in the middlegame, moves 23 to 34, and three of them are the same missed idea
(Ba4+ with the bishop). They are in `tests/positions`, which now holds 17.

**Blunders per game, real only: 8 (v2.2, r73), 4 (v2.2, r74), 5 (v2.3, r75).** Same order of
magnitude; the one game is not evidence either way on v2.3's blunder rate, but it is
evidence that the unspent-clock problem is gone.

### 2026-09-08, cycle 3: dynamic time and checks in quiescence. Nothing shipped.

Baseline v2.4. Three candidates, one change each, 64 fast games each plus the 45 s platform
proxy and two 120 s games for the timing ones. No disqualifiers.

**Extend when unstable** (`time/unstable-extend`, de956ad). Theory: when a finished depth
changes the best move or drops the score by 50 cp, the root is unsettled, so give that move
half as much soft budget again, within the hard budget; the standard "best-move stability"
extension. Fast: 51.6% vs v2.4 (Elo +11, -104 to +129). Proxy: exactly 50.0% (-144 to +144).
120 s: same depth and spend as v2.4 on the other side of the same board. **Rejected**: no
effect at any control. Now that v2.3 already spends the budget, there is little left for an
extension to add; instability at depth 5 is also common enough that "extend once" is close
to "always extend".

**Growth cap 4 on the hard gate** (`time/growth-cap-4`, e27abf8). Theory: cycle 2's tie-break.
Fast: 48.4% (-11, -129 to +104). Proxy: 52.1% (+14, -116 to +149). 120 s: 5% more time for
equal depth and a 9.3 s clock minimum. **Rejected**: it spends more for nothing measurable
and pushes the clock lower.

**Checks in quiescence** (`search/qs-checks`, bb0690e). Theory: the leaves cannot see checks,
which is where tactical blunders live, so search quiet checking moves at the first
quiescence ply. Fast: 40.6% vs v2.4 (Elo -66, -196 to +47); 75% vs Sunfish. On the tactical
test position it ran at a third of v2.4's speed, because filtering every legal move for
checks at every leaf is expensive in python-chess. Suite unchanged at 3 of 12. **Rejected**:
it gets sharper and weaker at once, the textbook case of why the decision is made on Elo.
Worth revisiting on the compiled board, where a check test costs nothing.

**What cycle 3 says.** With v2.3 spending the clock and v2.4 evaluating well, the remaining
Elo in the python-chess engine is small change; three sensible candidates measured within
noise of zero. The blunders left are depth, and depth is speed. Next is v3.0 (the search on
`fastboard.py`), described in `docs/BRIEF.md` section 8.

### 2026-09-08, v3.0: the search on the compiled board. In progress.

Branch `v3/compiled-search` off `prod` (c5974d2, v2.4's engine), with `tooling/bench` merged
in because its gauntlet is what produces the numbers. Baseline v2.4.

**What.** `fastsearch.py`: the v2.4 evaluation and search compiled by numba over
`fastboard.py`, the board that has shipped unused since v2.2. `agent.py` drives it with v2.3's
time management unchanged, packs the evaluation's tables for it at import, checks every move
legal, and keeps the whole python-chess engine as the fallback. Part 1 above has the shape.

**Why.** Cycles 1 to 3 measured the python-chess engine as squeezed: null move, LMR,
futility, three time changes and checks in quiescence all within noise of zero. The real
blunders in rounds 73 to 75 were two to four plies past depth 5 to 6, and at 27k nps on the
platform depth is speed. `fastboard.py` had already shown twenty times python-chess's
generate-and-make rate.

**How it was built and proved, in order.** Evaluation first: one `@njit` pass over the
mailbox, pawn structure from per-file bitmasks, no allocation. Gate: exactly `agent.evaluate`'s
integer on 10,000 positions from `tests/test_fastboard`'s generator plus 45 hand-built endings
(mop-up, the no-pawn draw rules, shields, passed and doubled pawns). Passed first time. Search
second: `negamax` and `quiescence` step for step, table, killers, history, repetition, fifty
moves, insufficient material, mate by ply, a node-count stop. Gate: with the table and killers
off on both sides, the same score as `agent._root` on 334 positions at depth 3 and 35 at
depth 4; 25 and 12 of those chose a different move, each verified a tie by scoring the
compiled move with the Python search. With everything on, no score drift on the same sets.
Wrapper third, warm-up fourth: import with warm-up 2.4 s locally (fastboard alone was 1.8 s),
so about 7 s at the platform's speed, against a 90 s budget.

**Measured so far.** Node rate on the eight harness openings: median 2.4M nps compiled
against 70k for v2.4 in the same process, 34x. One move each at a 75 s clock: v3.0 mean
depth 7.88 (two of eight partial at the hard budget) against v2.4's 5.50, at 2.42M against
79k nps. Peak RSS 264 to 282 MB on 120 s moves. Regression suite, 17 positions at 20 s: v3.0
solves 5 at depth 7 to 9, v2.4 4 at depth 5 to 7, three in common; v3.0 finds round 75's Ba4
idea twice where v2.4 never does, and drops one round 73 position it chose at depth 5 and
leaves at depth 9. Deeper is not automatically sharper by Stockfish's lights, which is why the
suite does not decide anything. `tests.test_fastboard`
clean. **200 fast games against v2.4: +181 =15 -4, 94.2%, Elo +486, interval +416 to +596,
no illegal moves, exceptions, flags or over-budget moves, worst move 1.28 s at a 9 s clock,
peak RSS 266 MB.** The first lower bound in this log above a hundred, let alone four. Two 120 s
games against v2.4, one per colour: both won by checkmate, depth 8.05 and 8.93 over the first
40 moves against 5.85 and 6.33 on the other side of the board, slowest moves landing inside
the hard budget with 25 ms the largest overshoot, clock never under 12 s, peak RSS 263 MB, no
fallback on any move. Gauntlet: 93.8% against v2.4 (+28 =4 -0), 16-0 against Sunfish (v2.4
scored 71.9%), 16-0 against minimax, no disqualifiers.

**One thing the gauntlet found, and the fix.** Its slowest move was 1.35 s at a 7.3 s clock,
inside the bench's quarter-of-the-clock rule but 48% past the 0.91 s hard budget, where every
python-chess version overshot by one clock-check slice of tens of milliseconds. It
reproduces on the position (`N1bk3r/pp2bppp/2np4/3qp1B1/8/8/PPP1BPPP/R2QK2R w KQ - 2 14`,
clock 7259: 1191 ms against hard 907, deterministic). The cause is the design: the node budget
for a root move is set from the node rate measured so far in the move, and this crushing
position's last subtree ran at 2.1M nps where the earlier iterations had run much faster, so
the budget was too generous. The python-chess engine could not do this because it read the
wall clock every 1024 nodes. The fix keeps the node budget as the ordinary stop and adds a
backstop: a timer thread that sleeps until the hard deadline and then zeroes the node budget,
so the compiled search stops at its next node whatever the estimate said; `negamax` and
`quiescence` release the interpreter lock (`nogil`) so the thread can run while the search is
in compiled code, and the timer is cancelled once the move is chosen. It does no computation.
The same position now takes 918 ms; with the node budget disabled, a 0.5 s and a 1.5 s timer
stopped a depth-12 search at 0.510 s and 1.510 s. This departs from the brief's "node count,
not the clock" only in adding the clock as a guard; the node budget still decides nearly every
stop. Because it changes the shipped code after the 200-game gate, the games are re-run on the
final code below.
