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

### 2026-09-09, rounds 76 to 81: v3.1's six rated games. Won 3, lost 3.

All six logs carry v3.1's banner (`compiled fasteval + fastsearch`, no `book`, no `fastnnue`
line), so none of them is v3.2 or v4.0. v4.0 had not played by round 81, which finished at
12:16 UTC. Its first log will say `fastnnue` and `evaluation nnue` in the banner.

| Round | Opponent | Colour | Result | Moves | Clock left | Our ACPL / real blunders | Their ACPL / real blunders |
|---|---|---|---|---|---|---|---|
| 76 | Chessbuster 9000 | White | won, mate | 63 | 11.6 s | 80 / 10 | 95 / 14 |
| 77 | Skill Issue | White | lost, mate | 34 | 22.8 s | 94 / 5 | 23 / 1 |
| 78 | Bongcloud | Black | lost, mate | 41 | 19.1 s | 63 / 2 | 17 / 0 |
| 79 | Naveen Ragav | Black | won, mate | 35 | 27.0 s | 11 / 0 | 67 / 2 |
| 80 | Milan | White | won, mate | 42 | 17.1 s | 39 / 3 | 91 / 6 |
| 81 | Team Anay | Black | lost, mate | 25 | 21.9 s | 79 / 2 | 22 / 0 |

Stockfish 19 at depth 18 through `tools/analyse_game.py`, plus a join of our printed root
score against Stockfish's evaluation of the same position, both from our side, for the 208
moves whose search line survives. The three losses were to opponents that made zero or one
real mistake in the whole game; the three wins were against opponents that made two to
fourteen. Platform speed was 1.0 to 1.5M nps, depth 7 or 8 on nearly every move, 90 to 100%
of the clock used, no flags, no fallbacks. Time is not the story of these games.

**1. The evaluation did not know it was losing.** Where Stockfish put us between -250 and
-500, our root score averaged -94 (16 moves); between -100 and -250, it averaged -16 (20
moves); and where Stockfish had us between +250 and +500 we said +99 (13 moves). Fourteen
moves were played with our score above -100 while Stockfish's was at or below -250: two in
round 76, seven in a row in round 77 (moves 16 to 24), five in a row in round 81 (moves 19
to 23). Both decisive losses were decided inside those runs. Round 77: 14.g4 and 15.Ra3 at
depth 7 let Black's Bd4, Bf3 and Ne5 stay in our camp, and from move 16 the engine printed
-20 to -76 while Stockfish read -360 to -680. Round 81: 13...Qxd5 (depth 8, our -65;
Stockfish wants ...exd5 and drops from -92 to -504), then ten moves at -30 to -150 in a
position Stockfish had at -320 to -590, ending in 23...b5?? and a forced mate. Round 78
was the slow version: 16...O-O-O at +20 (Stockfish -57 to -217, wants ...a5) into the
a4/b4/c4 storm, then twenty moves each 20 to 130 cp worse than best.

**2. Depth halves the blunder rate per ply.** Of the moves with a surviving line, those
finished at depth 7 were real blunders 18% of the time (14 of 78, mean drop 316 cp), at
depth 8 7% (7 of 97, mean drop 125), at depth 9 or 10 5% (1 of 20). Measured at fixed
depth on four of the game positions, the tree grows 4.5 to 6.5x per ply in quiet positions,
and quiescence is 50 to 64% of the nodes at depths 6 to 8, so the cost is the main search's
branching factor, not a quiescence explosion: this is plain fail-soft alpha-beta with the
table, killers and history, null move compiled in but off, and no PVS, LMR or futility. v4.0
searches exactly this tree with a slower leaf, so it will sit at depth 7 more often than
v3.1 did.

**3. The mate-horizon class.** 23...b5 in round 81 is the clearest: at fixed depth 7 the
engine rates the position -17 and sees no threat; depth 8 is where Qe7 followed by Qf6 and
Qg7# appears, and the depth-8 tree is 42 times the depth-7 tree (13.2M nodes against 314k)
because every root move now needs that refutation. In the game the depth-8 iteration hit
the hard budget (4.76 s, clock 38 s) after proving the depth-7 move lost and finding that
b5 lost slightly less, and before reaching ...Rfe8 or Stockfish's ...Qa3. `search_root`'s
partial-iteration rule is sound as written (a move handed back has outscored the previous
best at the new depth), but in a root fail-low that is "least bad of the moves searched so
far". Partial-iteration moves were real blunders 5 times in 23 (22%) against 17 in 185 (9%)
for completed iterations; those are also the hardest positions, so read it as a symptom of
depth rather than a bug. 35.Qxe5 in round 77 and 43...Rh7 in round 78 are the same shape
with the game already lost.

**4. Conversion.** Round 76's two-rooks-against-rook-and-bishop ending took 40 moves with
three real blunders (moves 31, 39, 44) and our score at +27 to +138 where Stockfish had
+390 to +546. Won anyway.

**What v4.0 already fixes.** The twenty positions above went into `tests/positions` (the
tool's other two were already lost by 13 and 21 pawns before the move). At 20 s per position
through `--engine fast`, on a loaded machine: **v3.1 solves 5 of 22, v4.0 solves 14 of 22.**
At fixed depth 7 on the fourteen "did not know it was losing" positions, v3.1's root score
averages -28, v4.0's -168, Stockfish's -484: the sign and the trend are there, still
compressed. v4.0 finds ...exd5 in round 81 move 13 at depth 5, exf5 in round 76 moves 14 to
16 at depth 1, and Ka1 / Re1 / Rxd4 in round 77. It still misses 23...b5 at depths 8 to 9,
and 14.g4 / 15.Ra3 in round 77 stay missed by both.

**What is left, for search cycle 4, in the order the evidence supports.** (a) Cheaper
nodes: null move on, PVS, LMR, delta pruning in quiescence; each ply is worth roughly half
the blunders, and v4.0 gives one back per node. The v3.0 null-move test was 32 games and said
nothing; the 300 games it asked for are the first job. (b) Check extension and first-ply
checks in quiescence on the compiled board: the class in finding 3 is quiet moves that set
up a mate, and the python-chess rejection was about the cost of the check test, which is
gone. (c) A root fail-low rule (when the first move's new-depth score drops by 150 cp or
more, let the iteration run to the hard budget and prefer the completed depth's move only if
nothing proven better exists) is worth a candidate, benched, not assumed. (d) Time: two of
the 22 blunders were depth-7 moves the gate stopped at 33% and 39% of the soft budget with
86 and 97 s on the clock, because a table-warmed depth 7 times the growth cap of 8 overshot
the hard budget; but gate-refused moves overall blundered at 8%, the same as the rest, so
this is last.

**Tooling.** `harness/readlog.py`'s `OUTPUT_LINE` predates v3.0's line format (`tt 47%`,
`null 0`) and matches none of a v3.x log's search lines, so it reports every one of these
games as having no surviving output. Two tokens of regex; not touched here because
`harness/` is off limits, flagged for a separate fix. `tools/analyse_game.py` is fine.

**Blunders per game, real only: 8 (v2.2, r73), 4 (v2.2, r74), 5 (v2.3, r75), then v3.1:
10, 5, 2, 0, 3, 2.** The v3.1 average is 3.7 against 5.7 before it, on games that are also
shorter. Suite is 38 positions.

### 2026-09-09, rounds 82 to 85: v4.0's first four rated games. Won 2, drew 2. What carried over from v3.1, what is fixed, what is new.

Which build played which round, from the log banners: 73 and 74 v2.2, 75 v2.3 (python-chess
engine, no banner); 76 to 81 v3.1 (`fasteval + fastsearch`, no `book`, no `fastnnue`); 82 to
85 v4.0 (`fastnnue 1.5s`, `evaluation nnue h256 qa256 qb1024 qc128 cp400`, `book 24,479`).

| Round | Opponent | Colour | Result | Moves | Clock left | Our ACPL / real blunders | Their ACPL / real blunders |
|---|---|---|---|---|---|---|---|
| 82 | Rudra | White | draw, insufficient material | 125 | 8.4 s | 9 / 2 | 9 / 2 |
| 83 | TheWinners | White | won, mate | 26 | 49.8 s | 20 / 0 | 80 / 4 |
| 84 | Zugzwang | Black | won, mate | 49 | 15.7 s | 29 / 1 | 59 / 4 |
| 85 | NajeebA | Black | draw, insufficient material | 73 | 8.4 s | 17 / 2 | 19 / 3 |

Same method as rounds 76 to 81: Stockfish 19 at depth 18, and our printed root score joined
to Stockfish's evaluation of the same position, 178 moves with a surviving line. Four games
and five real blunders is a small sample; everything below about v4.0 is provisional.

**Speed and depth.** v4.0 runs at 0.52 to 0.56M nps on the platform against v3.1's 1.16 to
1.37M, and its median depth is 7 where v3.1's was 8. Locally the same position at depth 7
runs at 1.56M nps with the net and 2.89M with the hand tables alone, so the network costs
46% of the node rate here and about 55% on the platform. In BLEND mode `leaf` computes both
`evaluate` and `infer` at every leaf.

**Strengths.** The evaluation is calibrated where v3.1's was blind: in positions Stockfish
had within 100 cp of level, our score averaged 5 cp from Stockfish's (120 moves; v3.1: 33
cp), and the "did not know it was losing" run that decided rounds 77 and 81 does not occur
(2 such moves in 178, one a forced recapture; v3.1 had 14 in 208). Against the two strong
opponents (ACPL 9 and 19) v4.0 drew; v3.1 lost all three of its games against opponents of
that quality. ACPL 9 to 29 against v3.1's 39 to 94. Both mates were found cleanly once the
position was won. Book moves (10.Be2 in round 82; 7.g4 and 8.Rg1 in round 83) all cost 10 cp
or less by Stockfish.

**Weaknesses, with where they live.**

1. *Depth, carried over and now worse per node.* The search is v3.1's tree unchanged
   (`fastsearch.negamax` and `quiescence`: fail-soft alpha-beta with table, killers and
   history; null move compiled in but off; no PVS, LMR, futility or delta pruning; measured
   4.5 to 6.5x growth per ply). All four middlegame blunders were depth-6 or depth-7 moves:
   34.Qc6 in round 82 (depth 6, +319 to +14), 42.Rxc5 in round 82 (-34 to -358, line lost
   in the log gap), 23...h5 in round 84 (depth 7, +648 to +154), 20...a5 in round 85 (depth
   7, -62 to -237). On these five positions at 20 s, v4.0 solves 3 and the faster v3.1
   solves 4: they are not evaluation misses. Blunder rate by depth in v4.0: 1 of 23 at
   depth 6, 2 of 60 at depth 7, 0 of 32 at depth 8. Size: medium; each technique is 20 to 40
   lines inside `negamax` plus a 300-game bench. This is search cycle 4 as already planned.

2. *The iteration gate refuses the next ply more often now.* 34.Qc6 was played after 1.5 s of
   a 2.2 s soft budget with 44 s on the clock, because the table-warmed depth 6 times
   `GROWTH_MAX` (8) overshot the hard budget (`clock / 8`); depth 7 finds Qd1. Two of the four
   blunders were gate-refused moves at 70% and 74% of soft. Overall, gate-refused moves
   blundered 1 in 50, so this is a small effect; with a slower engine every ply costs more
   and the same cap binds earlier. Location: `fastsearch.think`, `projected`, `budgets`
   (`SOFT_DIVISOR` 25, `HARD_DIVISOR` 8, `GROWTH_MIN` 2, `GROWTH_MAX` 8). Size: constants,
   needs a bench; the v2.x runs of the same idea measured nothing.

3. *Bare endgames: the hand tables call a dead draw a win, and contempt then refuses the
   draw.* New. Round 82 reached king and h-pawn against king with the defending king on g8/h8
   at move 77 and shuffled for 57 moves, playing 134.h7 with the halfmove clock at 99 to
   dodge the fifty-move rule. `fasteval.evaluate` scores those positions +182 to +292 (passed
   pawn plus tables; `DRAWISH_MARGIN` only applies with no pawns and there is no rook-pawn or
   wrong-corner rule), `fastnnue.bare_endgame` hands the leaf to those tables, and
   `root_contempt` reads +250 > `CONTEMPT_THRESHOLD` and sets contempt to -50, so every draw
   scores -50 and any non-repeating move scores better. Cost here: 57 moves and about 30 s of
   clock (both draws ended at 8.4 s). Risk elsewhere: in a drawn ending where the only way to
   avoid repetition is a worse move, contempt gives away the half point. Location:
   `fasteval.py` endgame section (mop-up and drawish scaling) and
   `fastsearch.root_contempt` / `draw_score`. Size: small and local. Two candidate shapes:
   contempt 0 whenever `bare_endgame` is true, or a KPvK rook-pawn rule in `fasteval`.

4. *Underestimates its own winning positions.* New, cosmetic so far. Where Stockfish had us
   between +250 and +500, our score averaged +50 (9 moves); round 83 moves 12 to 19 printed
   +6 to +60 against +263 to +438, round 84 moves 21 to 33 printed +150 to +475 against +555
   to +933. Location: `fastsearch.leaf`, BLEND = (hand + net) // 2, which halves the net's
   scale wherever the tables see less. No half point traced to it. Size: a weighting change,
   which is a bench candidate, not a code fix.

5. *Root fail-low inside an aborted iteration.* Carried over unchanged
   (`search_root` and `think`); did not fire in these four games (0 of 10 partial moves
   blundered) but the code path that produced 23...b5 in round 81 is the same.

6. *Promotion tie-break.* 56...g1=R in round 85 is not an underpromotion error: g1=Q and g1=R
   score identically (+21, both answered by Rxg1) and locally the engine picks the queen at
   every depth; the in-game choice was a tie broken by table state. The mistake Stockfish
   flags is promoting at all rather than ...Re1+ (0 to -256), a depth-12 find. Nothing to fix.

**The half point that was actually lost.** Round 82: +319 at move 34 (Stockfish), two errors
in eight moves to -358 at move 42, held to a draw because Rudra did not convert either.
Round 85 was never better than level after move 23 and ended in a drawn rook ending.

**Summary of the comparison.** Fixed by v4.0: the evaluation blindness under attack that lost
rounds 77 and 81 (on v3.1's 22 blunder positions v3.1 solves 5, v4.0 solves 14). Carried
over: search depth per node, the gate, the aborted-iteration rule; all in `fastsearch.py`,
and depth is now half a ply worse because the net costs half the node rate. New: the bare-
endgame contempt loop (`fasteval` endgame terms plus `root_contempt`), advantage compression
(`leaf` BLEND), and the per-node cost of the net (`fastnnue.infer` plus the double evaluation
in BLEND). Nothing here is a wrong design; the search is a correct bare alpha-beta missing its
standard pruning, and the evaluation's remaining faults are one handover rule and one blend
weight. The five positions above are in `tests/positions`, which now holds 43.

### 2026-09-09, round 86: v4.0's first loss, to AIY in 56 moves as Black.

Banner: v4.0. Stockfish 19 at depth 18: our ACPL 56, 2 real blunders; AIY's ACPL 15 with no
real mistake, the strongest opponent in the fourteen reviewed games. Clock 130.8 s used, 13.7 s
left, depth 6 to 7 through the middlegame at 0.41 to 0.54M nps, both first moves from the book
(7...cxd4 and 8...Qa5 cost 1 and 3 cp; 8...Qa5 is Stockfish's own choice, and the retreat
9...Qd8 our search's, 1 cp).

**How it was lost.** Three small slips at depth 7 against exact play, 10...Re8 (-74),
11...a5 (-115) and 12...e5, took Stockfish's figure from -43 to -229 by move 12 while ours read
-35 to -42. Then **16...Bg7 at depth 6** (-150 to -531, Stockfish wants ...b6), played after
1.5 s of a 4.0 s soft budget with 89 s on the clock: the gate refused depth 7 because a
table-warmed depth 6 times the growth cap of 8 overshot the 11.2 s hard budget. That is the
same shape as 34.Qc6 in round 82 and 14.g4 in round 77, and it is the third real blunder in
the reviewed games that a one-ply-deeper search would have avoided while most of the clock sat
unused. In this game the seven gate-refused moves (under half the soft budget) carried two of
the three real blunders; the 22 moves that ran past the soft budget carried none. The rest is a
strong engine converting; 45...Kd5 into a lost queen ending is the other flagged move.

**Calibration when losing.** From move 17 to 35 our root score sat 300 to 430 cp above
Stockfish's (-154 to -632 against -517 to -969). Rounds 82 to 85 showed the same compression on
the winning side (+50 where Stockfish had +250 to +500); it is symmetric, which is what
`leaf`'s `(hand + net) // 2` predicts wherever the net sees more than the tables. No "unseen
danger" move in the v3.1 sense (score above -100 while Stockfish is at or below -250): the sign
is right, the size is not.

**What it adds to the v4.1 candidates.** The gate candidate (`time/hard-divisor-6`) gets its
clearest example; the blend candidate (`eval/blend-net-heavy`) gets the losing-side half of its
evidence. The position before 16...Bg7 is in `tests/positions`, which now holds 44. Blunders
per game, real only, v4.0: 2, 0, 1, 2, 2.

### 2026-09-09, round 87: v4.0 drew OnlyBlunder by perpetual check as White, from +372.

Banner: v4.0 (before PR #17 merged). Stockfish 19 at depth 18: our ACPL 27, one real blunder;
OnlyBlunder's ACPL 26 / 2 real blunders. 122.8 s used, 18.7 s left. **Twenty of our 43 moves
were searched to depth 6 and ten to depth 7**, at 0.42 to 0.68M nps; the whole middlegame from
move 17 to move 38 ran at depth 6 with 30 to 75 s on the clock.

**Where the win went.** Stockfish had +372 before move 28. 28.bxc4 (depth 6, 1.6 s of a 2.3 s
soft budget with 48 s on the clock; Stockfish wants b4) dropped it to +129, then 29.Qd4 (-76)
and 30.Rb4 (-81), both depth 6, left +30 by move 30 and the position was level from move 31 on.
The perpetual itself was not the mistake: from 41.d7 onward Stockfish scores every alternative
0 at depth 30, and the only way out of the checks, 43.Rd2, loses a rook. Our own root score of
+189 and +280 at moves 41 and 42 was a horizon reading, and the -50 from move 43 on was the
search correctly finding that every king move repeats.

**What it adds.** The third game in a row whose decisive slips were depth-6 moves with plenty of
clock (34.Qc6 in round 82, 16...Bg7 in round 86, 28.bxc4 here); v4.0 at 0.5M nps spends most
of a middlegame at depth 6 because the next iteration is projected past the hard budget. The
winning-side compression is here too: our root score read +51 to +93 across moves 11 to 27
while Stockfish read +158 to +330. The position before 28.bxc4 is in `tests/positions`, which
now holds 45. Blunders per game, real only, v4.0: 2, 0, 1, 2, 2, 1.

### 2026-09-09, cycle 4, small fixes: three candidates from the rated-game reviews. One merged, one open, one rejected.

The reviews of rounds 76 to 87 named three items small enough for one-change branches; each was
benched against `local-opponents/v4.0`, 96 games at 10 s + 0.1 s, four at a time, with the
disqualifier counts and peak RSS, and the rows are in `docs/BENCH_LOG.md`. The regression suite
(prod's 17 positions, `--engine fast`, depth 11, 20 s) is 9 of 17 for v4.0 and for all three
candidates: none of them touches the fixed-depth search.

**`eval/kpk-rook-pawn-draw`, PR #17, merged.** King and rook pawn against a bare king scores 0
when the defender is in front, in both evaluations. 45.8% (-88 to +28), no disqualifiers, and the
bench cannot see it: the rule fired in 3 of 96 games, all with the candidate defending, all drawn
as they should be. Judged on `tests.test_fasteval`'s nine rook-pawn positions and on the round 82
positions it corrects (+182 to +292 with contempt -50 becomes 0 with contempt 0). Merged into
`prod` the same afternoon.

**`time/hard-divisor-6`, PR #18, open and not proven.** Hard budget a sixth of the clock instead
of an eighth. 10 s: 42.2% (-120 to +6), which is an artifact of that control: the candidate's
clock ran under the 1 s panic floor in 19 of 96 games against 0 for v4.0. 45 s proxy: 57.3%,
+51, -25 to +133. 120 s + 0.5 s, 16 games: 53.1%, slowest move 20.25 s on a 20.25 s budget,
clock minimum 5.5 s against v4.0's 7.4 s, no disqualifiers. The mechanism is on record three
times (rounds 82, 86, 87: decisive depth-6 moves with 44 to 89 s on the clock); the cost is a
thinner clock at the end. The 96-game proxy extension came back 53.1% (+22, -34 to +79); pooled over 144 proxy games, 54.5%,
+31, -13 to +77, no game under the 1 s floor. **Not proven, not shipped on the numbers**; PR #18 stays
open for the team to take or close on the mechanism evidence.

**`eval/blend-net-heavy`, rejected.** Leaf blend one part hand to three parts net. 37.0%, -93,
-162 to -30. Clean: the hand half of the blend carries signal the net lacks at these depths, and
the score compression the change was meant to fix (root scores of +50 where Stockfish had +250 to
+500, and 300 to 430 cp short while losing round 86) is a display problem, not a decision problem.
The untested direction is the opposite weighting; nobody has measured it.

**One lesson for the bench.** Any candidate that spends more per move looks bad at 10 s + 0.1 s
because that control sits close to the panic floor; `time/growth-cap` showed the same split last
cycle. Time and budget changes are decided at 45 s + 0.2 s and 120 s + 0.5 s, reading the clock
minima out of the PGNs; the fast row is a smoke test for them.

### 2026-09-09, round 88: v4.0 drew Phantom by repetition as White. Nothing to fix.

Banner: v4.0 (prod before PR #17 merged, by the finish time). Stockfish 19 at depth 18: our
ACPL 12 and Phantom's 10, no real blunder on either side, the cleanest of the fourteen reviewed
games. 99.4 s used, 34.1 s left after 27 moves, depth 6 to 8 at 0.39 to 0.64M nps. Our root
score sat within 22 cp of Stockfish's on average across the game (mean gap -8), the best
calibration in any reviewed game: the position was level from move 10 on (Stockfish -57 to +13),
the two moves that let a small opening edge go were 12.e4 (-66, Bd2 was better) and 16.Bg5
(-52, Be3), and the repetition from move 30 was taken at 0 against 0. With the hand evaluation
inside the contempt threshold, a draw scores 0, so accepting it from a position both engines
read as level is the intended behaviour. Blunders per game, real only, v4.0: 2, 0, 1, 2, 2, 1, 0.
