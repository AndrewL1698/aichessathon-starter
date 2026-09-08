# Where we are, in plain English

Written 2026-09-08 evening, before the v3.0 build. Everything here is measured; where a number
is an estimate it says so. The detailed engine walk-through is `docs/LOGBOOK.md` Part 1, the
version list is `docs/VERSIONS.md`, and every benchmark table is `docs/BENCH_LOG.md`.

## 1. The game we are playing

The platform runs our program against other teams' programs. Each side gets 120 seconds for
the whole game plus half a second back after every move, on one core of a server processor,
with 2 GB of memory, Python only, no internet. A program that crashes, plays an illegal move,
or runs out of time loses on the spot. Games start from a set of prepared opening positions
we are not shown, several moves in. A game still going after 600 half-moves is a draw.

Rated rounds run every hour from 08:00 to 22:00 UK time and give each team a rating. That
rating only seeds the thing that matters: a 13-round Swiss tournament on Thursday afternoon,
played on whatever build each team last uploaded before 11:00 UK on Thursday. The top 50
from that go to London. The ladder today has 377 teams; the 50th is rated about 1983, the
median about 1576, and the house bot "Sunfish" (a well-known small Python engine) sits at
1465. We have played three rated games so far.

One fact shapes everything below: **the platform's processor runs our code at about 0.38 of
the speed of the laptop we test on.** Our engine searches about 27,000 positions a second
there against 65,000 here. Every local result has to be read with that discount.

## 2. What our program actually is

The whole entry is one file, `agent.py`. The platform hands it the current position and how
much time we have left, and it hands back a move. Inside, it is a classical chess engine,
which means two things working together:

**The search** looks ahead. From the current position it considers every legal move, every
legal reply to each, every reply to those, and so on, building a tree of possibilities.
"Depth 5" means it has looked five half-moves ahead along every line it did not rule out.
The number of positions it can examine per second is the engine's speed; how far ahead it
can see in the time it has is its depth. Two standard tricks make this affordable:
*alpha-beta* lets it skip lines that are already proven worse than something it has, and
*iterative deepening* searches to depth 1, then 2, then 3, so that when time runs out there
is always a finished answer from the previous depth.

**The evaluation** is what the search uses to judge the positions at the end of each line,
where it stops looking. It is a hand-written score: material (a queen is worth nine pawns),
where each piece stands (a knight in the centre is better than one in the corner), pawn
structure, king safety, and a push toward mate when we are far ahead. It is a fixed
formula, not a learned model. Since v2.4 it is "tapered": it blends a middlegame view and an
endgame view depending on how much material is left.

Around those two sit the pieces that make it play a whole game: a memory of positions it
has already analysed this game so it does not redo them (the *transposition table*), a
memory of which moves tend to be good so it looks at them first, tracking of repeated
positions so it does not stumble into or out of a draw by repetition, a time manager that
decides how long to think on each move, and a safety net that returns some legal move if
anything inside ever fails, because a crash is a lost game and a bad move is only a risk.

## 3. The versions, and what each one was for

| Version | What changed | Why |
|---|---|---|
| v1.0 | Looks two half-moves ahead, scores material and piece placement. | First working entry. |
| v2.0 | Real search: alpha-beta, iterative deepening, quiescence (keep looking at captures until the position is quiet), time budgets. | v1.0 used 40 ms of a 120 s clock. Depth is where the strength is. |
| v2.1 | The memories: transposition table, killer and history moves, repetition and fifty-move awareness, contempt (a draw is bad when we are winning). | Same depth in far fewer positions, and no more drawn wins. |
| v2.2 | The compiled board `fastboard.py` shipped in the zip but not used. Plays exactly as v2.1. | Proving the platform accepts it. This is the build that played rounds 73 and 74. |
| v2.3 | One rule in the time manager changed. | Rounds 73 and 74 showed the engine stopping at depth 4 with most of its time unused. Played round 75. |
| v2.4 | The tapered evaluation with pawn structure, king safety and mop-up (a teammate's work). | Beat v2.3 in 83% of games. Built at 17:35 and handed over for upload. |

## 4. What the three rated games told us

**Round 73, v2.2, won against Castling (rated 1462).** We won on their mistakes. Stockfish
(the strongest engine there is, which we use only offline as a referee, never inside the
program) says we made 8 real blunders. The engine spent only 103 of its 143 available
seconds and finished with 40 seconds unused, searching 4 to 6 half-moves deep.

**Round 74, v2.2, lost against Makina in 19 moves.** Four real blunders, and every one of
them was a move where the engine stopped at depth 4 or 5 after one to three seconds with
four seconds of budget still available. The last one walked into a forced checkmate. 74 of
130 seconds unused. This game is the reason v2.3 exists.

**Round 75, v2.3, won against Brokefish in 82 moves.** The time fix worked: 93% of the clock
used, 12 seconds left at the end of a long game, no move anywhere near the limit. Five real
blunders (and four "cosmetic" ones in an ending we were winning by a queen, which cost
nothing and are counted separately). Depth still 4 to 6: the engine is now using its time,
and this is what its speed buys.

Blunders per game, real ones only: 8, 4, 5. That number is the target.

## 5. How we decide what to ship

Nothing goes into the shipped build on a hunch. Each candidate change is one isolated edit
on its own branch, and it plays a *gauntlet*: 32 games against the last shipped version, 16
against Sunfish, 16 against a house bot, from a fixed set of openings with colours swapped
so colour luck cancels. The result is a score, converted to an Elo difference, with a 95%
confidence interval. Any illegal move, crash or time loss disqualifies the candidate no
matter what its score is. The best candidate is then re-run from scratch, because picking
the best of several noisy measurements flatters it; and it ships only if the *lower end* of
its interval is above zero. Rejected changes are written up in the logbook with their
numbers, because they are what stops us trying the same thing twice.

What this has found so far:

- Three standard search shortcuts (null move, late move reductions, futility) did nothing
  measurable at 32 to 88 games. At this depth they are worth tens of Elo at most, and 32
  games cannot see that: the interval is about plus or minus 120 Elo wide.
- The time-manager fix (v2.3) scored 52 to 54% against v2.2 across three runs, never
  significant, but it fixed a measured failure, so it shipped on that evidence.
- The evaluation merge (v2.4) scored 83% against v2.3, Elo +273 with a lower bound of +153,
  the first clear win in the log, and 77% at a slower control that mimics the platform.
- Cycle 3, tonight: extending time when the search looks unsettled, a gentler cost cap, and
  looking at checks in quiescence. None is beating v2.4; the checks idea loses speed and
  scored 41%.

## 6. Why we still blunder, and why the top of the ladder does not

This is the question that decides the plan, so it gets its own section.

The blunders in our games are almost all *tactics two to four half-moves beyond where the
engine stopped looking*. That is not a bug. At 27,000 positions a second on the platform,
with each level of the tree roughly four times bigger than the last, 5 to 8 seconds of
thinking reaches depth 5 or 6. A fork or a mating net that needs depth 8 is invisible.

The games we watched between the ladder's ~2500-rated programs (FableEngine against APEX, for
example) are not perfect either: measured the same way, they lose 18 to 41 centipawns per
move on average against our 62 to 123, with accuracy 90 to 95% against our 63 to 79%. What
they do not do is make the two-to-four-ply mistakes, because they see them. They spend one to
two seconds a move, less than we do, and get more depth from it. The only way that arithmetic
works is that they examine something like a hundred times more positions per second.

Where the hundred times comes from: our engine is built on `python-chess`, a library that
represents the board as Python objects. Every move generated, played and taken back is
Python code, about 40 microseconds of work. A board written as plain arrays of numbers and
compiled with `numba` (which the platform provides) does the same in well under one
microsecond. Sunfish, a well-written pure-Python engine, is rated 1465 on this ladder and
searches depth 5 to 6; that is roughly the ceiling for engines built the way ours is, and we
are already near it. Each doubling of speed is worth about half a ply; a hundredfold is three
to three and a half plies; depth 8 to 9 is where the two-to-four-ply blunders stop.

We already own that board. `fastboard.py` is a compiled mailbox board with move generation,
make and unmake, attack detection and position hashing, verified against `python-chess` on
12,000 random positions and every published perft count, and it has shipped unused in every
zip since v2.2. Locally it runs about 530,000 full legal-move-generation-plus-make/unmake
rounds per second, twenty times the current engine's node rate. Nothing in `agent.py` calls
it yet. Building the search on top of it is the project called v3.0.

## 7. What is worth doing, ranked

1. **v3.0: the search on the compiled board.** The only change that moves the blunder count
   by a lot. Several hours. Details in section 8.
2. **A learned evaluation, after v3.0.** Lichess publishes hundreds of millions of positions
   with engine scores; training our own small net on them is allowed and translates well
   (it learns objective judgement, not human habits). It cannot help before v3.0: any
   per-position neural call from Python costs more than the whole position budget and would
   lose a ply.
3. **Endgame tablebases (3 and 4 pieces).** Allowed, fit in the zip, cheap to add with the
   library the platform ships. Turns won endings into wins and lost ones into holds.
4. **A much larger blunder suite from the Lichess puzzle database.** Measurement only. Makes
   "blunders per game" a number we can track on thousands of positions instead of 17.
5. **A situational drawing policy.** Not "play for draws"; we cannot force a draw against a
   stronger engine and it would cap us at half a point against weaker ones. But *when losing*,
   steering into repetitions, the fifty-move rule, the 600-ply cap, stalemates and fortress
   shapes is right, and many simple bots cannot convert. The engine already scores a draw as
   good when it is 1.5 pawns down; that can be pushed harder and measured as a draw rate from
   losing positions.
6. **A floor under the clock.** In long games every version sinks to a few seconds on the
   clock. A reserve fixes it; it has been tried, it is safe, it is not a strength change.

Human game records are the wrong training data for anything but openings: our opponents are
bots, and a model that learns how humans lose does not transfer.

## 8. The v3.0 plan

Goal: the same engine, on the compiled board, with the v2.4 evaluation, at five to twenty
times the speed. Ship only if it beats v2.4 on the bench with the lower bound above zero and
zero disqualifiers. v2.4 stays the shipped build until then.

**What gets built, in order.**

1. *Evaluation on the compiled board.* Port v2.4's `evaluate` (tapered tables, pawn
   structure, king shield, mop-up) to a `numba` function over the array board. Gate: it must
   return exactly v2.4's number on 10,000 random positions, checked against the existing
   Python evaluation. About 1.5 hours.
2. *Search on the compiled board.* Alpha-beta with quiescence, capture ordering, killer and
   history moves, a transposition table as fixed numpy arrays indexed by the board's own
   position key, repetition detection from a list of the game's keys, fifty-move and mate
   scoring, all compiled. Gate: at the same depth with no table and no pruning, it must
   return the same best move and score as the Python search on a few hundred positions,
   which proves the search is correct rather than merely fast. About 3 hours.
3. *Time management and the Python wrapper.* Compiled code cannot read the clock, so the
   search stops when it has examined a set number of positions; the wrapper converts the
   existing soft and hard time budgets into position budgets using the speed it measures as
   it goes, and drives iterative deepening exactly as v2.3 does. Every returned move is
   checked for legality with `python-chess` before it leaves `get_move`, and if anything in
   the compiled path raises, the wrapper falls back to the current v2.4 search, which stays
   in the file. So the worst case of a v3.0 bug is a v2.4 move, not a loss. About 1 hour.
4. *Warm-up at import.* All compiled functions must compile inside the platform's 90 second
   start-up budget, at 0.38x our speed. Measured locally and multiplied by three. Currently
   the board alone takes 1.8 seconds; the search and evaluation will add more. About 0.5
   hours.
5. *Verification, in this order, stopping at the first failure.* Evaluation parity; search
   parity; 200 fast games against v2.4 with zero illegal moves, crashes or time losses;
   two full 120 s games for clock and memory; the gauntlet with intervals; the regression
   suite; `make zip` and its smoke games; then a PR into `prod` with all of it in the
   description. About 2 hours.

**What could go wrong, and what catches it.** A bug in make or unmake produces an illegal
move: caught by the legality check before the move is sent, then by the fallback, then by
the bench's disqualifier count. A bug in the search plays legal but bad moves: caught by the
parity test against the Python search. Compile time too long on the platform: measured and
tripled before upload. Memory: the table is a fixed array sized in advance (about 64 MB).
Time: the position-count stop is checked every thousand positions, a few milliseconds.

**What to expect.** Locally, depth 7 to 8 at the 120 s control in the same time v2.4 reaches
5 to 6; on the platform one ply less than that. That is the range where the two-to-four-ply
blunders disappear from our games. It will not make us 2500; it makes the evaluation work
and the learned evaluation possible, and on the ladder as measured it should move us from
the Sunfish band toward the median.
