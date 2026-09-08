# Local benchmarking opponents

`baselines/` is the ladder the starter ships. This directory holds the reference opponents we
added, so a change can be scored against something that is still above us.

Every directory here is an agent directory in the harness's sense: it holds an `agent.py` with
`get_move(fen, time_left_ms)`, so `--agent` and `--opponent` take it exactly like a baseline.

## What is here

| Opponent | Ladder Elo | What it is |
|---|---|---|
| `baselines/random` | 802 | a uniformly random legal move |
| `baselines/greedy` | 939 | one ply on material |
| `baselines/minimax` | 1064 | two plies on material and mobility |
| `opponents/prod` | — | our own submission frozen at prod commit `29e6dc1` |
| `opponents/sunfish` | 1465 | Sunfish, wrapped in the agent contract |

The four house bots on the live ladder are exactly these opponents, so a local score against one
of them converts to ladder Elo. Minimax at 1064 is saturating for us; Sunfish at 1465 is the next
anchor that can still tell two versions apart.

`opponents/prod` is a byte-for-byte copy of the `agent.py` we last shipped, with one docstring
line naming the commit. Keep freezing versions here: "better than my last one" is the comparison
that decides whether a change was worth shipping.

## Fetching Sunfish

Sunfish is GPL-3 and it is somebody else's engine, so it is not committed. `sunfish.py` is in
`.gitignore` and ruff's `extend-exclude`; it never goes anywhere near `submission.zip`.

```
opponents/fetch_sunfish.sh
```

That downloads `sunfish.py` from thomasahle/sunfish at commit
`436f2d18dc2396b623928f4b878ba7c97c964cca`, checks its sha256, and writes it to
`opponents/sunfish/`. The wrapper raises an `ImportError` naming this script if the file is not
there. The harness puts the agent directory first on `sys.path`, so the wrapper's `import
sunfish` finds it and nothing else on the machine does.

## Benchmarking

```
uv run python -m harness.arena --opponent opponents/sunfish --games 32 --increment-ms 100
uv run python -m harness.arena --opponent opponents/prod --games 32 --increment-ms 100
uv run python -m harness.play --white . --black opponents/sunfish
```

The first two score `agent.py` at the arena's fast control, 10 s + 0.1 s. The third plays one
game at the real 120 s + 0.5 s, which is where time management gets tested and the fast games do
not. Add `--base-ms 120000 --increment-ms 500` to an arena run to score at the real control.

To measure the opponents against each other rather than against our agent, pass `--agent`:

```
uv run python -m harness.arena --agent opponents/sunfish --opponent baselines/minimax \
  --games 16 --increment-ms 100
uv run python -m harness.arena --agent opponents/sunfish --opponent opponents/prod \
  --games 16 --increment-ms 100
```

Arena raises on a failed termination for whichever side is `--agent`, so run the side you are
checking there. Run arena jobs one at a time: games that share a machine stop measuring time
management, which is half of what the real control is for.

## Time management

Sunfish gets `time_left / 30` plus a bonus of `time_left / 50` capped at 300 ms, never more than
`time_left - 1000`, and its deadline is set 100 ms (or a quarter of a short budget) before that
to absorb the overrun from checking the clock only every 2048 nodes.

The bonus shrinks with the clock on purpose. A flat bonus never drops under the increment, so
every move costs more than it earns, the clock ratchets down, and Sunfish flags in the endgame at
a fast control. A shrinking one crosses the increment and the clock parks there: about 1.9 s
under a 100 ms increment, about 9.4 s under the platform's 500 ms. The first move at 120 s + 0.5 s
is still budgeted 4.3 s, so strength at the real control is unchanged.

## Measured

All of these are after that change. Terminations are per game and no run flagged.

| Matchup | Games | Time control | Score | Terminations |
|---|---|---|---|---|
| sunfish vs minimax | 16 | 10 s + 0.1 s | 96.9% +- 6.1% (+15 =1 -0) | checkmate 15, threefold 1 |
| sunfish vs random | 16 | 2 s + 0.1 s | 100% (+16 =0 -0) | checkmate 16 |
| sunfish vs prod | 16 | 10 s + 0.1 s | 100% (+16 =0 -0) | checkmate 16 |

The prod row was measured before the budget change and has not been rerun; prod loses every game
at either budget, so the number is the same either way.
