# Local benchmarking opponents

`baselines/` is the ladder the starter ships. This directory holds the reference opponents we
added, so a change can be scored against something that is still above us.

Every directory here is an agent directory in the harness's sense: it holds an `agent.py` with
`get_move(fen, time_left_ms)`, so `--agent` and `--opponent` take it exactly like a baseline.

The name is hyphenated on purpose. `harness/package.py` decides what to zip by reading the import
statements in every root-level `*.py` and pulling in any same-named directory beside them. A
directory called `opponents` is therefore one `import opponents` away from being packaged, GPL
`sunfish.py` and all. `local-opponents` cannot be written in an import statement, so the scan can
never reach it.

## What is here

The live ladder runs five house bots. We have local counterparts for four of them; there is no
local Minimax Three, so scores against it have to come from the ladder itself.

| Opponent | Ladder Elo | Local counterpart | What it is |
|---|---|---|---|
| Random | 802 | `baselines/random` | a uniformly random legal move |
| Greedy | 939 | `baselines/greedy` | one ply on material |
| Minimax Two | 1064 | `baselines/minimax` | two plies on material and mobility |
| Minimax Three | 1153 | none | three plies; ladder only |
| Sunfish | 1465 | `local-opponents/sunfish` | Sunfish, wrapped in the agent contract |

`local-opponents/v1.0` has no ladder counterpart: it is our own v1.0, frozen at prod
`29e6dc1`. `v2.2` and `v2.3` are the later shipped builds, frozen the same way; `docs/VERSIONS.md`
says what each is. Prod has since moved to `1151aa3`, which only rewrote docstrings and added a `.vscode`
file, so the frozen copy still plays the identical game. Keep freezing versions here: "better
than my last one" is the comparison that decides whether a change was worth shipping.

Minimax Two at 1064 is saturating for us. Sunfish at 1465 is the next anchor that can still tell
two versions apart.

## Fetching Sunfish

Sunfish is GPL-3 and it is somebody else's engine, so it is not committed. `sunfish.py` is in
`.gitignore` and ruff's `extend-exclude`, and the hyphenated directory name keeps it out of
`submission.zip` structurally rather than by convention.

```
local-opponents/fetch_sunfish.sh
```

That downloads `sunfish.py` from thomasahle/sunfish at commit
`436f2d18dc2396b623928f4b878ba7c97c964cca`, checks its sha256, and writes it to
`local-opponents/sunfish/`. The wrapper raises an `ImportError` naming this script if the file is
not there. The harness puts the agent directory first on `sys.path`, so the wrapper's
`import sunfish` finds it and nothing else on the machine does.

## Benchmarking

```
uv run python -m harness.arena --opponent local-opponents/sunfish --games 32 --increment-ms 100
uv run python -m harness.arena --opponent local-opponents/v2.2 --games 32 --increment-ms 100
uv run python -m harness.play --white . --black local-opponents/sunfish
```

The first two score `agent.py` at the arena's fast control, 10 s + 0.1 s. The third plays one
game at the real 120 s + 0.5 s, which is where time management gets tested and the fast games do
not. Add `--base-ms 120000 --increment-ms 500` to an arena run to score at the real control.

To measure the opponents against each other rather than against our agent, pass `--agent`:

```
uv run python -m harness.arena --agent local-opponents/sunfish --opponent baselines/minimax \
  --games 16 --increment-ms 100
uv run python -m harness.arena --agent local-opponents/sunfish --opponent local-opponents/v1.0 \
  --games 16 --increment-ms 100
```

Arena raises on a failed termination for whichever side is `--agent`, so run the side you are
checking there. Run arena jobs one at a time: games that share a machine stop measuring time
management, which is half of what the real control is for.

## Time management

Sunfish gets `time_left / 30` plus a bonus of `time_left / 50` capped at 300 ms, never more than
`time_left - 1000` and never less than 20 ms, and its deadline is set 100 ms (or a quarter of a
short budget) before that to absorb the overrun from checking the clock only every 2048 nodes.
Below about 1.02 s the `time_left - 1000` ceiling falls under the 20 ms floor and the floor wins,
so the budget stops tracking the clock down there; nothing reaches that region in practice.

The bonus shrinks with the clock on purpose. A flat bonus never drops under the increment, so
every move costs more than it earns, the clock ratchets down, and Sunfish flags in the endgame at
a fast control. A shrinking one crosses the increment and the clock parks there: about 1.9 s
under a 100 ms increment, about 9.4 s under the platform's 500 ms. The first move at 120 s + 0.5 s
is still budgeted 4.3 s, so strength at the real control is unchanged.

## Lint and types

`uv run ruff check .` covers this directory; only the downloaded `sunfish.py` is excluded.

mypy does not. `pyproject.toml` sets `files = ["agent.py", "harness"]`, so nothing under
`local-opponents/` is type checked. That is deliberate: this is benchmarking tooling that never
ships, and the wrapper's whole job is to call an untyped third-party module, so a strict pass
would mean annotating someone else's engine or scattering `Any` to no benefit. The gate stays on
the file that actually gets uploaded.

## Measured

All of these are after the time-management fix. No run flagged, and no game ended illegal or
crashed on either side. Sunfish's slowest move at 120 s + 0.5 s is its first, at 4.24 s.

| Matchup | Games | Time control | Score for sunfish | Terminations |
|---|---|---|---|---|
| sunfish vs minimax | 16 | 10 s + 0.1 s | 96.9% +- 6.1% (+15 =1 -0) | checkmate 15, threefold 1 |
| sunfish vs random | 16 | 2 s + 0.1 s | 100% (+16 =0 -0) | checkmate 16 |
| sunfish vs our search agent | 32 | 10 s + 0.1 s | 56.2% +- 13.0% (+11 =14 -7) | checkmate 18, threefold 13, insufficient material 1 |
| sunfish vs prod | 16 | 10 s + 0.1 s | 100% (+16 =0 -0) | checkmate 16 |

The head-to-head is the one that matters. 56.2% is +44 elo with a 95% interval of -47 to +141,
so at 32 games our search agent and a 1465 reference are not yet distinguishable: read it as
"we have reached this anchor", not as "we are ahead of it". Closing that interval needs hundreds
of games, and the eight openings stop being independent past sixteen, so it is optimistic.

The prod row was measured before the budget change and has not been rerun; prod loses every game
at either budget, so the number is the same either way.
