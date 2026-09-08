# Competition logs

Drop each rated game here as `<round>-<opponent>.log` (the platform's log, first and last 4 KB
of our stderr) and `<round>-<opponent>.pgn`. Read them with:

    uv run python -m harness.readlog logs/74-makina.log
    uv run python tools/analyse_game.py logs/73-castling.pgn

Nothing in this directory ships. `harness/package.py` only zips root-level modules and the
directories they import by name, and no root module imports `logs`.
