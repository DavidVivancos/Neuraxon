"""The 25 public ARC-AGI-3 games, in the versions the brains were bred on, fetched with the official toolkit (pip install arc-agi).

The toolkit gets a free anonymous key by itself (or uses ARC_API_KEY) and writes each game to
<games_dir>/<game>/<version>/ (metadata.json + the game's Python file), the folder the runtime plays from.
"""
import os

from .runtime import DEFAULT_GAMES_DIR, GAME_VERSIONS


def have(games_dir, game):
    v = GAME_VERSIONS[game]
    return os.path.isfile(os.path.join(games_dir, game, v, 'metadata.json'))


def download(games_dir=DEFAULT_GAMES_DIR, games=None, log=print):
    """download the games that are missing; returns the games that are still missing"""
    import arc_agi
    games = list(games or GAME_VERSIONS)
    todo = [g for g in games if not have(games_dir, g)]
    if not todo:
        log(f'all {len(games)} games are in {games_dir}')
        return []
    os.makedirs(games_dir, exist_ok=True)
    try:
        arc = arc_agi.Arcade(environments_dir=games_dir)
    except Exception as e:
        log(f'could not reach the ARC-AGI-3 API ({type(e).__name__}: {str(e)[:160]}). Check the connection, or set ARC_API_KEY '
            f'(a free key from https://three.arcprize.org), or copy an environment_files folder and pass --games-dir.')
        return todo
    for g in todo:
        try:
            arc.make(f'{g}-{GAME_VERSIONS[g]}')
        except Exception as e:
            log(f'  {g}: {type(e).__name__}: {e}')
        log(f'  {g}-{GAME_VERSIONS[g]}: {"ok" if have(games_dir, g) else "FAILED"}')
    return [g for g in games if not have(games_dir, g)]
