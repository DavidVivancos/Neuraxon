"""The brain runtime: starts the Neuraxon 2.0 ARC-AGI-3 brain code (ARCAGI3/runtime) and sets each brain's settings.

A brain file plays exactly as it did when it was bred only with:
  * the runtime's 1178 play flags (MULTINXON_V1178, MULTINXON_BUS78);
  * its own cell settings (the MULTINXON_* values in its arch's _meta.cell_settings_1161);
  * the level and start it plays (MULTINXON_START_LEVEL / START_STEP) and the game (MULTINXON_SENSE_TRUTH).
use_brain() sets all of them before each play.
"""
import contextlib
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARC3 = os.path.dirname(HERE)
RUNTIME = os.path.join(ARC3, 'runtime')
DEFAULT_GAMES_DIR = os.path.join(ARC3, 'environment_files')

# the 25 public ARC-AGI-3 games, in the versions the brains were bred on
GAME_VERSIONS = {
    'ar25': '0c556536', 'bp35': '0a0ad940', 'cd82': 'fb555c5d', 'cn04': '2fe56bfb', 'dc22': 'fdcac232', 'ft09': '0d8bbf25',
    'g50t': '5849a774', 'ka59': '38d34dbb', 'lf52': '271a04aa', 'lp85': '305b61c3', 'ls20': '9607627b', 'm0r0': '492f87ba',
    'r11l': '495a7899', 're86': '8af5384d', 's5i5': '18d95033', 'sb26': '7fbdac44', 'sc25': '635fd71a', 'sk48': 'd8078629',
    'sp80': '589a99af', 'su15': '1944f8ab', 'tn36': 'ef4dde99', 'tr87': 'cd924810', 'tu93': '0768757b', 'vc33': '5430563c',
    'wa30': 'ee6fef47'}
GAMES = sorted(GAME_VERSIONS)

# cell settings that describe the run that bred a brain, not the brain: never applied
_SKIP = ('ARC_ENV_DIR', 'FAST_CACHE', 'QUEUE', 'RESUME', 'MISSION', 'RUN_TRIALS', 'TRIAL_CAP', 'TIME_BUDGET', 'MAX_TRIALS_NOCLEAR',
         'SEED_DIR', 'PRIORS_KEY', 'START_LEVEL', 'START_STEP', 'V1172_CFG', 'WORKERS', 'BOX_END', 'SMOKE', 'K_CONF', 'GAME76_PLAN',
         'UNIV_POOL', 'CAPTURE', 'LOG', 'OUT_DIR', 'RUN_DIR')
_RT = {}


def start(games_dir=None, fast=False, quiet=True):
    """import the brain runtime once and return its game loop (module game_loop).
    games_dir: the folder with the 25 games (get_games.py writes it); default ARCAGI3/environment_files or $NXON_GAMES_DIR.
    fast: the runtime's faster engine (needs numba); the plays may then differ slightly from the bred ones."""
    if _RT:
        return _RT['GL']
    if sys.version_info < (3, 12):
        raise SystemExit(f'Python 3.12 or later is needed (this is {sys.version.split()[0]})')
    games_dir = os.path.abspath(games_dir or os.environ.get('NXON_GAMES_DIR') or DEFAULT_GAMES_DIR)
    if not os.path.isdir(games_dir):
        raise SystemExit(f'no games in {games_dir}: run  python get_games.py  first (or pass --games-dir)')
    if RUNTIME not in sys.path:
        sys.path.insert(0, RUNTIME)
    os.environ.update(MULTINXON_ARC_ENV_DIR=games_dir, MULTINXON_V1178='1', MULTINXON_BUS78='1',
                      MULTINXON_FAST_ENGINE='1' if fast else '0')
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        import game_loop as GL
    with open(os.path.join(RUNTIME, 'level_prefix.json'), encoding='utf-8') as f:
        prefix = json.load(f)['prefix']
    _RT.update(GL=GL, base=dict(os.environ), games_dir=games_dir, fast=bool(fast), prefix=prefix, quiet=quiet)
    return GL


def quiet():
    """a context that hides the runtime's prints (as start(quiet=True) asked)"""
    return contextlib.redirect_stdout(io.StringIO()) if _RT.get('quiet', True) else contextlib.nullcontext()


def solution(game, L):
    """(the verified moves that reach level L, the verified moves that clear level L)"""
    if game not in _RT['prefix']:
        raise ValueError(f'unknown game {game!r}: one of {", ".join(GAMES)}')
    if not 1 <= int(L) <= levels(game):
        raise ValueError(f'{game} has levels 1 to {levels(game)}, not {L} (pass the level the brain plays)')
    P = _RT['prefix'][game]
    start_ = [] if L == 1 else P[str(L)]
    nxt = P.get(str(L + 1)) or []
    return list(start_), list(nxt[len(start_):])


def levels(game):
    """the number of levels of a game"""
    return max(int(x) for x in _RT['prefix'][game]) - (0 if str(1) in _RT['prefix'][game] else 1)


def prefix_of(game, L, k=0):
    """what is replayed before the brain plays: the moves to level L, then k moves of that level's verified solution.
    k = 0: the brain plays the whole level."""
    start_, sol = solution(game, L)
    return start_ + sol[:max(0, int(k))]


def human_moves(game, L):
    """the human baseline of level L (the game's metadata), or None"""
    d = os.path.join(_RT['games_dir'], game)
    for v in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        p = os.path.join(d, v, 'metadata.json')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                b = json.load(f).get('baseline_actions') or []
            if len(b) >= L:
                return int(b[L - 1])
    return None


def move_cap(game, L, k=0, cap_k=10.0):
    """the default move budget: cap_k x the human moves for the part of the level the brain plays"""
    _, sol = solution(game, L)
    n = max(1, len(sol))
    h = human_moves(game, L) or n
    return max(10, int(math.floor(cap_k * h * max(0, n - int(k)) / n + 0.5)))


def use_brain(arch_path, game, L, k=0, settings=None):
    """the environment of one play: the runtime flags, the brain's own cell settings (or `settings`), the level and start"""
    os.environ.clear()
    os.environ.update(_RT['base'])
    if settings is None:
        try:
            with open(arch_path, encoding='utf-8') as f:
                settings = (json.load(f).get('_meta') or {}).get('cell_settings_1161') or {}
        except Exception:
            settings = {}
    for key, v in settings.items():
        if key.startswith('MULTINXON_') and not any(s in key for s in _SKIP):
            os.environ[key] = str(v)
    os.environ.update(MULTINXON_ARC_ENV_DIR=_RT['games_dir'], MULTINXON_V1178='1', MULTINXON_BUS78='1',
                      MULTINXON_FAST_ENGINE='1' if _RT['fast'] else '0', MULTINXON_START_LEVEL=str(int(L)),
                      MULTINXON_START_STEP=str(int(k)), MULTINXON_SENSE_TRUTH=game)
