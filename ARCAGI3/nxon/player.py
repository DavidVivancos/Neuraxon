"""Play a brain: one cold try at a level, as the NAS plays it (the runtime's game_loop.cold_level_try).

A cold try: the brain file is loaded fresh; the game is reset; the verified moves up to the brain's start are replayed (the earlier
levels, then k moves of this level's solution); the brain then plays, learning as it goes, until it clears the level or uses up its
move cap. A game over, or a RESET by the brain, replays the start and the brain goes on (a restart). The same brain, seed and
settings always give the same moves.
"""
import os

from . import brains, runtime

ACTIONS = ('RESET', 'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6', 'ACTION7')


class _Recorder:
    """wraps the game: records every move (and, when asked, the frames) of a try"""

    def __init__(self, env, log, frames, replay_flag):
        self._e, self._log, self._frames, self._replay = env, log, frames, replay_flag

    def __getattr__(self, n):
        return getattr(self._e, n)

    def _add(self, action, data, obs):
        x = y = None
        if isinstance(data, dict):
            x, y = data.get('x'), data.get('y')
        ev = {'by': 'replay' if self._replay[0] else 'brain', 'action': getattr(action, 'name', str(action)), 'x': x, 'y': y,
              'levels': int(getattr(obs, 'levels_completed', 0) or 0), 'state': str(getattr(getattr(obs, 'state', None), 'name', ''))}
        if self._frames:
            fr = getattr(obs, 'frame', None) or []
            ev['frame'] = [list(map(list, g)) for g in fr[-1:]] if fr else []
        self._log.append(ev)

    def reset(self):
        o = self._e.reset()
        self._add('RESET', None, o)
        return o

    def step(self, action, data=None):
        o = self._e.step(action, data=data) if data is not None else self._e.step(action)
        self._add(action, data, o)
        return o


def play(brain_path, game=None, level=None, k=None, seed=53611, cap=None, cap_k=10.0, frames=False, settings=None):
    """one cold try. game / level / k default to what the brain's file name says (k: 0 when it says nothing).
    Returns {'cleared', 'moves' (brain moves), 'resets', 'seed', 'cap', 'game', 'level', 'k', 'events'}.
    events: every move of the try in order, by 'replay' (the verified start) or 'brain'; with frames=True each has its frame."""
    GL = runtime.start()
    n = brains.parse_name(brain_path)
    game = game or n['game']
    level = int(level or n['level'] or 1)
    k = int(k if k is not None else (n['k'] or 0))
    if not game:
        raise ValueError(f'which game? the name {os.path.basename(brain_path)} does not say: pass game=')
    arch = brains.arch_of(brain_path)
    cap = int(cap or runtime.move_cap(game, level, k, cap_k))
    runtime.use_brain(arch, game, level, k, settings)
    prefix = runtime.prefix_of(game, level, k)
    log, flag = [], [False]
    mk0, rp0 = GL.make_env, GL._v1153_replay

    def _replay(env, pre):
        flag[0] = True
        try:
            return rp0(env, pre)
        finally:
            flag[0] = False
    GL.make_env = lambda *a, **kw: _Recorder(mk0(*a, **kw), log, frames, flag)
    GL._v1153_replay = _replay
    try:
        with runtime.quiet():
            r = GL.cold_level_try(game, brain_path, arch, prefix, cap, int(seed))
    finally:
        GL.make_env, GL._v1153_replay = mk0, rp0
    return {'cleared': bool(r.get('cleared')), 'moves': int(r.get('actions') or 0) if r.get('cleared') else
            sum(1 for e in log if e['by'] == 'brain'), 'resets': int(r.get('resets') or 0), 'seed': int(seed), 'cap': cap,
            'game': game, 'level': level, 'k': k, 'events': log}


def _one(args):
    brain_path, game, level, k, seed, cap, cap_k = args
    r = play(brain_path, game, level, k, seed, cap, cap_k)
    r.pop('events', None)
    return r


def _init(games_dir):
    runtime.start(games_dir)


def best_seed(brain_path, seeds, game=None, level=None, k=None, cap=None, cap_k=10.0, workers=1):
    """play the brain on every seed; returns (the best try: cleared with the fewest brain moves, then fewest restarts, or None;
    every try)"""
    jobs = [(brain_path, game, level, k, int(s), cap, cap_k) for s in seeds]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        runtime.start()
        with mp.get_context('spawn').Pool(min(workers, len(jobs)), initializer=_init, initargs=(runtime._RT['games_dir'],)) as pool:
            tries = pool.map(_one, jobs)
    else:
        tries = [_one(j) for j in jobs]
    won = [t for t in tries if t['cleared']]
    best = min(won, key=lambda t: (t['moves'], t['resets'])) if won else None
    return best, tries


SEEDS = (53611, 24683, 81929, 37415, 66107, 12457, 90271, 45893, 71039, 28663, 59351, 83417, 16879, 47711, 92807, 34519)


def seed_list(n):
    """n seeds: the NAS's fixed seeds first, then reproducible extra ones"""
    import random
    out, r = list(SEEDS[:n]), random.Random(1178)
    while len(out) < n:
        s = r.randrange(10000, 100000)
        if s not in out:
            out.append(s)
    return out
