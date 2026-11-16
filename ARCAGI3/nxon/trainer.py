"""Train brains, three ways:

  newborn   a new brain built from an arch file lives N episodes of a level with plasticity on (it learns as it plays), then is saved
  practice  an existing brain plays N more episodes of a level, learning, then is saved with what it learned
  evolve    the way the NAS 1170s brains were bred: children = the parent with a few synapse weights mutated; each child plays cold
            tries on fixed seeds; a child that clears more seeds, or as many in fewer moves, becomes the parent

A saved brain is <stem>.brain.json.gz with its arch copied next to it as <stem>.arch.json, so play_brain.py plays it at once.
"""
import os
import random
import time

from . import brains, player, runtime


def _dump(arc_brain):
    from neuraxon.multisphere import save_multisphere_to_dict
    return save_multisphere_to_dict(getattr(arc_brain, 'brain', arc_brain))


def newborn(arch_path, game, level, k=0, episodes=4, seed=90001, cap=None, cap_k=10.0, out=None):
    """build a brain from an arch, give it a training life of `episodes` tries at the level, save it; returns (path, summary)"""
    GL = runtime.start()
    runtime.use_brain(arch_path, game, level, k)
    prefix = runtime.prefix_of(game, level, k)
    cap = int(cap or runtime.move_cap(game, level, k, cap_k))
    with runtime.quiet():
        lb, summary = GL.train_level_life(game, arch_path, prefix, int(episodes), cap, int(seed))
    summary = {**summary, 'start_level': int(level), 'start_step': int(k), 'seed': int(seed), 'cap': cap}
    out = out or f'nxon_{game}-l{level}_k{k}_newborn_s{seed}.brain.json.gz'
    brains.save(out, {'_meta': {'game': game, 'source': 'nxon-newborn', 'train': summary}, 'brain': _dump(lb)}, arch_from=arch_path)
    return out, summary


def practice(brain_path, game=None, level=None, k=None, episodes=4, seed=90001, cap=None, cap_k=10.0, out=None):
    """an existing brain plays `episodes` tries at the level (loaded as a cold try loads it, plasticity on), then is saved"""
    import gzip
    import json
    GL = runtime.start()
    n = brains.parse_name(brain_path)
    game, level = game or n['game'], int(level or n['level'] or 1)
    k = int(k if k is not None else (n['k'] or 0))
    arch = brains.arch_of(brain_path)
    runtime.use_brain(arch, game, level, k)
    prefix = runtime.prefix_of(game, level, k)
    cap = int(cap or runtime.move_cap(game, level, k, cap_k))
    from arc_brain import ArcBrain, build_params_from_arch
    from neuraxon.multisphere import load_multisphere_from_dict
    try:
        from neuraxon.network import _rebuild_net_from_dict as _rebuild
    except Exception:
        _rebuild = None
    eps = []
    with runtime.quiet():
        GL._v1098_seed_everything(int(seed))
        params = build_params_from_arch(arch, verbose=False)
        with gzip.open(brain_path, 'rt', encoding='utf-8') as fh:
            payload = json.load(fh)
        prebuilt = load_multisphere_from_dict(payload.get('brain', payload), rebuild_net_fn=_rebuild)
        env = GL.make_env(game, seed=int(seed), prefer_real=True, force_offline=False)
        bseed = int(os.environ.get('MULTINXON_FIXED_BRAIN_SEED', '').strip() or seed)     # as cold_level_try seeds the brain
        brain = ArcBrain(params, seed=bseed, micro_steps=3, prebuilt_brain=prebuilt)
        for i in range(int(episodes)):
            r = GL._v1153_play_level(brain, env, prefix, cap, first_reset=(i == 0))
            eps.append(int(r['actions']) if r.get('cleared') else 0)
    summary = {'episodes': int(episodes), 'ep_actions': eps, 'start_level': level, 'start_step': k, 'seed': int(seed), 'cap': cap,
               'from': os.path.basename(brain_path)}
    out = out or os.path.basename(brain_path).replace('.brain.json.gz', f'_practiced_s{seed}.brain.json.gz')
    brains.save(out, {'_meta': {'game': game, 'source': 'nxon-practice', 'train': summary}, 'brain': _dump(brain)}, arch_from=arch)
    return out, summary


def score(brain_path, game, level, k, seeds, cap=None, cap_k=10.0, workers=1):
    """(seeds cleared, mean brain moves of the clears): more clears is better, then fewer moves"""
    _, tries = player.best_seed(brain_path, seeds, game, level, k, cap, cap_k, workers)
    won = [t['moves'] for t in tries if t['cleared']]
    return len(won), (sum(won) / len(won) if won else float('inf'))


def _better(a, b):
    return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1])


def evolve(brain_path, game=None, level=None, k=None, generations=10, children=4, seeds=player.SEEDS[:4], sigma=0.008, frac=0.02,
           cap=None, cap_k=10.0, out_dir='evolved', rng_seed=1, workers=1, log=print):
    """(1 + children) evolution from a brain; returns the path of the best brain found (the start brain if no child beat it)"""
    n = brains.parse_name(brain_path)
    game, level = game or n['game'], int(level or n['level'] or 1)
    k = int(k if k is not None else (n['k'] or 0))
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(rng_seed)
    best, best_s = brain_path, score(brain_path, game, level, k, seeds, cap, cap_k, workers)
    log(f'start {os.path.basename(brain_path)}: {best_s[0]}/{len(seeds)} seeds cleared, {best_s[1]:.1f} moves')
    for g in range(1, int(generations) + 1):
        t0 = time.time()
        for c in range(int(children)):
            kid = os.path.join(out_dir, f'nxon_{game}-l{level}_k{k}_g{g:03d}c{c}.brain.json.gz')
            brains.child(best, kid, rng, sigma, frac)
            s = score(kid, game, level, k, seeds, cap, cap_k, workers)
            if _better(s, best_s):
                best, best_s = kid, s
                log(f'  gen {g} child {c}: {s[0]}/{len(seeds)} cleared, {s[1]:.1f} moves  <- new parent')
            else:                                          # a child that lost is not kept
                for p in (kid, kid.replace('.brain.json.gz', '.arch.json')):
                    if os.path.isfile(p):
                        os.remove(p)
        log(f'gen {g}: parent {os.path.basename(best)} {best_s[0]}/{len(seeds)}, {best_s[1]:.1f} moves ({time.time() - t0:.0f}s)')
    return best
