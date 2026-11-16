"""Train Neuraxon 2.0 brains on an ARC-AGI-3 level.

  python train_brain.py newborn ARCH.arch.json --game tn36 --level 1 --episodes 4 --test 4
  python train_brain.py practice BRAIN.brain.json.gz --episodes 4 --test 4
  python train_brain.py evolve BRAIN.brain.json.gz --generations 10 --children 4 --seeds 4 --workers 4

newborn: a new brain built from an arch (any .arch.json of the dataset) lives N tries of the level, learning, and is saved.
practice: an existing brain plays N more tries, learning, and is saved with what it learned.
evolve: children = the brain with a few weights mutated; the one that clears more seeds (or the same in fewer moves) is kept.
"""
import argparse
import sys

from nxon import brains, player, runtime, trainer


def _test(path, game, level, k, n, cap_k, workers):
    best, tries = player.best_seed(path, player.seed_list(n), game, level, k, None, cap_k, workers)
    won = [t for t in tries if t['cleared']]
    print(f'  cold tries: {len(won)}/{len(tries)} seeds cleared' + (f", best {best['moves']} moves (seed {best['seed']})" if best else ''))


def main():
    ap = argparse.ArgumentParser(description='train brains on an ARC-AGI-3 level')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('newborn', 'practice', 'evolve'):
        p = sub.add_parser(name)
        p.add_argument('file', help='an .arch.json (newborn) or a .brain.json.gz (practice, evolve)')
        p.add_argument('--game')
        p.add_argument('--level', type=int)
        p.add_argument('--k', type=int, help='start inside the level (moves of the verified solution replayed first)')
        p.add_argument('--cap-k', type=float, default=10.0, help='move cap per try, x human moves')
        p.add_argument('--workers', type=int, default=1)
        p.add_argument('--games-dir')
        if name in ('newborn', 'practice'):
            p.add_argument('--episodes', type=int, default=4, help='tries of the training life')
            p.add_argument('--seed', type=int, default=90001)
            p.add_argument('--out', help='the brain file to write')
            p.add_argument('--test', type=int, default=0, help='then play this many cold tries (seeds)')
        else:
            p.add_argument('--generations', type=int, default=10)
            p.add_argument('--children', type=int, default=4)
            p.add_argument('--seeds', type=int, default=4, help='cold tries per brain (fixed seeds)')
            p.add_argument('--sigma', type=float, default=0.008, help='mutation size (x (|w| + 0.1))')
            p.add_argument('--frac', type=float, default=0.02, help='fraction of synapses mutated')
            p.add_argument('--out-dir', default='evolved')
            p.add_argument('--rng-seed', type=int, default=1)
    a = ap.parse_args()
    runtime.start(a.games_dir)

    if a.cmd == 'newborn':
        n = brains.parse_name(a.file)
        game, level, k = a.game or n['game'], a.level or n['level'] or 1, a.k or 0
        if not game:
            ap.error('--game is needed')
        out, s = trainer.newborn(a.file, game, level, k, a.episodes, a.seed, None, a.cap_k, a.out)
    elif a.cmd == 'practice':
        out, s = trainer.practice(a.file, a.game, a.level, a.k, a.episodes, a.seed, None, a.cap_k, a.out)
    else:
        best = trainer.evolve(a.file, a.game, a.level, a.k, a.generations, a.children, player.seed_list(a.seeds), a.sigma, a.frac,
                              None, a.cap_k, a.out_dir, a.rng_seed, a.workers)
        print(f'best brain: {best}')
        return 0
    print(f"wrote {out}: {s['episodes']} tries of {s.get('cap')} moves at most; moves of each clear (0 = not cleared): "
          f"{[int(x) for x in s['ep_actions']]}")
    if a.test:
        i = brains.parse_name(out)
        _test(out, a.game or i['game'], a.level or i['level'], s['start_step'], a.test, a.cap_k, a.workers)
    return 0


if __name__ == '__main__':
    sys.exit(main())
