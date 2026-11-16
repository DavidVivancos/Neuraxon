"""Play a Neuraxon 2.0 brain on its ARC-AGI-3 level: one cold try, or the best of several seeds.

  python play_brain.py brains/tn36/r1177_tn36-l1m_t181.brain.json.gz
  python play_brain.py BRAIN --seeds 8 --workers 4            # the clear with the fewest moves among 8 seeds
  python play_brain.py BRAIN --level 2 --k 30 --seed 24683 --gif try.gif

The game and level come from the brain's file name (vc33-l4x: vc33, level 4) unless given. --k: the brain plays after the first k
moves of the level's verified solution (0: the whole level). The move cap is --cap-k (10) x the human moves for that part.
"""
import argparse
import sys

from nxon import brains, player, runtime


def main():
    ap = argparse.ArgumentParser(description='play a brain on its ARC-AGI-3 level')
    ap.add_argument('brain', help='a .brain.json.gz file (its .arch.json next to it)')
    ap.add_argument('--game', help='game id, e.g. tn36 (default: from the file name)')
    ap.add_argument('--level', type=int, help='level (default: from the file name)')
    ap.add_argument('--k', type=int, help='start inside the level: moves of the verified solution replayed first (default 0)')
    ap.add_argument('--seed', type=int, default=player.SEEDS[0], help='seed of the try')
    ap.add_argument('--seeds', type=int, default=0, help='play this many seeds and keep the best clear')
    ap.add_argument('--cap', type=int, help='move cap (default: --cap-k x human moves)')
    ap.add_argument('--cap-k', type=float, default=10.0)
    ap.add_argument('--gif', help='write the try as a GIF (needs Pillow)')
    ap.add_argument('--show', action='store_true', help='print the brain moves and the last board')
    ap.add_argument('--workers', type=int, default=1, help='processes for --seeds')
    ap.add_argument('--games-dir', help='the games folder (default ARCAGI3/environment_files)')
    ap.add_argument('--fast', action='store_true', help="the runtime's faster engine (numba); moves may differ slightly")
    a = ap.parse_args()

    runtime.start(a.games_dir, fast=a.fast)
    i = brains.info(a.brain)
    game, level = a.game or i['game'], a.level or i['level'] or 1
    k = a.k if a.k is not None else (i['k'] or 0)
    print(f"{i['file']}: {i['spheres']} spheres, {i['synapses']} synapses; {game} level {level} from move {k} of "
          f"{len(runtime.solution(game, level)[1])}")
    seed = a.seed
    if a.seeds:
        best, tries = player.best_seed(a.brain, player.seed_list(a.seeds), game, level, k, a.cap, a.cap_k, a.workers)
        for t in tries:
            print(f"  seed {t['seed']}: {'cleared in ' + str(t['moves']) + ' moves' if t['cleared'] else 'not cleared'}, "
                  f"{t['resets']} restart(s)")
        if not best:
            print(f'no clear in {len(tries)} seeds (cap {tries[0]["cap"]} moves)')
            return 1
        seed = best['seed']
        print(f'best seed: {seed}')
        if not (a.gif or a.show):
            return 0
    r = player.play(a.brain, game, level, k, seed, a.cap, a.cap_k, frames=bool(a.gif or a.show))
    print(f"seed {seed}: {'CLEARED' if r['cleared'] else 'not cleared'} in {r['moves']} brain moves, {r['resets']} restart(s), "
          f"cap {r['cap']}")
    if a.show:
        moves = [e for e in r['events'] if e['by'] == 'brain']
        print(' '.join(m['action'].replace('ACTION', 'A') + (f"({m['x']},{m['y']})" if m['x'] is not None else '') for m in moves))
        last = next((e for e in reversed(r['events']) if e.get('frame')), None)
        if last:
            from nxon import render
            print(render.board(last['frame'][0]))
    if a.gif:
        from nxon import render
        render.gif(r['events'], a.gif)
        print(f'wrote {a.gif}')
    return 0 if r['cleared'] else 1


if __name__ == '__main__':
    sys.exit(main())
