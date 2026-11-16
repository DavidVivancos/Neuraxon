# Neuraxon 2.0 on ARC-AGI-3

Python tools to **play, train and fetch** the Multi-Sphere Neuraxon 2.0 brains bred for the 25 public
[ARC-AGI-3](https://three.arcprize.org) games. The brains were bred on CPUs, with no LLM and no GPU.

- **All the brains:** [Nxon2ArcAGI3Brains](https://huggingface.co/datasets/DavidVivancos/Nxon2ArcAGI3Brains), 155,192 brains
  with their architectures.
- **Watch them play:** [NxonARCAGI3Play](https://huggingface.co/spaces/DavidVivancos/NxonARCAGI3Play), the brains running live
  in your browser.

## Setup

Python 3.12 or later.

```
cd ARCAGI3
pip install -r requirements.txt
python get_games.py                                     # the 25 games, into environment_files/
python get_brains.py get --game tn36 --level 1 --limit 3 # a few brains from the dataset, into brains/
```

## Play

```
python play_brain.py brains/tn36/<name>.brain.json.gz              # one cold try
python play_brain.py <brain> --seeds 8 --workers 4                  # the best clear among 8 seeds
python play_brain.py <brain> --level 2 --k 30 --seed 24683 --gif try.gif --show
```

A **cold try** loads the brain fresh, replays the verified moves up to its start, and lets it play, learning as it goes, until it
clears the level or uses its move cap (10 × the human moves by default). The start is the level (`--level`) and `--k`, the number
of moves of the level's verified solution replayed before the brain takes over (0: the whole level). The game and level come from
the file name (`r1170_vc33-l4x_…`: vc33, level 4) unless given.

A game over, or a RESET by the brain, replays the start (a restart). The same brain, seed and settings always give the same moves.

## Train

```
python train_brain.py newborn <arch>.arch.json --game tn36 --level 1 --episodes 4 --test 4
python train_brain.py practice <brain> --episodes 4 --test 4
python train_brain.py evolve <brain> --generations 10 --children 4 --seeds 4 --workers 4
```

- **newborn:** a new brain built from an architecture (any `.arch.json` of the dataset) lives N tries of the level, learning, and
  is saved.
- **practice:** an existing brain plays N more tries, learning, and is saved with what it learned.
- **evolve:** how the NAS 1170s brains were bred. A child is its parent with a few synapse weights mutated. A child that clears more
  seeds in cold tries, or as many in fewer moves, becomes the parent.

Every saved brain is `<stem>.brain.json.gz` with its `<stem>.arch.json` next to it.

## In Python

```python
import sys; sys.path.insert(0, 'ARCAGI3')
from nxon import player

brain = 'brains/sp80/<name>.brain.json.gz'
r = player.play(brain, seed=11003, frames=True)       # r['events']: every move, by 'replay' or 'brain'
print(r['cleared'], r['moves'], r['resets'])
best, tries = player.best_seed(brain, player.seed_list(8), workers=4)
```

## What is here

| | |
|---|---|
| `play_brain.py` | play a brain: one try, the best seed, a GIF |
| `train_brain.py` | newborn, practice and evolve |
| `get_brains.py` | list and download brains from the dataset (only the rows asked for) |
| `get_games.py` | download the 25 games in the versions the brains were bred on |
| `nxon/` | the tools as a library: `runtime`, `player`, `trainer`, `brains`, `dataset`, `games`, `render` |
| `runtime/` | the brain and its game loop: the Neuraxon 2.0 multi-sphere network (`neuraxon/`), the ARC brain (`arc_brain.py` and its behaviour modules), the game loop (`game_loop.py`), the game adapter (`arc_env/`) and their data |

`runtime/` is the brain and play layer of NAS 1178, unchanged. The architecture search itself is not included. Played through
`nxon`, it gives the same moves as the NAS: we checked 18 tries on 15 games (15,341 game steps), and the same trained brain from a
newborn's training life.

## License

MIT (see the root `LICENSE`); the Aigarth notice of the root README applies to the Aigarth hybrid features.
