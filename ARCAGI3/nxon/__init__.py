"""nxon: play, train and fetch the Neuraxon 2.0 brains of the ARC-AGI-3 games.

    from nxon import player
    r = player.play('brains/r1177_sp80-l1m_t15_103453.brain.json.gz', seed=11003)
    print(r['cleared'], r['moves'])
"""
from . import brains, player, runtime, trainer  # noqa: F401

__version__ = '1.0.0'
