"""Get brains from the dataset https://huggingface.co/datasets/DavidVivancos/Nxon2ArcAGI3Brains (pip install huggingface_hub pyarrow).

  python get_brains.py list --game tn36 --level 1
  python get_brains.py get --game tn36 --level 1 --limit 5 --out brains
  python get_brains.py get --name "r1178_*vc33-l2*" --out brains
  python get_brains.py info brains/tn36/*.brain.json.gz

The first list/get reads the file names of the 25 shards (a few MB each) and caches them. get then reads only the rows it needs.
--source DIR uses shards you downloaded instead of the Hub.
"""
import argparse
import sys

from nxon import brains, dataset


def main():
    ap = argparse.ArgumentParser(description='get brains from the Nxon2ArcAGI3Brains dataset')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('list', 'get'):
        p = sub.add_parser(name)
        p.add_argument('--game')
        p.add_argument('--level', type=int)
        p.add_argument('--name', help='a glob on the file name, e.g. "r1178_*"')
        p.add_argument('--limit', type=int, default=0)
        p.add_argument('--source', help='a folder with the downloaded .parquet shards (default: the Hub)')
        p.add_argument('--refresh', action='store_true', help='read the shard names again')
        if name == 'get':
            p.add_argument('--out', default='brains')
    p = sub.add_parser('info')
    p.add_argument('files', nargs='+')
    a = ap.parse_args()

    if a.cmd == 'info':
        for f in a.files:
            print(brains.info(f))
        return 0
    if not (a.game or a.level or a.name):
        ap.error('give --game, --level or --name')
    idx = dataset.index(a.source, a.refresh)
    pairs = dataset.find(idx, a.game, a.level, a.name, a.limit or None)
    if a.cmd == 'list':
        for p in pairs:
            print(p[2])
        print(f'{len(pairs)} brain(s)')
        return 0
    if not pairs:
        print('no brain matches')
        return 1
    for path in dataset.fetch(pairs, a.out, a.source):
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
