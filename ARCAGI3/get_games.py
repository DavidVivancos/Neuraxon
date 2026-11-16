"""Download the 25 public ARC-AGI-3 games (the versions the brains were bred on) with the official toolkit (pip install arc-agi).

  python get_games.py                       # into ARCAGI3/environment_files
  python get_games.py --games-dir D:/arc3/environment_files --games tn36,vc33
"""
import argparse
import sys

from nxon import games, runtime


def main():
    ap = argparse.ArgumentParser(description='download the 25 ARC-AGI-3 games')
    ap.add_argument('--games-dir', default=runtime.DEFAULT_GAMES_DIR)
    ap.add_argument('--games', help='comma-separated game ids (default: all 25)')
    a = ap.parse_args()
    missing = games.download(a.games_dir, a.games.split(',') if a.games else None)
    if missing:
        print(f'missing: {", ".join(missing)}')
        return 1
    print(f'games ready in {a.games_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
