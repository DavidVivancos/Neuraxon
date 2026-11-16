"""The brains dataset on Hugging Face: https://huggingface.co/datasets/DavidVivancos/Nxon2ArcAGI3Brains

25 Parquet shards, one row per file: filename, relative_path, bytes (the file's exact content). Every brain, its arch and its
trial record are there (603,054 files, about 83 GB). This module finds files by name and fetches only those rows:
  index()  reads the two name columns of every shard (a few MB per shard), cached in ~/.cache/nxon_arc3
  find()   picks brain + arch pairs by game, level or name
  fetch()  reads the content of just those rows and writes them under a folder, keeping their dataset paths

source: None = the Hub (needs `pip install huggingface_hub pyarrow`), or a local folder holding downloaded .parquet shards.
"""
import fnmatch
import gzip
import hashlib
import json
import os

from .brains import parse_name

REPO = 'DavidVivancos/Nxon2ArcAGI3Brains'
CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'nxon_arc3')


def _shards(source=None):
    """[(shard name, opener)]"""
    if source:
        names = sorted(f for f in os.listdir(source) if f.endswith('.parquet'))
        return [(n, (lambda p=os.path.join(source, n): open(p, 'rb'))) for n in names]
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    paths = sorted(p for p in fs.ls(f'datasets/{REPO}', detail=False) if p.endswith('.parquet'))
    return [(os.path.basename(p), (lambda p=p: fs.open(p, 'rb', block_size=4 << 20))) for p in paths]


def index(source=None, refresh=False, log=print):
    """{shard name: [the dataset path of each row's file]} for every shard (cached)"""
    import pyarrow.parquet as pq
    key = hashlib.sha1((os.path.abspath(source) if source else REPO).encode()).hexdigest()[:12]
    cache = os.path.join(CACHE, f'index_{key}.json.gz')
    if os.path.isfile(cache) and not refresh:
        with gzip.open(cache, 'rt', encoding='utf-8') as f:
            return json.load(f)
    out = {}
    for name, opener in _shards(source):
        with opener() as fh:
            t = pq.ParquetFile(fh).read(columns=['filename', 'relative_path'])
        rows = []
        for fn, rp in zip(t.column('filename').to_pylist(), t.column('relative_path').to_pylist()):
            fn, rp = str(fn or ''), str(rp or '').replace('\\', '/')
            rows.append(rp if rp.endswith(fn) else (rp.rstrip('/') + '/' + fn).lstrip('/'))   # the path, with the file name
        out[name] = rows
        log(f'  indexed {name}: {len(out[name])} files')
    os.makedirs(CACHE, exist_ok=True)
    with gzip.open(cache + '.tmp', 'wt', encoding='utf-8') as f:
        json.dump(out, f)
    os.replace(cache + '.tmp', cache)
    return out


def find(idx, game=None, level=None, pattern=None, limit=None):
    """brain + arch pairs: [(shard, row of the brain, its relative_path, shard, row of the arch, its relative_path)].
    game / level match the line in the file name (vc33-l4x: game vc33, level 4); pattern is a glob on the file name."""
    where = {}
    for shard, rows in idx.items():
        for i, rp in enumerate(rows):
            where[rp] = (shard, i)
    out = []
    for rp, (shard, i) in where.items():
        if not rp.endswith('.brain.json.gz'):
            continue
        name = os.path.basename(rp)
        if pattern and not fnmatch.fnmatch(name, pattern):
            continue
        if game or level:
            n = parse_name(name)
            if (game and n['game'] != game) or (level and n['level'] != int(level)):
                continue
        arp = rp[:-len('.brain.json.gz')] + '.arch.json'
        if arp in where:
            out.append((shard, i, rp) + (where[arp][0], where[arp][1], arp))
    out.sort(key=lambda r: r[2])
    return out[:limit] if limit else out


def _safe_path(out_dir, rp):
    rp = rp.replace('\\', '/').lstrip('/')
    parts = [p for p in rp.split('/') if p not in ('', '.', '..')]
    return os.path.join(out_dir, *parts)


def fetch(pairs, out_dir, source=None, log=print):
    """write the files of find()'s pairs under out_dir (their dataset paths kept); returns the brain paths written"""
    import pyarrow.parquet as pq
    want = {}
    for shard, i, rp, ashard, ai, arp in pairs:
        want.setdefault(shard, {})[i] = rp
        want.setdefault(ashard, {})[ai] = arp
    openers = dict(_shards(source))
    for shard, rows in sorted(want.items()):
        with openers[shard]() as fh:
            pf = pq.ParquetFile(fh, pre_buffer=False, buffer_size=4 << 20)    # streamed: only the row groups that hold the rows
            start = 0
            for rg in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                mine = sorted(r for r in rows if start <= r < start + n)
                if mine:
                    pos, last = start, mine[-1]
                    for batch in pf.iter_batches(batch_size=32, row_groups=[rg], columns=['bytes']):
                        col = batch.column(0)
                        for j in range(batch.num_rows):
                            if pos + j in rows:
                                dst = _safe_path(out_dir, rows[pos + j])
                                os.makedirs(os.path.dirname(dst), exist_ok=True)
                                with open(dst, 'wb') as f:
                                    f.write(col[j].as_py())
                        pos += batch.num_rows
                        if pos > last:
                            break
                start += n
        log(f'  {shard}: {len(rows)} file(s)')
    return [_safe_path(out_dir, p[2]) for p in pairs]
