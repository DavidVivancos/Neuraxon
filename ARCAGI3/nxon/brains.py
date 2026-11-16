"""Brain files: <stem>.brain.json.gz (the trained network) + <stem>.arch.json (its architecture and cell settings).

A brain file is gzipped JSON {'_meta': {...}, 'brain': <the Neuraxon multi-sphere network>}. Its name tells where it was bred:
r<NAS version>_<game>-l<level><variant>_..._t<trial>, for example r1170_vc33-l4x_e_t1742 (vc33, level 4); a _k<n>_ part is the
start inside the level (the brain plays after the first n moves of the level's verified solution). The first <game>-l<level> of a
name is its breeding line; a brain can also own other levels (pass the game and level then).
"""
import copy
import gzip
import json
import os
import re

_NAME = re.compile(r'(?:^|[_-])(?P<game>[a-z][a-z0-9]{3})-l(?P<level>\d+)')
_VER = re.compile(r'(?:^|_)[a-z]{0,2}(?P<ver>1\d{3})_')
_K = re.compile(r'_k(?P<k>\d+)(?:_|$)')


def arch_of(brain_path):
    """the arch file that goes with a brain file"""
    p = re.sub(r'\.brain\.json(\.gz)?$', '', brain_path) + '.arch.json'
    if not os.path.isfile(p):
        raise FileNotFoundError(f'{p} is missing: a brain plays only with its own arch file (same name, .arch.json)')
    return p


def parse_name(path):
    """what a brain's file name says: {'game', 'level', 'k', 'version'} (None where it says nothing)"""
    s = os.path.basename(path)
    s = re.sub(r'\.(brain|arch|record)\.json(\.gz)?$', '', s)
    m = _NAME.search(s)
    v = _VER.search(s)
    k = _K.search(s)
    return {'game': m.group('game') if m else None, 'level': int(m.group('level')) if m else None,
            'k': int(k.group('k')) if k else None, 'version': int(v.group('ver')) if v else None}


def load(path):
    """the whole brain file: {'_meta': ..., 'brain': ...}"""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        d = json.load(f)
    return d if 'brain' in d else {'_meta': {}, 'brain': d}


def save(path, payload, arch_from=None):
    """write a brain file (and copy its arch next to it when arch_from is given). Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    if arch_from:
        dst = re.sub(r'\.brain\.json(\.gz)?$', '', path) + '.arch.json'
        if os.path.abspath(dst) != os.path.abspath(arch_from):
            with open(arch_from, encoding='utf-8') as f:
                a = f.read()
            with open(dst, 'w', encoding='utf-8') as f:
                f.write(a)
    return path


def _spheres(brain):
    s = brain.get('spheres') or {}
    return list(s.values()) if isinstance(s, dict) else list(s)


def info(path):
    """a short summary of a brain file"""
    d = load(path)
    b = d['brain']
    sp = _spheres(b)
    nets = [(x or {}).get('network') or {} for x in sp]
    return {'file': os.path.basename(path), **parse_name(path),
            'spheres': len(sp),
            'neurons': sum(len(n.get('neurons') or []) for n in nets if isinstance(n.get('neurons'), list)),
            'synapses': sum(len(n.get('synapses') or []) for n in nets),
            'links': len(b.get('links') or {}),
            'steps_lived': b.get('step_count'),
            'source': (d.get('_meta') or {}).get('source'),
            'parent': (d.get('_meta') or {}).get('parent')}


def mutate(brain, rng, sigma=0.05, frac=0.1):
    """the NAS's child operator, in place: a fraction `frac` of the synapses (w_fast, w_slow, w_meta / 4) and of the
    inter-sphere link weights moves by N(0, sigma x (|w| + 0.1)). The structure (neurons, synapses, links) is unchanged."""
    n_syn = n_mut = n_lw = 0
    for s_ in _spheres(brain):
        for syn in ((s_ or {}).get('network') or {}).get('synapses') or []:
            n_syn += 1
            if isinstance(syn, dict) and rng.random() < frac:
                for key, sc in (('w_fast', 1.0), ('w_slow', 1.0), ('w_meta', 0.25)):
                    w = syn.get(key)
                    if isinstance(w, (int, float)):
                        syn[key] = float(w) + rng.gauss(0.0, sigma * sc * (abs(float(w)) + 0.1))
                n_mut += 1
    for l_ in (brain.get('links') or {}).values():
        M = (l_ or {}).get('weight_matrix')
        if isinstance(M, list):
            for row in M:
                if isinstance(row, list):
                    for j, w in enumerate(row):
                        if isinstance(w, (int, float)) and rng.random() < frac:
                            row[j] = float(w) + rng.gauss(0.0, sigma * (abs(float(w)) + 0.1))
                            n_lw += 1
    return {'synapses': n_syn, 'mutated_synapses': n_mut, 'mutated_link_weights': n_lw}


def child(parent_path, out_path, rng, sigma=0.05, frac=0.1):
    """write a mutated copy of a brain (with its arch) and return its path"""
    d = load(parent_path)
    b = copy.deepcopy(d['brain'])
    stats = mutate(b, rng, sigma, frac)
    meta = {'source': 'nxon-child', 'parent': os.path.basename(parent_path), 'mutation': {'sigma': sigma, 'frac': frac, **stats}}
    return save(out_path, {'_meta': meta, 'brain': b}, arch_from=arch_of(parent_path))


def find(folder, game=None, level=None):
    """the brain files under a folder (that have their arch), optionally of one game / level"""
    out = []
    for dp, _, fs in os.walk(folder):
        for f in fs:
            if f.endswith('.brain.json.gz'):
                p = os.path.join(dp, f)
                n = parse_name(p)
                if game and n['game'] != game:
                    continue
                if level and n['level'] != int(level):
                    continue
                if os.path.isfile(re.sub(r'\.brain\.json\.gz$', '', p) + '.arch.json'):
                    out.append(p)
    return sorted(out)
