"""bus1178.py -- NAS 1178 change 1: THE SENSORY BUS. Every 1178 sense in a fixed slot, grafted SILENT, grown by selection.

WHAT 1177 MEASURED (nas_1177_report.md sections 7, 8, 8b): the eyes carry the right content -- the brains selected with them play
+7.6 to +12.5 points better WITH them than without on the same seeds -- but the 1177 graft gave every brain 2 random synapses per new
input at a FIXED |w| of 0.025-0.075, 6 x louder than the median brain's own input synapses (its own scale is ~0.008 in most lineages,
~0.37 in a few). The transplanted brains lost more the louder the graft (-1.8 / -3.3 / -5.8 points), and NOT ONE eye synapse evolved:
0 of 150,234 in the 485 kept bred brains left the graft's range, none was added (the weight operator moves 1% of synapses by
sigma ~0.001; no operator grows structure at the inputs). And the wider network cost a quarter of the fleet's plays.

WHAT 1178 DOES:
  1. ONE VERSIONED LAYOUT for the appended block (the brain's own base layout is untouched and stays frozen, eyes1177.WIDTH_GENES):
       m2 8 | fdiff 8 | trans 2 | budget 2 | steer 16 | fovea 81 | mover 25 | where 16 | obj 30 | plan 6       (W78 = 194)
     every block ALWAYS present (1177 dropped trans / budget when the base had them: the layout depended on the brain), so a slot
     means the same thing in every brain; new senses (obj: the 6 most salient objects x 5 features, Reki's small / rare first;
     plan: the planner's proposal, plan1178) take new slots, never shifting an old one.
  2. THE MIGRATION (dict surgery, build time; at load in a 1178 cell for a brain that arrives without it): a 1177 brain's eye inputs
     are re-placed block by block at their bus positions WITH their synapses; the new blocks get fresh input neurons WITHOUT synapses;
     a brain that never had eyes gets all 194 inputs without synapses. GAIN (gene bus78_gain): 1 keeps the brain's 1177 eye synapses
     (the brains selected WITH the eyes in 1177), 0 removes them (every other brain: it plays as its 1176 self, input-wise).
  3. PAY ONLY FOR WHAT IS USED: an input neuron with no outgoing synapse is INACTIVE (is_active False: never stepped, never drawn
     for, skipped by the afferent pass) and the blocks with no active input are not even computed -- a brain with a silent bus costs
     what its 1176 self cost.
  4. GROWN BY SELECTION (evo1178, change 9): GROW adds afferent synapses from bus inputs to hidden neurons at the brain's OWN input
     scale (x 0.5) and activates them; SCALE multiplies one block's synapses; PRUNE removes near-zero bus synapses and deactivates
     inputs left without any. Selection keeps what pays.
  5. MEASURED: use(brain) -- per block, active inputs, synapses and their weight mass against the brain's own inputs (the advance
     file and the analysis read it).
Inert unless the brain carries the bus (brain['_bus78'], arch bus78 = 1). TALLY: st_b8_on, st_b8_migrated, st_b8_fast (steps whose
bus was computed only for its active blocks), st_b8_zero (steps with a silent bus)."""
from __future__ import annotations

import copy
import os
import random

import numpy as np

VERSION = 2
LAYOUT78 = [('m2', 8), ('fdiff', 8), ('trans', 2), ('budget', 2), ('steer', 16), ('fovea', 81), ('mover', 25), ('where', 16),
            ('obj', 30), ('plan', 6)]
W78 = sum(w for _, w in LAYOUT78)
OBJ_K = 6
OBJ_F = 5
PLAN_W = 6
GROW_SYN = 2
GROW_SCALE = 0.5
GENES = ('bus78', 'bus78_nin0', 'bus78_gain')
E77_READS = ('m2', 'fdiff', 'steer', 'mover', 'obj', 'plan')    # the blocks that read the steering's state (st._e77, the sense, the objects)
_STATE = {'migrated': 0, 'loads': 0, 'error': None, 'installed': False}


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def _int(v, d=0):
    try:
        return int(float(v)) if v not in (None, '') else int(d)
    except Exception:
        return int(d)


def on_env():
    """a 1178 cell: a brain loaded without the bus gets it (silent) before its network is built"""
    return str(os.environ.get('MULTINXON_BUS78', '0') or '0').strip() == '1'


def offsets(layout=None):
    out = {}; o = 0
    for b, w in (layout or LAYOUT78):
        out[b] = (o, w); o += w
    return out


# ======================================================================================================== the migration (dict surgery)
def is_bus(brain):
    try:
        e = brain.get('_bus78') or {}
        return int(e.get('version') or 0) == VERSION and int(e.get('width') or -1) == W78
    except Exception:
        return False


def own_scale(brain, nin0=None):
    """the mean |w_fast + w_slow| of the synapses leaving the brain's OWN (base) inputs: the scale a grown bus synapse is set to"""
    try:
        net = brain['spheres']['sensory']['network']
        if nin0 is None:
            nin0 = int((brain.get('_bus78') or brain.get('_eyes77') or {}).get('nin0') or len(net['neurons']['input']))
        ws = [abs(float(s.get('w_fast') or 0) + float(s.get('w_slow') or 0)) for s in net.get('synapses') or [] if 0 <= int(s['pre_id']) < nin0]
        return float(np.mean(ws)) if ws else 0.01
    except Exception:
        return 0.01


def migrate(brain, gain=0, rng=None):
    """re-lay a brain dict (in place) onto the 1178 bus: from a 1177 eyes brain (version 1) or from a brain without eyes.
    gain 1 keeps the 1177 eye synapses, gain 0 removes them. Returns the info dict (also stored in brain['_bus78'])."""
    if not isinstance(brain, dict) or not isinstance(brain.get('spheres'), dict) or 'sensory' not in brain['spheres']:
        raise ValueError('no sensory sphere')
    if is_bus(brain):
        return dict(brain['_bus78'], already=True)
    sp = brain['spheres']['sensory']; net = sp['network']; N = net['neurons']
    e77 = brain.get('_eyes77') or {}
    nin_now = len(N['input'])
    if int(e77.get('version') or 0) == 1:
        nin0 = int(e77['nin0']); W77 = int(e77['width']); lay77 = [tuple(x) for x in e77.get('layout') or []]
        if nin0 + W77 != nin_now or sum(w for _, w in lay77) != W77:
            raise ValueError(f'eyes record mismatch: nin {nin_now} vs {nin0}+{W77}')
        src = 'eyes77'
    else:
        nin0 = nin_now; W77 = 0; lay77 = []
        src = 'none'
    o77 = offsets(lay77); o78 = offsets()
    shift_hid = W78 - W77

    def remap(i):
        i = int(i)
        if i < nin0:
            return i
        if i < nin0 + W77:                                                  # a 1177 eye input: its block's place on the bus
            rel = i - nin0
            for b, (o, w) in o77.items():
                if o <= rel < o + w:
                    return nin0 + o78[b][0] + (rel - o)
            raise ValueError(f'eye input {i} outside the 1177 layout')
        return i + shift_hid                                                # hidden / output neurons move past the bus
    for grp in ('hidden', 'output'):
        for nd in N[grp]:
            nd['id'] = remap(nd['id'])
    old_in = {int(nd['id']): nd for nd in N['input']}
    t_in = N['input'][0]
    new_inputs = [old_in[i] for i in range(nin0)]
    for i in range(nin0, nin0 + W78):
        new_inputs.append(None)
    for i in range(nin0, nin0 + W77):                                       # the 1177 eye neurons, re-placed
        j = remap(i)
        nd = old_in[i]; nd['id'] = j
        new_inputs[j] = nd
    from eyes1177 import _fresh_input
    for j in range(nin0, nin0 + W78):
        if new_inputs[j] is None:
            new_inputs[j] = _fresh_input(t_in, j)
    syns = []
    removed = 0
    for s in net.get('synapses') or []:
        a = remap(s['pre_id']); b = remap(s['post_id'])
        if nin0 <= a < nin0 + W78 and int(gain) <= 0:
            removed += 1
            continue
        s['pre_id'] = a; s['post_id'] = b
        nb = s.get('neighbor_synapse_ids')
        if isinstance(nb, list):
            s['neighbor_synapse_ids'] = [[remap(x), remap(y)] for x, y in nb]
        syns.append(s)
    net['synapses'] = syns
    fan = {}
    for s in syns:
        fan[int(s['pre_id'])] = fan.get(int(s['pre_id']), 0) + 1
    active = 0
    for j in range(nin0, nin0 + W78):
        on = fan.get(j, 0) > 0
        new_inputs[j]['is_active'] = bool(on)
        active += int(on)
    N['input'] = new_inputs
    net['num_inputs'] = nin0 + W78
    if isinstance(net.get('parameters'), dict):
        net['parameters']['num_input_neurons'] = nin0 + W78
    itf = sp.get('interface') or {}
    base_ids = [int(x) for x in (itf.get('sensory_input_ids') or list(range(nin0))) if int(x) < nin0]
    itf['sensory_input_ids'] = base_ids + list(range(nin0, nin0 + W78))      # the base ids as they were, then the bus
    for k in ('relay_output_ids', 'readout_output_ids'):
        if isinstance(itf.get(k), list):
            itf[k] = [remap(x) for x in itf[k]]
    sp['interface'] = itf
    for lk in (brain.get('links') or {}).values():
        if not isinstance(lk, dict):
            continue
        if lk.get('source_sphere_id') == 'sensory' and isinstance(lk.get('source_output_ids'), list):
            lk['source_output_ids'] = [remap(x) for x in lk['source_output_ids']]
        if lk.get('target_sphere_id') == 'sensory' and isinstance(lk.get('target_input_ids'), list):
            lk['target_input_ids'] = [remap(x) for x in lk['target_input_ids']]
    info = {'version': VERSION, 'width': W78, 'nin0': nin0, 'layout': [list(x) for x in LAYOUT78], 'gain': int(gain), 'from': src,
            'removed': removed, 'active': active, 'scale': round(own_scale(brain, nin0), 6)}
    brain['_bus78'] = info
    brain['_eyes77'] = {'version': VERSION, 'width': W78, 'nin0': nin0, 'hidden': len(N['hidden']), 'synapses_added': 0,
                        'layout': [list(x) for x in LAYOUT78], 'bus78': True}
    return info


def bus_arch(arch, nin0, gain):
    """the arch of a bus brain: the input width, bus78 on (and eyes77 on: the eyes1177 brain hooks read the bus through it), the
    gain, the old width; the forced behaviour genes of 1177 are NOT re-forced (change 2 decides them per line)"""
    nn = arch.setdefault('neural', {})
    nn['num_input_neurons'] = int(nin0) + W78
    nn['bus78'] = 1; nn['bus78_nin0'] = int(nin0); nn['bus78_gain'] = int(gain)
    nn['eyes77'] = 1; nn['eyes77_nin0'] = int(nin0)
    mg = (arch.get('_meta') or {}).get('genes')
    if isinstance(mg, dict):
        for k in ('num_input_neurons', 'bus78', 'bus78_nin0', 'bus78_gain', 'eyes77', 'eyes77_nin0'):
            if k in mg:
                mg[k] = nn[k]
    return arch


# ======================================================================================================== growth (evo1178, change 9)
def _bus_range(brain):
    e = brain.get('_bus78') or {}
    nin0 = int(e.get('nin0') or 0)
    return nin0, nin0 + W78


def growable(genes):
    """the bus blocks growth may wire for a brain with these genes: trans / budget only when the brain's own base lacks them (else the
    base already carries them, and the base's own call must stay the first of the step: _v55_budget_channels keeps a 'previous' state)"""
    g = genes if isinstance(genes, dict) else {}
    out = []
    for b, _ in LAYOUT78:
        if b == 'trans' and _int(g.get('transient_sense'), 0):
            continue
        if b == 'budget' and _int(g.get('budget_sense'), 0):
            continue
        out.append(b)
    return out


def drivers(brain):
    """the sensory sphere's neurons whose state reaches its outputs in one synapse: the output neurons (they relay to the other spheres)
    and the hidden neurons with a synapse to an output. Measured on the 1178 champions: in 21 of 60 brains NO hidden neuron of the
    sensory sphere has a synapse to an output (the base inputs reach the outputs directly, 'io'); a synapse onto any other hidden neuron
    only moves the sphere's global activity (the neuromodulators), never the information the outputs carry."""
    net = brain['spheres']['sensory']['network']; N = net['neurons']
    outs = [int(nd['id']) for nd in N['output'] if nd.get('is_active', True)]
    so = {int(nd['id']) for nd in N['output']}
    hid = {int(nd['id']) for nd in N['hidden'] if nd.get('is_active', True)}
    h2o = sorted({int(s['pre_id']) for s in net.get('synapses') or [] if int(s['pre_id']) in hid and int(s['post_id']) in so})
    return h2o, outs


def grow(brain, rng=None, n_inputs=None, blocks=None):
    """add GROW_SYN afferent synapses from n bus inputs (default 2-6) to neurons that DRIVE the sphere's outputs (drivers(): the outputs
    and the hidden neurons with a synapse to one; random hidden neurons only when there is none), at the brain's own input scale x
    GROW_SCALE, and activate those inputs. Returns the number of synapses added."""
    rng = rng or random.Random()
    if not is_bus(brain):
        return 0
    lo, hi = _bus_range(brain)
    net = brain['spheres']['sensory']['network']; N = net['neurons']
    h2o, outs = drivers(brain)
    hid = h2o + outs or [int(nd['id']) for nd in N['hidden'] if nd.get('is_active', True)]
    if not hid:
        return 0
    o78 = offsets()
    pool = []
    for b, (o, w) in o78.items():
        if blocks is not None and b not in blocks:
            continue
        pool += list(range(lo + o, lo + o + w))
    if not pool:
        return 0
    n = int(n_inputs or rng.randint(2, 6))
    syns = net.get('synapses') or []
    tmpl = next((s for s in syns if int(s['pre_id']) < lo), syns[0] if syns else None)
    if tmpl is None:
        return 0
    sc = GROW_SCALE * own_scale(brain, lo)
    have = {(int(s['pre_id']), int(s['post_id'])) for s in syns}
    byid = {int(nd['id']): nd for nd in N['input']}
    added = 0
    for i in rng.sample(pool, min(n, len(pool))):
        for p in rng.sample(hid, min(GROW_SYN, len(hid))):
            if (i, p) in have:
                continue
            s = copy.deepcopy(tmpl)
            sg = 1.0 if rng.random() < 0.5 else -1.0
            s.update(pre_id=int(i), post_id=int(p), w_fast=0.5 * sg * sc * rng.uniform(0.5, 1.5), w_slow=0.5 * sg * sc * rng.uniform(0.5, 1.5),
                     w_meta=0.0, is_silent=False, integrity=1.0, potential_delta_w=0.0, pre_trace=0.0, post_trace=0.0, pre_trace_ltd=0.0,
                     associative_strength=0.0)
            for k in ('eligibility', 'chrono_fast_trace', 'chrono_slow_trace'):
                if k in s:
                    s[k] = 0.0
            s['neighbor_synapse_ids'] = []
            syns.append(s); have.add((i, p)); added += 1
        if i in byid:
            byid[i]['is_active'] = True
    net['synapses'] = syns
    _refresh(brain)
    return added


NEURON_IN = (2, 4)
NEURON_OUT = (1, 2)


def _fresh_syn(tmpl, pre, post, w_fast, w_slow):
    s = copy.deepcopy(tmpl)
    s.update(pre_id=int(pre), post_id=int(post), w_fast=float(w_fast), w_slow=float(w_slow), w_meta=0.0, is_silent=False, integrity=1.0,
             potential_delta_w=0.0, pre_trace=0.0, post_trace=0.0, pre_trace_ltd=0.0, associative_strength=0.0)
    for k in ('eligibility', 'chrono_fast_trace', 'chrono_slow_trace'):
        if k in s:
            s[k] = 0.0
    s['neighbor_synapse_ids'] = []
    return s


def add_neuron(brain, rng=None, blocks=None):
    """A NEW HIDDEN NEURON in the sensory sphere (change 9: 'a new hidden neuron with fan-in from a slot and fan-out to the motor readout'):
    its fan-in from 2-4 inputs of the growable bus blocks (activated) at the brain's own input scale, its fan-out to 1-2 of the sphere's
    OUTPUT neurons (they relay to the other spheres) at GROW_SCALE x that scale -- a new feature of the senses with a path to behaviour.
    The neuron is a copy of one of the sphere's hidden neurons (its time constants, thresholds, rates) with a fresh state; it takes the
    first output id, so the output neurons move up one id: their records, every synapse (and its neighbour list), the sphere interface,
    the links that leave the sphere and the network's num_hidden_neurons are renumbered, as the migration does for the bus.
    Returns {'neuron': id, 'fan_in': n, 'fan_out': n} or {} when impossible (no bus, no hidden or output neuron, ids not contiguous)."""
    rng = rng or random.Random()
    if not is_bus(brain):
        return {}
    lo, hi = _bus_range(brain)
    sp = brain['spheres']['sensory']; net = sp['network']; N = net['neurons']
    hid = N.get('hidden') or []; outs = N.get('output') or []; ins = N.get('input') or []
    nin, nh = len(ins), len(hid)
    if not hid or not outs or not isinstance(net.get('parameters'), dict):
        return {}
    if any(int(nd['id']) != nin + i for i, nd in enumerate(hid)) or any(int(nd['id']) != nin + nh + i for i, nd in enumerate(outs)):
        return {}
    o78 = offsets()
    pool = []
    for b, (o, w) in o78.items():
        if blocks is not None and b not in blocks:
            continue
        pool += list(range(lo + o, lo + o + w))
    syns = net.get('synapses') or []
    tmpl = next((s for s in syns if int(s['pre_id']) < lo), syns[0] if syns else None)
    if not pool or tmpl is None:
        return {}
    new_id = nin + nh                                                       # the first output id: the outputs move up one

    def remap(i):
        i = int(i)
        return i + 1 if i >= new_id else i
    for nd in outs:
        nd['id'] = remap(nd['id'])
    for s in syns:
        s['pre_id'] = remap(s['pre_id']); s['post_id'] = remap(s['post_id'])
        nb = s.get('neighbor_synapse_ids')
        if isinstance(nb, list):
            s['neighbor_synapse_ids'] = [[remap(x), remap(y)] for x, y in nb]
    itf = sp.get('interface') or {}
    for k in ('relay_output_ids', 'readout_output_ids'):
        if isinstance(itf.get(k), list):
            itf[k] = [remap(x) for x in itf[k]]
    for k in ('last_user_inputs', 'last_link_inputs', 'last_combined_inputs'):
        if isinstance(sp.get(k), dict):
            sp[k] = {str(remap(x)): v for x, v in sp[k].items()}
    for lk in (brain.get('links') or {}).values():
        if isinstance(lk, dict) and lk.get('source_sphere_id') == 'sensory' and isinstance(lk.get('source_output_ids'), list):
            lk['source_output_ids'] = [remap(x) for x in lk['source_output_ids']]
    # the neuron: a hidden neuron's parameters, a fresh state
    nd = copy.deepcopy(rng.choice(hid))
    nd.update(id=int(new_id), membrane_potential=0.0, trinary_state=0, adaptation=0.0, health=1.0, is_active=True, fitness_score=0.0,
              autoreceptor=0.0, last_firing_time=-1000.0, phase=rng.random() * 2.0 * np.pi)
    if 'energy_baseline' in nd:
        nd['energy_level'] = nd['energy_baseline']
    if 'state_history' in nd:
        nd['state_history'] = []
    for bd in nd.get('dendritic_branches') or []:
        if isinstance(bd, dict):
            bd['branch_potential'] = 0.0; bd['plateau_potential'] = 0.0
            if 'local_spike_history' in bd:
                bd['local_spike_history'] = []
    hid.append(nd)
    net['parameters']['num_hidden_neurons'] = int(net['parameters'].get('num_hidden_neurons') or nh) + 1
    # its synapses: from the bus, to the outputs
    sc = own_scale(brain, lo)
    out_ids = [int(x['id']) for x in outs if x.get('is_active', True)] or [int(x['id']) for x in outs]
    byid = {int(x['id']): x for x in ins}
    k_in = rng.randint(*NEURON_IN); k_out = rng.randint(*NEURON_OUT)
    fin = rng.sample(pool, min(k_in, len(pool)))
    for i in fin:
        sg = 1.0 if rng.random() < 0.5 else -1.0
        syns.append(_fresh_syn(tmpl, i, new_id, 0.5 * sg * sc * rng.uniform(0.5, 1.5), 0.5 * sg * sc * rng.uniform(0.5, 1.5)))
        if i in byid:
            byid[i]['is_active'] = True
    fout = rng.sample(out_ids, min(k_out, len(out_ids)))
    tmpo = next((s for s in syns if nin <= int(s['pre_id']) < new_id), tmpl)
    for q in fout:
        sg = 1.0 if rng.random() < 0.5 else -1.0
        syns.append(_fresh_syn(tmpo, new_id, q, 0.5 * sg * GROW_SCALE * sc * rng.uniform(0.5, 1.5),
                               0.5 * sg * GROW_SCALE * sc * rng.uniform(0.5, 1.5)))
    net['synapses'] = syns
    if isinstance(brain.get('_eyes77'), dict):
        brain['_eyes77']['hidden'] = len(hid)
    brain['_bus78']['neurons'] = int(brain['_bus78'].get('neurons') or 0) + 1
    _refresh(brain)
    return {'neuron': int(new_id), 'fan_in': len(fin), 'fan_out': len(fout)}


def scale_block(brain, block, factor):
    """multiply the synapses leaving one bus block by factor (0 removes them)"""
    if not is_bus(brain):
        return 0
    lo, _ = _bus_range(brain)
    o, w = offsets()[block]
    a, z = lo + o, lo + o + w
    net = brain['spheres']['sensory']['network']
    n = 0; keep = []
    for s in net.get('synapses') or []:
        if a <= int(s['pre_id']) < z:
            if factor <= 0:
                n += 1
                continue
            s['w_fast'] = float(s.get('w_fast') or 0) * factor; s['w_slow'] = float(s.get('w_slow') or 0) * factor
            n += 1
        keep.append(s)
    net['synapses'] = keep
    _refresh(brain)
    return n


def prune(brain, eps=None):
    """remove the bus synapses whose |w_fast + w_slow| < eps (default 5% of the brain's own scale); deactivate inputs left bare"""
    if not is_bus(brain):
        return 0
    lo, hi = _bus_range(brain)
    eps = float(eps if eps is not None else 0.05 * own_scale(brain, lo))
    net = brain['spheres']['sensory']['network']
    keep = []; n = 0
    for s in net.get('synapses') or []:
        if lo <= int(s['pre_id']) < hi and abs(float(s.get('w_fast') or 0) + float(s.get('w_slow') or 0)) < eps:
            n += 1
            continue
        keep.append(s)
    net['synapses'] = keep
    _refresh(brain)
    return n


def _refresh(brain):
    """input neurons on the bus are active iff they have an outgoing synapse; the record's counters follow"""
    lo, hi = _bus_range(brain)
    net = brain['spheres']['sensory']['network']
    fan = set(int(s['pre_id']) for s in net.get('synapses') or [] if lo <= int(s['pre_id']) < hi)
    act = 0
    for nd in net['neurons']['input']:
        i = int(nd['id'])
        if lo <= i < hi:
            nd['is_active'] = i in fan
            act += int(i in fan)
    brain['_bus78']['active'] = act


def use(brain):
    """per block: active inputs, synapses, mean |w| and the block's share of the input weight mass (base + bus)"""
    out = {}
    if not is_bus(brain):
        return out
    lo, hi = _bus_range(brain)
    net = brain['spheres']['sensory']['network']
    base = [abs(float(s.get('w_fast') or 0) + float(s.get('w_slow') or 0)) for s in net.get('synapses') or [] if 0 <= int(s['pre_id']) < lo]
    mass_base = float(sum(base))
    for b, (o, w) in offsets().items():
        a, z = lo + o, lo + o + w
        ws = [abs(float(s.get('w_fast') or 0) + float(s.get('w_slow') or 0)) for s in net.get('synapses') or [] if a <= int(s['pre_id']) < z]
        ins = len(set(int(s['pre_id']) for s in net.get('synapses') or [] if a <= int(s['pre_id']) < z))
        out[b] = {'inputs': ins, 'syn': len(ws), 'mean_w': round(float(np.mean(ws)), 5) if ws else 0.0,
                  'share': round(float(sum(ws)) / max(1e-12, mass_base + float(sum(ws))), 4) if ws else 0.0}
    out['_base'] = {'syn': len(base), 'mean_w': round(float(np.mean(base)), 5) if base else 0.0}
    return out


# ======================================================================================================== the brain side
def _obj_block(brain, G, st):
    """the 6 most salient objects (Reki: small and rare first) x [present, colour rarity, size, row, what clicking its type did]"""
    v = [0] * (OBJ_K * OBJ_F)
    try:
        from steer1162 import components, sbk
        bg = int(getattr(st, 'bg', -1)) if st is not None else -1
        if bg < 0:
            bg = int(np.bincount(np.clip(G.ravel(), 0, 15), minlength=16).argmax())
        O = None
        if st is not None:
            try:
                k = st.key(G)
                O = (getattr(st, 'objs', None) or {}).get(k)
            except Exception:
                O = None
        if O is None:
            ign = st.ignore() if (st is not None and hasattr(st, 'ignore')) else np.zeros(G.shape, bool)
            O = components(G, bg, ign)
        if not O:
            return v
        cnt = {}
        for o in O:
            cnt[int(o['col'])] = cnt.get(int(o['col']), 0) + 1
        H = G.shape[0]
        Os = sorted(O, key=lambda o: (cnt[int(o['col'])], int(o['n']), int(o['cy']), int(o['cx'])))[:OBJ_K]
        sig_n = (getattr(st, 'sig_n', None) or {}) if st is not None else {}
        sig_chg = (getattr(st, 'sig_chg', None) or {}) if st is not None else {}
        prior = getattr(brain, '_p78_prior', None)
        for i, o in enumerate(Os):
            c = int(o['col']); n = int(o['n'])
            rar = 1 if cnt[c] == 1 else (-1 if cnt[c] >= 4 else 0)
            sz = -1 if n <= 4 else (1 if n >= 40 else 0)
            row = -1 if o['cy'] < H / 3.0 else (1 if o['cy'] > 2 * H / 3.0 else 0)
            eff = 0
            for sig, nn in sig_n.items():
                if isinstance(sig, tuple) and len(sig) >= 3 and sig[0] == 'C' and sig[1] == c and sig[2] == sbk(n) and nn:
                    eff = 1 if sig_chg.get(sig, 0) else -1
                    break
            if eff == 0 and prior is not None:
                try:
                    p = prior.p_effect(c, n, cnt[c], o)
                    eff = 1 if (p is not None and p >= 0.5) else (-1 if (p is not None and p < 0.15) else 0)
                except Exception:
                    pass
            v[i * OBJ_F:(i + 1) * OBJ_F] = [1, rar, sz, row, eff]
    except Exception as e:
        try:
            brain._b78_error = f'obj: {type(e).__name__}: {e}'[:160]
        except Exception:
            pass
    return v


def _plan_block(brain, st):
    """the planner's running plan (plan1178, st._pl78): [running, its remaining length, next is a click / a key, predicted drop,
    confidence, near the goal]"""
    v = [0] * PLAN_W
    try:
        P = getattr(st, '_pl78', None) if st is not None else None
        if not P or not P.get('plan'):
            return v
        plan = P['plan']; i = int(P.get('i', 0))
        if i >= len(plan):
            return v
        nxt = plan[i]
        L = len(plan) - i
        v[0] = 1
        v[1] = -1 if L <= 2 else (1 if L >= 6 else 0)
        v[2] = 1 if (nxt and nxt[0] and nxt[0][0] == 'C') else -1
        dh = P.get('dh')
        v[3] = 1 if (dh is not None and dh < 0) else (-1 if (dh is not None and dh > 0) else 0)
        cf = P.get('conf')
        v[4] = 1 if (cf is not None and cf >= 0.66) else (-1 if (cf is not None and cf < 0.33) else 0)
        h1 = P.get('h_end')
        v[5] = 1 if (h1 is not None and h1 <= 1) else 0
    except Exception:
        pass
    return v


def install(ArcBrain):
    """eyes1177 reads the bus: its layout() returns the bus layout for a bus brain, its channels gain the obj / plan blocks, and only
    the blocks with an active input are computed (a silent bus returns zeros at once)"""
    if getattr(ArcBrain, '_v1178_bus', False):
        return True
    import eyes1177 as E7
    ArcBrain._v1178_bus = True
    _lay0 = E7.layout
    E7.layout0 = _lay0

    def layout(genes):
        try:
            g = genes.get('bus78') if isinstance(genes, dict) else getattr(genes, 'bus78', None)
            if _int(g, 0) == 1:
                return list(LAYOUT78)
        except Exception:
            pass
        return _lay0(genes)
    E7.layout = layout
    E7.width = lambda genes: sum(w for _, w in layout(genes))
    _isr0 = E7.is_rewired

    def is_rewired(brain, genes):
        if is_bus(brain):
            return True
        return _isr0(brain, genes)
    E7.is_rewired = is_rewired
    _ch0 = E7.eyes_channels

    def eyes_channels(brain, frame, grid):
        W = int(getattr(brain, '_e77_w', 0) or 0)
        if W <= 0:
            return []
        act = getattr(brain, '_b78_act', None)
        if getattr(brain, '_e77_silent', False) or (act is not None and not any(act.values())):
            _tally('b8_zero')
            return [0] * W
        if act is not None and getattr(brain, '_b78', False):
            # compute only the active blocks: the original routine for its blocks (with the inactive ones skipped), ours for obj / plan
            lay = brain._e77_layout
            keep = [(b, w) for b, w in lay if act.get(b)]
            saved = brain._e77_layout
            out = []
            try:
                brain._e77_layout = [(b, w) for b, w in keep if b not in ('obj', 'plan')]
                part = _ch0(brain, frame, grid) if brain._e77_layout else []
            finally:
                brain._e77_layout = saved
            pi = 0
            st = getattr(brain, '_steer', None)
            G = None
            for b, w in lay:
                if not act.get(b):
                    out.extend([0] * w)
                    continue
                if b == 'obj':
                    if G is None:
                        G = np.asarray(grid, dtype=np.int16)
                    out.extend(_obj_block(brain, G, st))
                elif b == 'plan':
                    out.extend(_plan_block(brain, st))
                else:
                    out.extend(part[pi:pi + w]); pi += w
            _tally('b8_fast')
            return (out + [0] * W)[:W]
        return _ch0(brain, frame, grid)
    E7.eyes_channels = eyes_channels
    ArcBrain._eyes77_channels = eyes_channels
    _init0 = ArcBrain.__init__

    def __init__(self, params, *a, **k):
        _init0(self, params, *a, **k)
        self._b78 = False; self._b78_act = None
        try:
            info = getattr(getattr(self, 'brain', None), '_bus78', None)
            nin0 = 0
            if isinstance(info, dict) and _int(info.get('width'), 0) == W78:
                nin0 = _int(info.get('nin0'), 0)
            elif _int(getattr(params, 'bus78', 0), 0) == 1:
                nin0 = _int(getattr(params, 'bus78_nin0', 0), 0) or _int(getattr(params, 'eyes77_nin0', 0), 0)
            if nin0 > 0 and int(self.n_in) == nin0 + W78:
                self._b78 = True
                self._e77_w = W78; self._e77_layout = list(LAYOUT78); self._n_in_base = nin0
                if str(os.environ.get('MULTINXON_EYES77', '0')) != '1':
                    # 1178: the behaviour genes are the brain's own (1177 forced the settled view on every rewired brain)
                    self._settled_view = _int(getattr(params, 'settled_view', 0), 0)
                net = self.sensory.network
                active = set()
                for nd in getattr(net, 'input_neurons', []) or []:
                    try:
                        i = int(nd.id)
                    except Exception:
                        continue
                    if nin0 <= i < nin0 + W78 and getattr(nd, 'is_active', True):
                        active.add(i - nin0)
                acts = {}
                for b, (o, w) in offsets().items():
                    acts[b] = any((o + j) in active for j in range(w))
                self._b78_act = acts
                st = getattr(self, '_steer', None)
                pin = str(os.environ.get('MULTINXON_PIN_BUS78', '') or '').strip()
                silent = (pin == '0') if pin != '' else (_int(getattr(params, 'bus78_gain', 1), 1) <= 0)
                if silent and any(acts.values()):
                    self._e77_silent = True                                # the bus off (gene bus78_gain 0 or the bench's bus-off arm): zeros
                    _tally('b8_silenced')
                if st is not None and not silent and any(acts.get(b) for b in E77_READS):
                    st.e77_on = True                                       # the blocks read the steering's 1177 observations
                    if acts.get('steer'):
                        st.sense_w = 16                                    # the steer block reads the full sense
                mute = [x.strip() for x in str(os.environ.get('MULTINXON_BUS78_MUTE', '') or '').split(',') if x.strip()]
                if mute and any(acts.get(b) for b in mute):
                    # the gate's block screen (nas1178 blk78): the network silenced on these blocks -- they read zeros; the steering's
                    # 1177 machinery stays as the wiring set it, so the screen measures what the NETWORK takes from each block
                    self._b78_act = {b: (bool(v) and b not in mute) for b, v in acts.items()}
                    _tally('b8_muted')
                _tally('b8_on')
                if any(acts.values()):
                    _tally('b8_live')
        except Exception as e:
            self._b78_error = f'init: {type(e).__name__}: {e}'[:160]
    ArcBrain.__init__ = __init__
    _STATE['installed'] = True
    return True


def install_loader():
    """a brain loaded in a 1178 cell (MULTINXON_BUS78 = 1) without the bus is migrated (silent) before its network is built; the arch
    it was built from must say bus78 (the NAS's derive / the fleet's force genes set it)"""
    try:
        import neuraxon.multisphere as MS
    except Exception as e:                                                       # pragma: no cover
        _STATE['error'] = f'loader: {type(e).__name__}: {e}'
        return False
    f0 = MS.load_multisphere_from_dict
    if getattr(f0, '_v1178', False):
        return True

    def load_multisphere_from_dict(data, rebuild_net_fn=None, *a, **k):
        if on_env() and isinstance(data, dict) and isinstance(data.get('spheres'), dict) and 'sensory' in data['spheres'] \
                and not is_bus(data):
            try:
                migrate(data, gain=0)
                _STATE['migrated'] += 1; _tally('b8_migrated')
            except Exception as e:
                _STATE['error'] = f'migrate: {type(e).__name__}: {e}'[:160]
        _STATE['loads'] += 1
        obj = f0(data, rebuild_net_fn, *a, **k)
        try:
            if isinstance(data, dict) and is_bus(data):
                obj._bus78 = dict(data['_bus78'])                         # ArcBrain reads the bus from the brain it was built from
        except Exception:
            pass
        return obj

    load_multisphere_from_dict._v1178 = True
    load_multisphere_from_dict.__wrapped__ = f0
    MS.load_multisphere_from_dict = load_multisphere_from_dict
    return True


# ======================================================================================================== the build: a brain's files on the bus
def migrate_files(b_in, a_in, b_out, a_out, gain, arch_fn=None):
    """read a brain + arch, migrate the brain onto the bus (gain 1 keeps its 1177 eye synapses), write both (arch_fn(arch, info) may set
    the configuration). Returns the info dict."""
    import gzip
    import json
    with gzip.open(b_in, 'rt', encoding='utf-8') as fh:
        payload = json.load(fh)
    brain = payload.get('brain', payload) if isinstance(payload, dict) else payload
    arch = json.load(open(a_in, encoding='utf-8'))
    info = migrate(brain, gain=gain)
    bus_arch(arch, info['nin0'], gain)
    if arch_fn is not None:
        arch_fn(arch, info)
    if isinstance(payload, dict) and isinstance(payload.get('_meta'), dict):
        payload['_meta']['bus78'] = {k: info.get(k) for k in ('version', 'width', 'nin0', 'gain', 'from', 'removed', 'active')}
    with gzip.open(b_out, 'wt', encoding='utf-8', compresslevel=6) as fh:
        json.dump(payload, fh)
    with open(a_out, 'w', encoding='utf-8') as fh:
        json.dump(arch, fh, indent=1)
    return info
