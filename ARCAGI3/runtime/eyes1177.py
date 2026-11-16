"""eyes1177.py -- NAS 1177 change 1: THE EYES MILESTONE 2 SAYS A PLAYER NEEDS, FORCED ON EVERY BRAIN, UPSCALED WHERE IT LOOKS.

WHAT MILESTONE 2 MEASURED (dfranzen's write-up, ARC-AGI-3 milestone 2, 27.89 on the public leaderboard): the largest perception gains
came from WHAT THE AGENT SEES -- the image scale 4x -> 10x (small details visible), the frames between the action and the settled
board (animations; tn36 "essentially unsolvable" without), difference images (the changed pixels), the board after a game over
(the death frame), the interior board apart from its 4-cell border (gameplay_changed vs board_changed), and an object-level
frame_diff (moved / appeared / disappeared / recoloured / resized / rotated).

WHAT THE 1176 BRAINS SAW (nas_1176_report.md section 15, all 288,317 trial archs): the board pooled to a 6 x 6 retina (one input per
10.7 x 10.7 cells), the full-resolution fovea off on 92%, the change tracker and the death record off on 100%, the transient /
object / budget senses absent on 84%, and 403 of 511 book brains reading frame[0] (the board BEFORE the move on animated games).
The senses exist as genes but cannot be switched on in a trained brain: they are INSERTED in the middle of the encoder, every later
channel shifts and the evolved weights read the wrong inputs -- so evolution never kept one.

WHAT 1177 DOES (no twin decides it: the capability is forced, the gate measures it against a control):
  1. THE REWIRE (build time, every brain shipped; run time, any brain loaded in a 1177 cell without it): the sensory sphere gets
     WIDTH more input neurons, APPENDED after the brain's own inputs (ids n_in0 ..; the hidden and output neurons, their synapses, the
     sphere interface and the links renumbered), each with EYES_SYN small afferent synapses (+-EYES_W) to the sphere's hidden
     neurons that plasticity and mutation can grow. The brain's own channels keep their positions and weights: the old network is
     untouched, it gains inputs. Arch: num_input_neurons += WIDTH, eyes77 = 1, eyes77_nin0 = the old width.
  2. THE EYES BLOCK (appended to the encoder after the brain's own channels, every value trinary, mostly 0 = sparse):
       m2      8  the milestone-2 state: the last attempt died (+1 fatal action / -1 budget ran out / 0), the budget used this attempt,
                  undo offered and reliable, the last action was an undo, a guard fired, the last action animated (+1 transient
                  pixels / 0 animated / -1 settled at once), a NEW object type on this level, the last prediction hit / missed
       fdiff   8  frame_diff of the last action from the steering's object model: moved, went to the click, recoloured, resized,
                  vanished, appeared, rotated, cells changed (log bucket)
       trans   2  the animation sense (v1.155) -- only when the brain's own encoder lacks it
       budget  2  the budget-bar sense (v1.155) -- only when the brain's own encoder lacks it
       steer  16  what the steering knows (novel state, untried share, no-op, transient, mover dy / dx, goal alignment, hidden
                  state, new atoms, miss, predicted value, probe phase, goal on, h, dh, clock / wake)
       fovea  81  a 9 x 9 FULL-RESOLUTION patch at the brain's cursor (the selected click candidate): 0 background, -1 the colour
                  under the cursor, +1 any other colour -- one input per cell instead of one per 10.7 x 10.7 block (the upscale)
       mover  25  a 5 x 5 full-resolution patch on the object the keys move (the steering's mover): 0 free, -1 the mover, +1 an
                  obstacle / target -- "what is ahead of the avatar", the cell a fatal move targets
       where  16  the difference image, 4 x 4: +1 where the interior changed since the previous frame
  3. FORCED BEHAVIOUR GENES (no width change): settled_view 1 (read frame[-1], the board the action produced), change_border 4 and
     hud_border 4 (change = the interior, the 4-cell border excluded), death_memory 1 (fatal deaths only: steer1177's diagnosis).
     The brain's own width genes (retina, delta, fovea, colour, the v1.155 senses, steer_sense ...) are never changed: they define
     the base layout the weights were bred on.
Inert unless the brain is rewired AND eyes77 is on (gene, or MULTINXON_EYES77 = 1 in a 1177 cell); MULTINXON_EYES77_OFF = 1 plays a
rewired brain with the eyes block silent (the gate's control arm). TALLY: st_e7_on, st_e7_rewired (run-time rewires), st_e7_missing."""
from __future__ import annotations

import copy
import hashlib
import os
import random

import numpy as np

VERSION = 1
FOVEA_K = 9
MOVER_K = 5
WHERE_N = 4
STEER_W = 16
EYES_SYN = 2
EYES_W = 0.05
FORCED_BEHAVIOUR = {'settled_view': 1, 'change_border': 4, 'hud_border': 4, 'death_memory': 1}
# the genes that define the brain's own encoder layout: never flipped in a rewired brain (evo1177 freezes them)
WIDTH_GENES = ('retina_size', 'retina_saliency_levels', 'colour_channels', 'fovea_patch_size', 'fovea_patch_colour', 'efference_copy',
               'delta_retina_size', 'change_track', 'select_phase', 'effect_fovea', 'agency_sense', 'budget_sense', 'object_sense',
               'transient_sense', 'steer_sense', 'level_context_scale', 'max_level_context', 'num_input_neurons')
_STATE = {'rewired': 0, 'loads': 0, 'error': None, 'loader': False, 'brain': False}


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
    return str(os.environ.get('MULTINXON_EYES77', '0') or '0').strip() == '1'


def off_env():
    return str(os.environ.get('MULTINXON_EYES77_OFF', '0') or '0').strip() == '1'


def ctrl_env():
    return str(os.environ.get('MULTINXON_V1177_CTRL', '0') or '0').strip() == '1'


# ======================================================================================================== the layout
def base_has(genes, name):
    try:
        return _int((genes or {}).get(name) if isinstance(genes, dict) else getattr(genes, name, 0), 0) > 0
    except Exception:
        return False


def layout(genes):
    """[(block, width)] of the eyes block for a brain with these genes (a dict or a params object)"""
    out = [('m2', 8), ('fdiff', 8)]
    if not base_has(genes, 'transient_sense'):
        out.append(('trans', 2))
    if not base_has(genes, 'budget_sense'):
        out.append(('budget', 2))
    out += [('steer', STEER_W), ('fovea', FOVEA_K * FOVEA_K), ('mover', MOVER_K * MOVER_K), ('where', WHERE_N * WHERE_N)]
    return out


def width(genes):
    return sum(w for _, w in layout(genes))


# ======================================================================================================== the rewire (dict surgery)
def _fresh_input(t, nid):
    nd = copy.deepcopy(t)
    nd['id'] = int(nid)
    for k, v in (('membrane_potential', 0.0), ('trinary_state', 0), ('adaptation', 0.0), ('fitness_score', 0.0), ('last_firing_time', -1000.0),
                 ('autoreceptor', 0.0), ('health', 1.0), ('is_active', True)):
        if k in nd:
            nd[k] = v
    if 'state_history' in nd:
        nd['state_history'] = []
    for b in nd.get('dendritic_branches') or []:
        if isinstance(b, dict):
            for k in ('branch_potential', 'plateau_potential'):
                if k in b:
                    b[k] = 0.0
            if 'local_spike_history' in b:
                b['local_spike_history'] = [0.0] * len(b['local_spike_history'] or [])
    for k in ('dsn_input_buffer',):
        if isinstance(nd.get(k), list):
            nd[k] = [0.0] * len(nd[k])
    return nd


def brain_seed(brain):
    """a stable seed from the brain's own synapses (the same brain is rewired the same way wherever it is loaded)"""
    h = hashlib.md5()
    try:
        net = brain['spheres']['sensory']['network']
        for s in (net.get('synapses') or [])[:64]:
            h.update(f"{s.get('pre_id')},{s.get('post_id')},{round(float(s.get('w_fast') or 0), 6)};".encode())
        h.update(str(len(net.get('synapses') or [])).encode())
    except Exception:
        h.update(b'?')
    return int(h.hexdigest()[:12], 16)


def is_rewired(brain, genes):
    """True when the sensory sphere already carries the eyes inputs for these genes (recorded by the rewire)"""
    try:
        e = brain.get('_eyes77') or {}
        return int(e.get('version') or 0) == VERSION and int(e.get('width') or -1) == width(genes)
    except Exception:
        return False


def rewire(brain, genes, seed=None, n_syn=EYES_SYN, w=EYES_W):
    """append width(genes) input neurons to the sensory sphere of a brain dict (in place); returns the info dict"""
    if not isinstance(brain, dict) or not isinstance(brain.get('spheres'), dict) or 'sensory' not in brain['spheres']:
        raise ValueError('no sensory sphere')
    if is_rewired(brain, genes):
        return dict(brain['_eyes77'], already=True)
    W = width(genes)
    rng = random.Random(brain_seed(brain) if seed is None else int(seed))
    sp = brain['spheres']['sensory']; net = sp['network']; N = net['neurons']
    nin0 = len(N['input']); nh = len(N['hidden'])
    shift = lambda i: int(i) + W if int(i) >= nin0 else int(i)
    for grp in ('hidden', 'output'):
        for nd in N[grp]:
            nd['id'] = shift(nd['id'])
    syns = net.get('synapses') or []
    for s in syns:
        s['pre_id'] = shift(s['pre_id']); s['post_id'] = shift(s['post_id'])
        nb = s.get('neighbor_synapse_ids')
        if isinstance(nb, list):
            s['neighbor_synapse_ids'] = [[shift(a), shift(b)] for a, b in nb]
    t_in = N['input'][0]
    new_ids = list(range(nin0, nin0 + W))
    N['input'] = list(N['input']) + [_fresh_input(t_in, i) for i in new_ids]
    hid = [nd['id'] for nd in N['hidden']] or [nd['id'] for nd in N['output']]
    tmpl = next((s for s in syns if int(s['pre_id']) < nin0), syns[0] if syns else None)
    added = 0
    if tmpl is not None and hid and n_syn > 0:
        for i in new_ids:
            posts = rng.sample(hid, min(len(hid), n_syn))
            group = []
            for p in posts:
                s = copy.deepcopy(tmpl)
                sg = 1.0 if rng.random() < 0.5 else -1.0
                s.update(pre_id=int(i), post_id=int(p), w_fast=sg * w * rng.uniform(0.5, 1.5), w_slow=sg * w * rng.uniform(0.5, 1.5),
                         w_meta=0.0, is_silent=False, integrity=1.0, potential_delta_w=0.0, pre_trace=0.0, post_trace=0.0, pre_trace_ltd=0.0,
                         associative_strength=0.0)
                for k in ('eligibility', 'chrono_fast_trace', 'chrono_slow_trace'):
                    if k in s:
                        s[k] = 0.0
                group.append(s)
            for s in group:
                s['neighbor_synapse_ids'] = [[int(o['pre_id']), int(o['post_id'])] for o in group if o is not s]
            syns.extend(group); added += len(group)
    net['synapses'] = syns
    net['num_inputs'] = nin0 + W
    if isinstance(net.get('parameters'), dict):
        net['parameters']['num_input_neurons'] = nin0 + W
    itf = sp.get('interface') or {}
    itf['sensory_input_ids'] = list(itf.get('sensory_input_ids') or list(range(nin0))) + new_ids
    for k in ('relay_output_ids', 'readout_output_ids'):
        if isinstance(itf.get(k), list):
            itf[k] = [shift(x) for x in itf[k]]
    sp['interface'] = itf
    for lk in (brain.get('links') or {}).values():
        if not isinstance(lk, dict):
            continue
        if lk.get('source_sphere_id') == 'sensory' and isinstance(lk.get('source_output_ids'), list):
            lk['source_output_ids'] = [shift(x) for x in lk['source_output_ids']]
        if lk.get('target_sphere_id') == 'sensory' and isinstance(lk.get('target_input_ids'), list):
            lk['target_input_ids'] = [shift(x) for x in lk['target_input_ids']]
    info = {'version': VERSION, 'width': W, 'nin0': nin0, 'hidden': nh, 'synapses_added': added, 'layout': layout(genes)}
    brain['_eyes77'] = info
    return info


def rewire_arch(arch, nin0, W):
    """the arch of a rewired brain: the new input width, eyes77 on, the old width, the forced behaviour genes"""
    nn = arch.setdefault('neural', {})
    nn['eyes77_b0'] = {k: nn.get(k) for k in FORCED_BEHAVIOUR}             # the 1176 values (the bench's control plays with them)
    nn['num_input_neurons'] = int(nin0) + int(W)
    nn['eyes77'] = 1; nn['eyes77_nin0'] = int(nin0)
    nn.update(FORCED_BEHAVIOUR)
    mg = (arch.get('_meta') or {}).get('genes')
    if isinstance(mg, dict):
        for k in ('num_input_neurons', 'eyes77', 'eyes77_nin0') + tuple(FORCED_BEHAVIOUR):
            if k in mg:
                mg[k] = nn[k]
    return arch


# ======================================================================================================== the loader (run-time rewire)
def install_loader():
    """a brain loaded in a 1177 cell (MULTINXON_EYES77 = 1) without the eyes inputs is rewired before it is built (the layout from the
    arch loaded just before: build_params_from_arch always precedes the load)"""
    try:
        import neuraxon.multisphere as MS
    except Exception as e:                                                       # pragma: no cover
        _STATE['error'] = f'loader: {type(e).__name__}: {e}'
        return False
    f0 = MS.load_multisphere_from_dict
    if getattr(f0, '_v1177', False):
        return True

    def load_multisphere_from_dict(data, rebuild_net_fn=None, *a, **k):
        if on_env() and isinstance(data, dict) and isinstance(data.get('spheres'), dict) and 'sensory' in data['spheres']:
            try:
                import architecture as AR
                genes = {g: AR.get_param('neural', g, None) for g in ('transient_sense', 'budget_sense', 'eyes77_nin0')}
                net = data['spheres']['sensory']['network']
                nin = len(net['neurons']['input'])
                nin0 = _int(genes.get('eyes77_nin0'), 0)
                if not is_rewired(data, genes) and not (nin0 and nin == nin0 + width(genes)):
                    rewire(data, genes)
                    _STATE['rewired'] += 1; _tally('e7_rewired')
            except Exception as e:
                _STATE['error'] = f'rewire: {type(e).__name__}: {e}'[:160]
        _STATE['loads'] += 1
        return f0(data, rebuild_net_fn, *a, **k)

    load_multisphere_from_dict._v1177 = True
    load_multisphere_from_dict.__wrapped__ = f0
    MS.load_multisphere_from_dict = load_multisphere_from_dict
    _STATE['loader'] = True
    return True


# ======================================================================================================== the eyes block
def _patch(G, cy, cx, K, ref, bg):
    """K x K full-resolution patch centred at (cy, cx): 0 background / off-board, -1 colour ref, +1 any other colour"""
    H, W = G.shape
    h = K // 2
    y0, y1, x0, x1 = cy - h, cy + h + 1, cx - h, cx + h + 1
    out = np.zeros((K, K), np.int8)
    ys0, ys1, xs0, xs1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
    if ys0 >= ys1 or xs0 >= xs1:
        return out.ravel().tolist()
    sub = G[ys0:ys1, xs0:xs1]
    v = np.where(sub == bg, 0, np.where(sub == ref, -1, 1)).astype(np.int8)
    out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = v
    return out.ravel().tolist()


FD_KINDS = ('move', 'to_click', 'recolour', 'resize', 'vanish', 'appear', 'rotate')


def eyes_channels(brain, frame, grid):
    """the eyes block for this frame (len == brain._e77_w); the steering has already observed the last transition"""
    W = int(getattr(brain, '_e77_w', 0) or 0)
    if W <= 0:
        return []
    if getattr(brain, '_e77_silent', False):
        return [0] * W
    st = getattr(brain, '_steer', None)
    try:
        G = np.asarray(grid, dtype=np.int16)
        if G.ndim != 2 or not G.size:
            return [0] * W
    except Exception:
        return [0] * W
    H_, W_ = G.shape
    bg = int(getattr(st, 'bg', -1)) if st is not None else -1
    if bg < 0:
        bg = int(np.bincount(np.clip(G.ravel(), 0, 15), minlength=16).argmax())
    E = (getattr(st, '_e77', None) or {}) if st is not None else {}
    out = []
    for blk, n in brain._e77_layout:
        try:
            if blk == 'm2':
                died = int(getattr(brain, '_e77_died', 0) or 0)
                pace = E.get('pace')
                und = E.get('undo_state', -1 if not getattr(brain, '_e77_undo_offered', False) else 0)
                la7 = 1 if int(getattr(getattr(brain, 'last_action', None), 'value', 0) or 0) == 7 else -1
                v = [died, (0 if pace is None else (1 if pace > 0.66 else (-1 if pace < 0.33 else 0))), int(und), la7,
                     1 if E.get('guard') else 0, int(E.get('anim', -1)), 1 if E.get('newelem') else 0, int(E.get('pred', 0))]
            elif blk == 'fdiff':
                fd = E.get('fdiff') or {}
                v = [1 if fd.get(k) else 0 for k in FD_KINDS]
                nc = int(E.get('nchg', 0) or 0)
                v.append(0 if nc == 0 else (1 if nc > 16 else -1))
            elif blk == 'trans':
                v = list(brain._v55_transient_channels())
            elif blk == 'budget':
                v = list(brain._v55_budget_channels(grid))
            elif blk == 'steer':
                v = [0.0] * n
                if st is not None and hasattr(st, 'sense'):
                    s = [float(x) for x in st.sense()]
                    v = (s + [0.0] * n)[:n]
                v = [1 if x > 0.25 else (-1 if x < -0.25 else 0) for x in v]
            elif blk == 'fovea':
                cy, cx = brain.cursor if getattr(brain, 'cursor', None) else (H_ // 2, W_ // 2)
                cy = max(0, min(H_ - 1, int(cy))); cx = max(0, min(W_ - 1, int(cx)))
                v = _patch(G, cy, cx, FOVEA_K, int(G[cy, cx]), bg)
            elif blk == 'mover':
                mp = E.get('mover')
                if mp is None:
                    v = [0] * n
                else:
                    my, mx = max(0, min(H_ - 1, int(round(mp[0])))), max(0, min(W_ - 1, int(round(mp[1]))))
                    v = _patch(G, my, mx, MOVER_K, int(E.get('mover_col', G[my, mx])), bg)
            elif blk == 'where':
                P = getattr(brain, '_e77_prev', None)
                brain._e77_prev = G
                v = [0] * n
                if P is not None and P.shape == G.shape:
                    m = 4
                    D = (P != G)
                    D[:m, :] = False; D[-m:, :] = False; D[:, :m] = False; D[:, -m:] = False
                    if D.any():
                        bh, bw = H_ // WHERE_N, W_ // WHERE_N
                        for i in range(WHERE_N):
                            for j in range(WHERE_N):
                                if D[i * bh:(i + 1) * bh if i < WHERE_N - 1 else H_, j * bw:(j + 1) * bw if j < WHERE_N - 1 else W_].any():
                                    v[i * WHERE_N + j] = 1
            else:
                v = [0] * n
        except Exception as e:
            brain._e77_error = f'{blk}: {type(e).__name__}: {e}'[:160]
            v = [0] * n
        out.extend((list(v) + [0] * n)[:n])
    return out


# ======================================================================================================== the brain
def install(ArcBrain):
    """ArcBrain: the eyes are on when the loaded network carries them (n_in == base + width) and eyes77 is on (gene or env)"""
    if getattr(ArcBrain, '_v1177_eyes', False):
        return True
    ArcBrain._v1177_eyes = True
    _init0 = ArcBrain.__init__
    _ps0 = ArcBrain.play_step
    _rs0 = ArcBrain.reset
    _srs0 = ArcBrain.soft_reset

    def __init__(self, params, *a, **k):
        _init0(self, params, *a, **k)
        self._e77_w = 0; self._e77_layout = []; self._e77_prev = None; self._e77_died = 0; self._e77_died_t = 0
        self._e77_silent = off_env(); self._e77_undo_offered = False; self._n_in_base = None
        want = _int(getattr(params, 'eyes77', 0), 0) == 1 or on_env()
        if not want:
            return
        lay = layout(params); W = sum(w for _, w in lay)
        nin0 = _int(getattr(params, 'eyes77_nin0', 0), 0) or _int(getattr(params, 'num_input_neurons', 0), 0)
        if self.n_in == nin0 + W and nin0 > 0:
            self._e77_w = W; self._e77_layout = lay; self._n_in_base = nin0
            if ctrl_env():                                                 # the bench's control: the brain's 1176 behaviour genes
                b0 = getattr(params, 'eyes77_b0', None) or {}
                try:
                    _sv = os.environ.get('MULTINXON_PIN_SETTLED_VIEW', '')       # the 1176 cells pinned it (settled_view 1): the 1176 self
                    self._settled_view = _int(_sv, 0) if str(_sv).strip() != '' else _int(b0.get('settled_view'), 0)
                    self._v55_border = _int(b0.get('change_border'), 0)
                    self._hud_border = _int(b0.get('hud_border'), 0); self._v55_death = _int(b0.get('death_memory'), 0)
                    if b0.get('action_mask') not in (None, ''):                  # the 1176 mask (1177 adds ACTION7, the undo)
                        self._action_mask = set(int(c) for c in str(b0['action_mask']) if c.isdigit()) or None
                except Exception:
                    pass
                self._e77_silent = True
            elif not getattr(self, '_settled_view', 0):
                self._settled_view = 1                                      # forced: the settled board, frame[-1]
            _tally('e7_on')
        else:
            _tally('e7_missing')

    def play_step(self, frame):
        if getattr(self, '_e77_w', 0):
            try:
                fr = getattr(frame, 'frame', None)
                if not getattr(self, '_v55_trans', 0):
                    self._v55_raw = fr if (isinstance(fr, list) and fr and isinstance(fr[0], list)) else None
                av = getattr(frame, 'available_actions', None) or []
                self._e77_undo_offered = any(int(getattr(x, 'value', x) or 0) == 7 for x in av)
                if self._e77_died and int(getattr(self, '_play_steps', 0)) - int(self._e77_died_t) > 8:
                    self._e77_died = 0
            except Exception:
                pass
        return _ps0(self, frame)

    def reset(self, frame):
        self._e77_prev = None
        return _rs0(self, frame)

    def soft_reset(self, frame):
        self._e77_prev = None
        return _srs0(self, frame)

    ArcBrain.__init__ = __init__; ArcBrain.play_step = play_step; ArcBrain.reset = reset; ArcBrain.soft_reset = soft_reset
    ArcBrain._eyes77_channels = eyes_channels
    _STATE['brain'] = True
    return True
