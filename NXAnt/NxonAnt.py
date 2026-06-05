# Neuraxon Ant Colony 1.02 internal version 09
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonAnt.py — NxonAnt 1.02 ant runtime (chc6 multi-sphere), CONSENSUS EDITION.

The ant runtime is UNCHANGED: a pure function that builds the chc6 brain from a
genome, runs the fixed 1-second protocol, and returns RAW metrics. In the
consensus edition the ant is the "~1-second experiment" every node re-runs to
re-verify a submission. Scoring is NOT done here (see NxonScore.py, the public
integer score every node recomputes) and there is NO secret seed and NO
commit-reveal — the trace digest below is retained but unused by the consensus
node, which re-executes the sim directly. Determinism of the sim across hardware
is the one production requirement (see NxonScore.py).

================================================================================
WHAT AN ANT DOES (1-second task)
================================================================================
An ant is handed a CANDIDATE ARCHITECTURE GENOME — the same kind of object the
NAS searches over (the override sections of nas_best.json: neural,
operating_ranges, genetic_lottery, plus a few biology levers). Its job is to
*evaluate* that architecture by:

  1. Instantiating a COMPRESSED Neuraxon brain from the genome, using the
     paper's chc6 SIX-FUNCTIONAL-SPHERE topology (Multi-Neuraxon 2.0):
        sensory -> {visual, auditory, intero} -> {assoc_fluid <-> assoc_cryst}
                                                              -> motor -> (feedback)
     with cross-sphere coupling kappa, crystallised capacity lambda_c, and
     free-energy beta scaling the communication-through-coherence (CTC) gates —
     exactly the dominant architectural levers from the paper. Real V2.0 neuron
     mechanisms (trinary +1/0/-1, leaky membrane, MSTH-style adaptation,
     autoreceptors, refractory + AHP, STDP) at a size that fits 1 CPU-second.

  2. Driving it with a fixed sensorimotor protocol and measuring the
     paper-fidelity research metrics (a compressed subset of M1-M10):
        M1  trinary excitatory fraction
        M2  inter-sphere CTC gate (assoc_fluid <-> assoc_cryst — the link the
            paper severs in its kappa=0 lesion)
        M5  branching ratio (criticality)
        M6  spontaneous-vs-driven fraction
        M7  self-sustained activity (silence probe)
        M9  compositional transfer ratio
        sensory_motor_corr, input_saturation_fraction

  3. Returning the RAW measured metric vector. THE ANT DOES NOT KNOW THE TARGET
     BANDS. It just reports what the architecture produced; the Overseer holds
     the hidden healthy-band targets and computes the score.

================================================================================
WHY THIS CAN'T BE FAKED
================================================================================
- The ant returns RAW measured metrics + the genome + the eval_seed it was
  issued. It does not return a "score" — scoring is the Overseer's job.
- The metrics are a DETERMINISTIC function of (genome, eval_seed). The Overseer
  re-runs the same genome with the same seed and checks the metrics reproduce.
  A lying ant that fabricates metrics is caught because the re-run won't match.
- The ant doesn't choose its eval_seed (the Overseer issues it) and can't
  predict a genome's metrics without actually running the proxy — which is the
  honest work we want.

================================================================================
1-SECOND BUDGET
================================================================================
Hard constraint, non-negotiable. The chc6 proxy is sized so one evaluation
finishes well within a wall-second. Budget-checked: if it can't finish, the ant
returns status="timeout" and the result is discarded.

Pure function: run_ant(packet) -> result. No global state, no Overseer
connection. Runs identically on 1 core or 1 million machines.

================================================================================
NOTE ON "MultiNeuraxon2"
================================================================================
There is no separate MultiNeuraxon2.py. "Multi-Neuraxon 2.0" is the paper's
multi-sphere architecture; in the real codebase it lives in multisphere.py as
NeuraxonMultiSphere + build_chc_multisphere. Here the equivalent is the
ChcBrain class below — the multi-sphere topology is part of the ant's 1-second
evaluation, not a separate import.
"""

import math
import time
import random
import hashlib
import json
from collections import deque


# =============================================================================
# COMPRESSED TRINARY NEURAXON NEURON
# =============================================================================
# Faithful-but-small implementation of the real V2.0 mechanisms that drive the
# research metrics: trinary firing, leaky membrane, refractory + AHP, MSTH-style
# adaptation, autoreceptors. Game-world bookkeeping (energy/health/phase
# clustering) is dropped — we keep what moves M1-M10.

class CompressedNeuraxon:
    __slots__ = ("nid", "is_inhibitory", "mp", "thr_exc", "thr_inh",
                 "tau", "adapt", "adapt_target", "autoreceptor",
                 "state", "refractory_until", "ahp", "trace",
                 "fire_count", "state_streak", "last_state")

    def __init__(self, nid, is_inhibitory, params, rng):
        self.nid = nid
        self.is_inhibitory = is_inhibitory
        self.thr_exc = params["firing_threshold_excitatory"]
        self.thr_inh = params["firing_threshold_inhibitory"]
        self.tau = max(1.0, params["intrinsic_timescale_default"])
        self.adapt = 0.0
        self.adapt_target = 0.0
        self.autoreceptor = 0.0
        self.ahp = params["post_spike_mp_reset"]
        self.mp = rng.uniform(self.thr_inh * 0.35, self.thr_exc * 0.35)
        self.state = 0
        self.refractory_until = -1
        self.trace = 0.0
        self.fire_count = 0
        self.state_streak = 0
        self.last_state = 0


# =============================================================================
# A SPHERE — one NeuraxonNetwork in the multi-sphere brain
# =============================================================================

class Sphere:
    """One sphere: a small recurrent trinary network with input/hidden/output
    neurons, plus a phase oscillator used for the CTC inter-sphere gate.

    relay_out_ids / relay_in_ids are the 'port neurons' (sparse projection,
    Markov et al. 2014) that carry signal across sphere boundaries.
    """

    def __init__(self, name, n_in, n_hid, n_out, params, plast, rng,
                 natural_freq):
        self.name = name
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n = n_in + n_hid + n_out
        self.first_hid = n_in
        self.first_out = n_in + n_hid
        self.rng = rng
        self._tick = 0

        thr_jitter = plast["thr_jitter"]
        ts_jitter = plast["ts_jitter"]
        base = {
            "firing_threshold_excitatory": params["firing_threshold_excitatory"],
            "firing_threshold_inhibitory": params["firing_threshold_inhibitory"],
            "intrinsic_timescale_default": params["intrinsic_timescale_default"],
            "post_spike_mp_reset": params["post_spike_mp_reset"],
        }
        self.neurons = []
        for i in range(self.n):
            is_inh = (rng.random() < 0.2)
            p = dict(base)
            if thr_jitter > 0:
                p["firing_threshold_excitatory"] += rng.uniform(-thr_jitter, thr_jitter)
                p["firing_threshold_inhibitory"] += rng.uniform(-thr_jitter, thr_jitter)
            if ts_jitter > 0:
                p["intrinsic_timescale_default"] += rng.uniform(-ts_jitter, ts_jitter)
            self.neurons.append(CompressedNeuraxon(i, is_inh, p, rng))

        # Intra-sphere connectivity (adjacency lists of [j, w]).
        conn_p = plast["conn_p"]
        afferent = plast["afferent"]
        sm_coupling = plast["sm_coupling"]
        self.out_edges = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                p_ij = conn_p
                if i < self.n_in and j >= self.first_out:
                    p_ij = min(1.0, conn_p * (1.0 + sm_coupling))
                if rng.random() < p_ij:
                    mag = afferent * rng.uniform(0.5, 1.0)
                    w = -mag if self.neurons[i].is_inhibitory else mag
                    self.out_edges[i].append([j, w])

        # Port neurons.
        self.relay_out_ids = list(range(self.first_out, self.n))
        self.relay_in_ids = list(range(0, self.n_in))

        # Plasticity params.
        self.lr = plast["lr"]
        self.plasticity_threshold = plast["plasticity_threshold"]
        self.adapt_tau = plast["adapt_tau"]
        self.adapt_target_exc = plast["adapt_target_exc"]
        self.adapt_target_inh = plast["adapt_target_inh"]
        self.auto_coeff = plast["auto_coeff"]
        self.auto_tau = plast["auto_tau"]
        self.auto_rate = plast["auto_rate"]
        self.refractory_ticks = plast["refractory_ticks"]
        self.resting_decay = plast["resting_decay"]
        self.spontaneous_p = plast["spontaneous_p"]
        self.symmetric_stdp = plast["symmetric_stdp"]
        self.sensory_gain = plast["sensory_gain"]
        self.boost_scale = plast["boost_scale"]

        # Phase oscillator for CTC gate.
        self.phase = rng.random() * 2 * math.pi
        self.natural_frequency = natural_freq
        self._ext = {}
        self._inter = {}

    def inject_external(self, idx, val):
        self._ext[idx] = self._ext.get(idx, 0.0) + val

    def inject_inter(self, idx, val):
        self._inter[idx] = self._inter.get(idx, 0.0) + val

    def mean_activity(self):
        if self.n == 0:
            return 0.0
        return sum(1 for nrn in self.neurons if nrn.state != 0) / self.n

    def output_activity(self):
        if not self.relay_out_ids:
            return 0.0
        return sum(1 for i in self.relay_out_ids
                   if self.neurons[i].state != 0) / len(self.relay_out_ids)

    def step(self, dt_phase):
        n = self.n
        neurons = self.neurons
        t = self._tick

        # Advance phase oscillator (for CTC gate next tick).
        self.phase = (self.phase + self.natural_frequency * dt_phase) % (2 * math.pi)

        # 1) Recurrent synaptic drive from last-tick traces.
        drive = [0.0] * n
        for i in range(n):
            tr = neurons[i].trace
            if tr == 0.0:
                continue
            for edge in self.out_edges[i]:
                drive[edge[0]] += edge[1] * tr

        # 2) External sensory drive (saturating boost) into input neurons.
        for idx, val in self._ext.items():
            if 0 <= idx < self.n_in:
                x = val * self.sensory_gain
                drive[idx] += self.boost_scale * math.tanh(self.boost_scale * x)

        # 3) Inter-sphere drive into relay-input neurons.
        for idx, val in self._inter.items():
            if 0 <= idx < n:
                drive[idx] += val

        # 4) Membrane update + trinary firing.
        states = [0] * n
        for i in range(n):
            nrn = neurons[i]
            if t < nrn.refractory_until:
                nrn.state = 0
                states[i] = 0
                nrn.mp *= (1.0 - self.resting_decay)
                nrn.mp += (drive[i] - nrn.adapt - nrn.autoreceptor) / nrn.tau
                nrn.trace *= 0.6
                continue
            nrn.mp *= (1.0 - self.resting_decay)
            nrn.mp += (drive[i] - nrn.adapt - nrn.autoreceptor) / nrn.tau
            spont = (self.rng.random() < self.spontaneous_p)
            new_state = 0
            if nrn.mp >= nrn.thr_exc or (spont and not nrn.is_inhibitory):
                new_state = 1
            elif nrn.mp <= nrn.thr_inh or (spont and nrn.is_inhibitory):
                new_state = -1
            if new_state != 0:
                nrn.fire_count += 1
                if self.refractory_ticks > 0:
                    nrn.refractory_until = t + self.refractory_ticks
                nrn.mp *= (1.0 - nrn.ahp)
                tgt = (self.adapt_target_exc if new_state == 1 else self.adapt_target_inh)
                nrn.adapt_target = 0.55 * tgt
            nrn.state = new_state
            states[i] = new_state
            if new_state == nrn.last_state:
                nrn.state_streak += 1
            else:
                nrn.state_streak = 0
                nrn.last_state = new_state

        # 5) Adaptation relax, autoreceptor, trace.
        adapt_decay = 1.0 - 1.0 / self.adapt_tau
        auto_decay = 1.0 - 1.0 / self.auto_tau
        for i in range(n):
            nrn = neurons[i]
            nrn.adapt = nrn.adapt * adapt_decay + (1.0 - adapt_decay) * nrn.adapt_target
            fired = 1.0 if states[i] != 0 else 0.0
            nrn.autoreceptor = (nrn.autoreceptor * auto_decay
                                + self.auto_rate * self.auto_coeff * fired)
            if self.symmetric_stdp:
                nrn.trace = nrn.trace * 0.6 + states[i]
            else:
                nrn.trace = nrn.trace * 0.6 + (1.0 if states[i] == 1 else 0.0)

        # 6) STDP plasticity (gated by co-activity threshold).
        if self.lr > 0:
            for i in range(n):
                tri = neurons[i].trace
                if tri == 0.0:
                    continue
                for edge in self.out_edges[i]:
                    trj = neurons[edge[0]].trace
                    co = tri * trj
                    if abs(co) < self.plasticity_threshold * 0.1:
                        continue
                    neww = edge[1] + self.lr * co
                    if edge[1] > 0 and neww < 0:
                        neww = 0.0
                    elif edge[1] < 0 and neww > 0:
                        neww = 0.0
                    if neww > 2.0:
                        neww = 2.0
                    elif neww < -2.0:
                        neww = -2.0
                    edge[1] = neww

        self._ext = {}
        self._inter = {}
        return states


# =============================================================================
# INTER-SPHERE LINK — CTC-gated projection with delay
# =============================================================================

class SphereLink:
    """A directed link between two spheres with communication-through-coherence
    gating (Fries 2015 / paper Eq. 12) and a transmission delay.

    gate = (1 - c) + c * 0.5 * (1 + cos(phase_src - phase_dst))
      where c = coherence_strength (already scaled by free_energy_beta).
    Effective gain = base_gain (already kappa-scaled) * gate.
    """

    def __init__(self, src, dst, gain, coherence, delay, rng):
        self.src = src
        self.dst = dst
        self.gain = gain
        self.coherence = max(0.0, min(1.0, coherence))
        self.delay = max(1, int(delay))
        self.src_ports = list(src.relay_out_ids)
        self.dst_ports = list(dst.relay_in_ids)
        self.W = []
        for _ in self.dst_ports:
            row = [rng.uniform(0.3, 1.0) for _ in self.src_ports]
            self.W.append(row)
        self.buffer = deque([[0.0] * len(self.dst_ports) for _ in range(self.delay)],
                            maxlen=self.delay)

    def gate_value(self):
        if self.coherence <= 0.0:
            return 1.0
        phase_diff = self.src.phase - self.dst.phase
        phase_gate = 0.5 * (1.0 + math.cos(phase_diff))
        return (1.0 - self.coherence) + self.coherence * phase_gate

    def propagate(self):
        gate = self.gate_value()
        gain_gate = self.gain * gate
        src_states = [self.src.neurons[i].state for i in self.src_ports]
        payload = []
        for row in self.W:
            total = sum(w * s for w, s in zip(row, src_states))
            payload.append(gain_gate * total)
        self.buffer.append(payload)
        delivered = self.buffer.popleft()
        for k, dst_idx in enumerate(self.dst_ports):
            if k < len(delivered):
                self.dst.inject_inter(dst_idx, delivered[k])
        return gate


# =============================================================================
# CHC SIX-SPHERE BRAIN (Multi-Neuraxon 2.0, compressed)
# =============================================================================
# Mirrors build_chc_multisphere from the real multisphere.py:
#   sensory -> {visual, auditory, intero} -> {assoc_fluid <-> assoc_cryst} -> motor
#   + motor -> assoc_fluid feedback, assoc_fluid -> sensory thalamic-like top-down.
# Levers from the genome:
#   cross_sphere_coupling kappa   — global multiplier on every inter-sphere gain
#   cryst_capacity        lambda_c — scales assoc_cryst hidden width
#   free_energy_beta      beta_f  — scales link coherence (CTC strength)

class ChcBrain:
    def __init__(self, genome, seed):
        self.genome = genome
        self.rng = random.Random(seed)
        neural = genome["neural"]
        opr = genome["operating_ranges"]
        lottery = genome.get("genetic_lottery", {})

        self.n_in = int(neural["num_input_neurons"])
        self.n_out = int(neural["num_output_neurons"])
        H = max(3, int(neural["num_hidden_neurons_default"]))

        kappa = max(0.0, float(neural.get("cross_sphere_coupling", 1.0)))
        lam_c = max(0.1, float(neural.get("cryst_capacity", 1.0)))
        beta_f = max(0.0, float(neural.get("free_energy_beta", 1.0)))
        self.kappa = kappa
        self.lam_c = lam_c
        self.beta_f = beta_f

        params = {
            "firing_threshold_excitatory": neural["firing_threshold_excitatory"],
            "firing_threshold_inhibitory": neural["firing_threshold_inhibitory"],
            "intrinsic_timescale_default": neural["intrinsic_timescale_default"],
            "post_spike_mp_reset": neural.get("post_spike_mp_reset", 0.3),
        }
        plast = {
            "thr_jitter": float(lottery.get("firing_threshold_jitter", 0.0)),
            "ts_jitter": float(lottery.get("intrinsic_timescale_jitter", 0.0)),
            "conn_p": float(neural["connection_probability"]),
            "afferent": float(neural.get("afferent_synapse_strength", 0.7)),
            "sm_coupling": float(neural.get("sensorimotor_coupling", 0.0)),
            "lr": float(opr["learning_rate"]),
            "plasticity_threshold": float(opr.get("plasticity_threshold", 0.5)),
            "adapt_tau": max(1.0, float(opr.get("adaptation_tau_ticks", 20.0))),
            "adapt_target_exc": float(opr.get("adaptation_target_excitatory_multiplier", 1.5)),
            "adapt_target_inh": float(opr.get("adaptation_target_inhibitory_multiplier", 1.2)),
            "auto_coeff": float(opr.get("autoreceptor_coefficient", 0.15)),
            "auto_tau": max(1.0, float(opr.get("autoreceptor_tau_ticks", 150.0))),
            "auto_rate": float(opr.get("autoreceptor_rate_coeff", 0.35)),
            "refractory_ticks": int(neural.get("refractory_period_ticks", 0)),
            "resting_decay": float(neural.get("resting_potential_decay", 0.2)),
            "spontaneous_p": float(neural.get("spontaneous_firing_rate", 0.01)),
            "symmetric_stdp": bool(neural.get("symmetric_stdp", False)),
            "sensory_gain": float(neural.get("sensory_input_gain", 0.9)),
            "boost_scale": float(opr.get("sensory_boost_scale", 1.0)),
        }

        def _freq():
            return self.rng.uniform(0.15, 0.45)

        def mk(name, n_in, n_hid, n_out):
            return Sphere(name, n_in, max(3, n_hid), n_out, params, plast,
                          self.rng, _freq())

        # Seven spheres (mirror build_chc_multisphere widths).
        self.spheres = {}
        self.spheres["sensory"] = mk("sensory", self.n_in, max(3, H // 2), 6)
        self.spheres["visual"] = mk("visual", 6, max(3, H // 2), 5)
        self.spheres["auditory"] = mk("auditory", 6, max(3, H // 2), 5)
        self.spheres["intero"] = mk("intero", 6, max(3, H // 2), 5)
        self.spheres["assoc_fluid"] = mk("assoc_fluid", 8, max(4, H), 6)
        cryst_h = max(4, int(round(H * lam_c)))
        self.spheres["assoc_cryst"] = mk("assoc_cryst", 8, cryst_h, 6)
        self.spheres["motor"] = mk("motor", 6, max(3, H // 2), self.n_out)

        # Inter-sphere links.
        self.links = []
        self._lateral_links = []

        def link(a, b, gain, coh, delay=1):
            self.links.append(SphereLink(
                self.spheres[a], self.spheres[b],
                gain=gain * kappa, coherence=coh * beta_f,
                delay=delay, rng=self.rng))

        for sid in ("visual", "auditory", "intero"):
            link("sensory", sid, 1.0, 0.25)
        for sid in ("visual", "auditory", "intero"):
            link(sid, "assoc_fluid", 0.9, 0.25)
            link(sid, "assoc_cryst", 0.9, 0.25)
        # lateral assoc_fluid <-> assoc_cryst (the kappa=0 lesion link)
        self._lateral_links.append(len(self.links)); link("assoc_fluid", "assoc_cryst", 1.0, 0.30)
        self._lateral_links.append(len(self.links)); link("assoc_cryst", "assoc_fluid", 1.0, 0.30)
        link("assoc_fluid", "motor", 1.0, 0.25)
        link("assoc_cryst", "motor", 0.9, 0.25)
        link("motor", "assoc_fluid", 0.7, 0.20, delay=2)
        link("assoc_fluid", "sensory", 0.6, 0.35)

        self.tick_count = 0
        self.dt_phase = 1.0
        self._last_lateral_gates = []

    def step(self, input_vector):
        t = self.tick_count
        for sp in self.spheres.values():
            sp._tick = t

        sens = self.spheres["sensory"]
        for k in range(min(self.n_in, sens.n_in)):
            sens.inject_external(k, input_vector[k])

        lateral_gates = []
        for idx, lk in enumerate(self.links):
            gate = lk.propagate()
            if idx in self._lateral_links:
                lateral_gates.append(gate)
        self._last_lateral_gates = lateral_gates

        states = {}
        for name, sp in self.spheres.items():
            states[name] = sp.step(self.dt_phase)

        self.tick_count += 1
        return states

    def excitatory_fraction(self, states):
        exc = inh = 0
        for sl in states.values():
            for s in sl:
                if s == 1:
                    exc += 1
                elif s == -1:
                    inh += 1
        active = exc + inh
        return (exc / active) if active > 0 else None

    def total_active(self, states):
        return sum(1 for sl in states.values() for s in sl if s != 0)

    def lateral_gate_mean(self):
        af = self.spheres["assoc_fluid"].mean_activity()
        ac = self.spheres["assoc_cryst"].mean_activity()
        gate = (sum(self._last_lateral_gates) / len(self._last_lateral_gates)
                if self._last_lateral_gates else 0.0)
        return gate * math.sqrt(max(af, 0.0) * max(ac, 0.0))

    def input_activity(self, states):
        sens = self.spheres["sensory"]
        s = states["sensory"]
        return sum(1 for i in range(sens.n_in) if s[i] != 0) / max(sens.n_in, 1)

    def motor_activity(self, states):
        motor = self.spheres["motor"]
        s = states["motor"]
        return sum(1 for i in range(motor.first_out, motor.n) if s[i] != 0) / max(motor.n_out, 1)

    def input_saturation(self):
        sens = self.spheres["sensory"]
        return sum(1 for i in range(sens.n_in)
                   if sens.neurons[i].state_streak > 5) / max(sens.n_in, 1)


# =============================================================================
# METRIC RECORDER — compressed M1-M10 proxy over the chc6 brain
# =============================================================================

class MetricRecorder:
    def __init__(self, brain):
        self.brain = brain
        self.exc_frac_hist = []
        self.gate_hist = []
        self.active_total_hist = []
        self.driven_active = []
        self.spont_active = []
        self.input_states = []
        self.output_states = []
        self.input_sat = []
        self._driven = True

    def set_phase(self, driven):
        self._driven = driven

    def observe(self, states):
        b = self.brain
        ef = b.excitatory_fraction(states)
        if ef is not None:
            self.exc_frac_hist.append(ef)
        active = b.total_active(states)
        self.active_total_hist.append(active)
        if self._driven:
            self.driven_active.append(active)
        else:
            self.spont_active.append(active)
        self.gate_hist.append(b.lateral_gate_mean())
        self.input_states.append(b.input_activity(states))
        self.output_states.append(b.motor_activity(states))
        self.input_sat.append(b.input_saturation())

    def _branching(self):
        h = self.active_total_hist
        if len(h) < 2:
            return 0.0
        ratios = [h[t + 1] / h[t] for t in range(len(h) - 1) if h[t] > 0]
        return (sum(ratios) / len(ratios)) if ratios else 0.0

    def _pearson(self, xs, ys):
        if len(xs) < 3:
            return 0.0
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        vx = sum((xs[i] - mx) ** 2 for i in range(n))
        vy = sum((ys[i] - my) ** 2 for i in range(n))
        if vx <= 0 or vy <= 0:
            return 0.0
        return cov / math.sqrt(vx * vy)

    def compute(self, self_sustain, transfer_ratio):
        m1 = (sum(self.exc_frac_hist) / len(self.exc_frac_hist)
              if self.exc_frac_hist else 0.0)
        m2 = (sum(self.gate_hist) / len(self.gate_hist)
              if self.gate_hist else 0.0)
        m5 = self._branching()
        dm = (sum(self.driven_active) / len(self.driven_active)
              if self.driven_active else 0.0)
        sm = (sum(self.spont_active) / len(self.spont_active)
              if self.spont_active else 0.0)
        denom = dm + sm
        m6 = (sm / denom) if denom > 0 else 0.0
        smc = abs(self._pearson(self.input_states, self.output_states))
        sat = (sum(self.input_sat) / len(self.input_sat)
               if self.input_sat else 0.0)
        return {
            "M1_excitatory_fraction": m1,
            "M2_mean_gate": m2,
            "M5_branching_ratio": m5,
            "M6_spontaneous_fraction": m6,
            "M7_zero_input_mi_ratio": self_sustain,
            "M9_transfer_ratio": transfer_ratio,
            "sensory_motor_corr": smc,
            "input_saturation_fraction": sat,
        }


# =============================================================================
# STIMULUS PROTOCOL — drives the chc6 brain, fits the 1-second budget
# =============================================================================

def _evaluate_architecture(genome, eval_seed, budget_deadline, want_trace=False):
    """Build the chc6 brain, run the protocol, return measured metrics.

    Returns (metrics_dict, ticks_done, timed_out) when want_trace is False.
    When want_trace is True, returns (metrics_dict, ticks_done, timed_out,
    trace_digest) where trace_digest is a hash chain of per-tick activity.

    The trace digest binds the metrics to ACTUAL EXECUTION on this exact seed:
    reproducing it requires running every tick (you can't shortcut it), so the
    Overseer can cheaply confirm the work was really done by recomputing the
    chain on the same seed and comparing. A precomputed or fabricated result
    won't carry the correct chain for an audit seed it didn't run.
    """
    brain = ChcBrain(genome, eval_seed)
    rec = MetricRecorder(brain)
    rng = random.Random((eval_seed ^ 0x5DEECE66D) & 0xFFFFFFFF)
    n_in = brain.n_in

    WARMUP, DRIVEN, SILENCE, TRANSFER = 30, 50, 20, 20
    ticks_done = 0

    # Rolling trace-digest hash chain (only maintained when want_trace).
    # chain_{t} = H(chain_{t-1} || tick_index || total_active)
    trace_state = hashlib.sha256(("seed:%d" % eval_seed).encode()) if want_trace else None

    def _chain(active):
        # Fold one tick's activity into the rolling hash.
        trace_state.update(b"%d:%d;" % (ticks_done, int(active)))

    def over():
        return time.time() >= budget_deadline

    def _finish_trace():
        return trace_state.hexdigest()[:32] if want_trace else None

    # WARMUP
    rec.set_phase(True)
    for _ in range(WARMUP):
        if over():
            return (None, ticks_done, True, None) if want_trace else (None, ticks_done, True)
        inp = [1.0 if rng.random() < 0.4 else 0.0 for _ in range(n_in)]
        states = brain.step(inp)
        if want_trace:
            _chain(brain.total_active(states))
        ticks_done += 1

    # DRIVEN
    base = [1.0 if (i % 2 == 0) else 0.0 for i in range(n_in)]
    rec.set_phase(True)
    for t in range(DRIVEN):
        if over():
            return (None, ticks_done, True, None) if want_trace else (None, ticks_done, True)
        inp = base if (t // 5) % 2 == 0 else [1.0 - x for x in base]
        states = brain.step(inp)
        rec.observe(states)
        if want_trace:
            _chain(brain.total_active(states))
        ticks_done += 1

    # SILENCE (M7)
    rec.set_phase(False)
    silence_active = []
    for _ in range(SILENCE):
        if over():
            return (None, ticks_done, True, None) if want_trace else (None, ticks_done, True)
        states = brain.step([0.0] * n_in)
        rec.observe(states)
        act = brain.total_active(states)
        silence_active.append(act)
        if want_trace:
            _chain(act)
        ticks_done += 1
    driven_mean = (sum(rec.driven_active) / len(rec.driven_active)
                   if rec.driven_active else 1.0)
    silence_mean = sum(silence_active) / len(silence_active) if silence_active else 0.0
    self_sustain = (silence_mean / driven_mean) if driven_mean > 0 else 0.0

    # TRANSFER (M9)
    pre_out = (sum(rec.output_states[-DRIVEN:]) / DRIVEN
               if len(rec.output_states) >= DRIVEN else 0.0)
    rec.set_phase(True)
    novel_out = []
    for t in range(TRANSFER):
        if over():
            return (None, ticks_done, True, None) if want_trace else (None, ticks_done, True)
        inp = [0.0] * n_in
        for i in range(n_in):
            if i < n_in // 2:
                inp[i] = 1.0 if (t % 3 == 0) else 0.0
            else:
                inp[i] = 1.0 if (t % 3 == 1) else 0.0
        states = brain.step(inp)
        novel_out.append(brain.motor_activity(states))
        if want_trace:
            _chain(brain.total_active(states))
        ticks_done += 1
    novel_mean = sum(novel_out) / len(novel_out) if novel_out else 0.0
    transfer_ratio = (novel_mean / pre_out) if pre_out > 0 else (1.0 if novel_mean > 0 else 0.0)

    metrics = rec.compute(self_sustain, transfer_ratio)
    if want_trace:
        return metrics, ticks_done, False, _finish_trace()
    return metrics, ticks_done, False


# =============================================================================
# GENOME HASHING — lineage + tamper detection
# =============================================================================

def _canonical_genome_str(genome):
    parts = []
    for section in ("neural", "operating_ranges", "genetic_lottery", "biology"):
        sec = genome.get(section, {})
        for k in sorted(sec.keys()):
            if k.startswith("_"):
                continue
            v = sec[k]
            if isinstance(v, float):
                parts.append("{}.{}={:.10g}".format(section, k, v))
            elif isinstance(v, (list, tuple)):
                inner = ",".join("{:.10g}".format(x) if isinstance(x, float) else str(x)
                                 for x in v)
                parts.append("{}.{}=[{}]".format(section, k, inner))
            else:
                parts.append("{}.{}={}".format(section, k, v))
    return "|".join(parts)


def hash_genome(genome):
    return hashlib.sha256(_canonical_genome_str(genome).encode("utf-8")).hexdigest()[:16]


# =============================================================================
# ANT ENTRY POINT
# =============================================================================

def run_ant(packet):
    """Run one second of work: evaluate a candidate architecture on chc6.

    Returns: status, genome, genome_hash, metrics (RAW), parent_hash,
    eval_seed, ticks_done, wall_time, worker_id.

    The ant does NOT score the metrics and does not know the target bands.
    """
    t0 = time.time()
    budget = packet.get("budget") or {}
    max_wall = float(budget.get("max_wall_seconds", 1.0))
    deadline = t0 + max_wall
    try:
        genome = packet["genome"]
        eval_seed = int(packet.get("eval_seed", 12345))
        # Always produce the trace digest in 1.02 — it binds the metrics to
        # actual per-tick execution on this seed (proof-of-work for the audit).
        metrics, ticks_done, timed_out, trace = _evaluate_architecture(
            genome, eval_seed, deadline, want_trace=True)
        if timed_out:
            return {"status": "timeout", "parent_hash": packet.get("parent_hash"),
                    "ticks_done": ticks_done, "wall_time": time.time() - t0,
                    "worker_id": packet.get("worker_id"),
                    "_local_id": packet.get("_local_id")}
        return {
            "status": "ok",
            "genome": genome,
            "genome_hash": hash_genome(genome),
            "metrics": metrics,
            "trace_digest": trace,
            "parent_hash": packet.get("parent_hash"),
            "eval_seed": eval_seed,
            "ticks_done": ticks_done,
            "wall_time": time.time() - t0,
            "worker_id": packet.get("worker_id"),
            "_local_id": packet.get("_local_id"),
        }
    except Exception as e:
        import traceback
        return {"status": "error", "error": "{}: {}".format(type(e).__name__, e),
                "traceback": traceback.format_exc(),
                "parent_hash": packet.get("parent_hash"),
                "worker_id": packet.get("worker_id"), "wall_time": time.time() - t0,
                "_local_id": packet.get("_local_id")}


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/nas_best.json"
    with open(path) as f:
        arch = json.load(f)
    packet = {"genome": arch, "parent_hash": None, "eval_seed": 42,
              "worker_id": "selftest", "budget": {"max_wall_seconds": 1.0}}
    t0 = time.time()
    result = run_ant(packet)
    print("=" * 70)
    print("NxonAnt 1.01 — chc6 ant self-test on", path)
    print("=" * 70)
    print("status:      {}".format(result["status"]))
    print("wall_time:   {:.3f}s  (budget 1.0s)".format(result.get("wall_time", 0)))
    print("ticks_done:  {}".format(result.get("ticks_done", 0)))
    print("genome_hash: {}".format(result.get("genome_hash", "?")))
    if result["status"] == "ok":
        b = ChcBrain(arch, 42)
        print("\nchc6 brain: {} spheres, {} inter-sphere links".format(
            len(b.spheres), len(b.links)))
        print("  kappa (cross_sphere_coupling): {:.3f}".format(b.kappa))
        print("  lambda_c (cryst_capacity):     {:.3f}".format(b.lam_c))
        print("  beta_f (free_energy_beta):     {:.3f}".format(b.beta_f))
        total_neurons = sum(sp.n for sp in b.spheres.values())
        print("  total neurons: {}".format(total_neurons))
        for name, sp in b.spheres.items():
            print("    {:<14} n={:<3} (in={} hid={} out={})".format(
                name, sp.n, sp.n_in, sp.n_hid, sp.n_out))
        print("\nMeasured metrics (RAW — ant does not score these):")
        for k, v in result["metrics"].items():
            print("  {:<32} {:+.4f}".format(k, v))
    else:
        print("error:", result.get("error"))
        print(result.get("traceback", ""))
