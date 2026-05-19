# Neuraxon Game of Life v.4.79 multisphere (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 171
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
"""
Multi-Sphere Architecture for the Neuraxon Game of Life
-------------------------------------------------------
Wraps multiple NeuraxonNetwork instances (Spheres) into a modular,
graph-structured brain. Spheres can be independently trained, saved,
loaded, and swapped — enabling "brain surgery" on NxErs.

Default topology: Sensory → Association → Motor (3-sphere hierarchy)
Supports arbitrary directed-graph topologies per Paper §7.

Design Principles:
1. Modularity with port neurons
2. Non-linear, non-sequential connectivity
3. Frequency-gated transmission
4. Hierarchical layering
5. Volume-transmission neuromodulation
"""

import copy
import math
import json
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

from config import NetworkParameters


# =============================================================================
# SPHERE INTERFACE — Port neuron designations
# =============================================================================

@dataclass
class SphereInterface:
    """Defines which neurons of a sphere are accessible across sphere boundaries.
    
    BIOINSPIRED: Only a small fraction of cortical neurons send long-range axons
    to other areas (Markov et al., 2014). Port neurons model this sparse projection.
    
    - sensory_input_ids: receive external (environmental) signals
    - relay_input_ids: receive inter-sphere projections 
    - relay_output_ids: project to other spheres
    - readout_output_ids: used for final behavior readout
    """
    sensory_input_ids: List[int] = field(default_factory=list)
    relay_input_ids: List[int] = field(default_factory=list)
    relay_output_ids: List[int] = field(default_factory=list)
    readout_output_ids: List[int] = field(default_factory=list)

    def all_input_ids(self) -> List[int]:
        return list(set(self.sensory_input_ids + self.relay_input_ids))

    def all_output_ids(self) -> List[int]:
        return list(set(self.relay_output_ids + self.readout_output_ids))

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# SPHERE LAYER — Hierarchical grouping
# =============================================================================

@dataclass
class SphereLayer:
    """Optional hierarchical grouping of spheres (Paper §7 Design Principle 4).
    
    BIOINSPIRED: Cortical areas are organized into processing hierarchies:
    primary sensory → secondary sensory → association → executive/motor.
    """
    layer_id: str = "L0"
    depth: int = 0
    description: str = ""
    sphere_ids: List[str] = field(default_factory=list)

    def add_sphere(self, sphere_id: str):
        if sphere_id not in self.sphere_ids:
            self.sphere_ids.append(sphere_id)

    def remove_sphere(self, sphere_id: str):
        self.sphere_ids = [s for s in self.sphere_ids if s != sphere_id]

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# SPHERE LINK PARAMETERS
# =============================================================================

@dataclass
class SphereLinkParameters:
    """Parameters for inter-sphere projection bundles (Paper §7 Eq. 11-13).
    
    BIOINSPIRED defaults:
    - Positive weights (long-range cortical projections are mostly excitatory)
    - Finite conduction delay
    - Phase-coherence gating (communication-through-coherence)
    - Local Hebbian plasticity for the inter-sphere projection matrix
    """
    gain: float = 1.0
    delay_steps: int = 1
    transmission_threshold: float = 0.25
    coherence_strength: float = 0.2
    coherence_band: str = "theta"
    topology: str = "dense"           # dense | sparse | topographic | one_to_one
    sparse_prob: float = 1.0
    allow_negative_weights: bool = False
    plasticity_rate: float = 0.0
    weight_decay: float = 0.0
    weight_clip: float = 1.5
    normalize_rows: bool = True
    bias: float = 0.0
    kind: str = "feedforward"         # feedforward | feedback | lateral | thalamic_like | context
    # Structural plasticity
    integrity: float = 1.0
    integrity_decay_rate: float = 0.0001
    sprouting_probability: float = 0.001

    def __post_init__(self):
        self.gain = float(self.gain)
        self.delay_steps = max(0, int(self.delay_steps))
        self.transmission_threshold = max(0.0, float(self.transmission_threshold))
        self.coherence_strength = max(0.0, min(1.0, float(self.coherence_strength)))
        self.sparse_prob = max(0.0, min(1.0, float(self.sparse_prob)))
        self.plasticity_rate = max(0.0, float(self.plasticity_rate))
        self.weight_decay = max(0.0, float(self.weight_decay))
        self.weight_clip = max(0.1, float(self.weight_clip))
        self.bias = float(self.bias)
        self.topology = str(self.topology)
        self.kind = str(self.kind)
        self.coherence_band = str(self.coherence_band)
        # Bio-inspired coherence band defaults by link class
        if self.kind == "feedforward" and self.coherence_band == "theta":
            self.coherence_band = "gamma"
        elif self.kind == "feedback" and self.coherence_band == "theta":
            self.coherence_band = "alpha"
        elif self.kind == "thalamic_like" and self.coherence_band == "gamma":
            self.coherence_band = "theta"


# =============================================================================
# CONTINUOUS → TRINARY HELPER
# =============================================================================

def _continuous_to_trinary(value: float, threshold: float = 0.25) -> int:
    v = float(value)
    if v > threshold:
        return 1
    if v < -threshold:
        return -1
    return 0


# =============================================================================
# NEURAXON SPHERE — Wraps a single NeuraxonNetwork
# =============================================================================

class NeuraxonSphere:
    """Wraps an unchanged NeuraxonNetwork as one trainable sphere/module.
    
    BIOINSPIRED: Each sphere is analogous to one cortical area (V1, A1, M1, PFC, etc.).
    Internal dynamics are fully preserved — multi-sphere only adds graph-level composition.
    
    Spheres are the modular units that can be:
    - Trained independently (modality-specific pre-training)
    - Saved/loaded individually (brain surgery / transplant)
    - Swapped between NxErs (sphere exchange)
    """

    def __init__(self, sphere_id: str, network=None,
                 interface: Optional[SphereInterface] = None,
                 label: str = "", layer_id: str = "L0",
                 modality_tags: Optional[List[str]] = None,
                 description: str = ""):
        self.sphere_id = str(sphere_id)
        self.network = network  # NeuraxonNetwork instance
        self.label = label or self.sphere_id
        self.layer_id = layer_id
        self.modality_tags = list(modality_tags or [])
        self.description = description
        self.interface = interface or self._default_interface()
        self._validate_interface()
        # Track last inputs for diagnostics
        self.last_user_inputs: Dict[int, float] = {}
        self.last_link_inputs: Dict[int, float] = {}
        self.last_combined_inputs: Dict[int, int] = {}

    def _default_interface(self) -> SphereInterface:
        input_ids = [n.id for n in self.network.input_neurons]
        output_ids = [n.id for n in self.network.output_neurons]
        return SphereInterface(
            sensory_input_ids=list(input_ids),
            relay_input_ids=list(input_ids),
            relay_output_ids=list(output_ids),
            readout_output_ids=list(output_ids),
        )

    def _validate_interface(self):
        input_ids = {n.id for n in self.network.input_neurons}
        output_ids = {n.id for n in self.network.output_neurons}
        for nid in self.interface.sensory_input_ids + self.interface.relay_input_ids:
            if nid not in input_ids:
                # Auto-fix: remove invalid IDs
                pass
        for nid in self.interface.relay_output_ids + self.interface.readout_output_ids:
            if nid not in output_ids:
                pass

    def relay_outputs(self) -> Dict[int, int]:
        # v4.52 PERF (#15a): reuse the network's pre-built _neuron_map.
        neuron_map = getattr(self.network, '_neuron_map', None) or {n.id: n for n in self.network.all_neurons}
        return {nid: neuron_map[nid].trinary_state 
                for nid in self.interface.relay_output_ids if nid in neuron_map}

    def readout_outputs(self) -> Dict[int, int]:
        neuron_map = getattr(self.network, '_neuron_map', None) or {n.id: n for n in self.network.all_neurons}
        return {nid: neuron_map[nid].trinary_state 
                for nid in self.interface.readout_output_ids if nid in neuron_map}

    def relay_inputs(self) -> Dict[int, int]:
        neuron_map = getattr(self.network, '_neuron_map', None) or {n.id: n for n in self.network.all_neurons}
        return {nid: neuron_map[nid].trinary_state 
                for nid in self.interface.relay_input_ids if nid in neuron_map}

    def all_states(self) -> dict:
        data = self.network.get_all_states() if hasattr(self.network, 'get_all_states') else {}
        data['relay_inputs'] = self.relay_inputs()
        data['relay_outputs'] = self.relay_outputs()
        data['readout_outputs'] = self.readout_outputs()
        return data

    def to_dict(self) -> dict:
        return {
            'sphere_id': self.sphere_id,
            'label': self.label,
            'layer_id': self.layer_id,
            'modality_tags': list(self.modality_tags),
            'description': self.description,
            'interface': self.interface.to_dict(),
            'last_user_inputs': {str(k): v for k, v in self.last_user_inputs.items()},
            'last_link_inputs': {str(k): v for k, v in self.last_link_inputs.items()},
            'last_combined_inputs': {str(k): v for k, v in self.last_combined_inputs.items()},
            'network': self.network.to_dict(),
        }


# =============================================================================
# SPHERE LINK — Inter-sphere projection bundle
# =============================================================================

class SphereLink:
    """Weighted, delayed projection bundle between two spheres (Paper §7 Eq. 11).
    
    BIOINSPIRED: Models long-range cortico-cortical projections with:
    - Finite conduction delay (axonal propagation time)
    - Frequency-gated transmission (communication-through-coherence)
    - Hebbian plasticity (projection weight learning)
    - Structural plasticity (pruning and sprouting)
    """

    def __init__(self, link_id: str, source_sphere_id: str, target_sphere_id: str,
                 source_output_ids: List[int], target_input_ids: List[int],
                 params: Optional[SphereLinkParameters] = None,
                 weight_matrix: Optional[List[List[float]]] = None):
        self.link_id = str(link_id)
        self.source_sphere_id = str(source_sphere_id)
        self.target_sphere_id = str(target_sphere_id)
        self.source_output_ids = list(source_output_ids)
        self.target_input_ids = list(target_input_ids)
        self.params = params or SphereLinkParameters()
        self.weight_matrix = self._init_weight_matrix(weight_matrix)
        self.delay_buffer = deque(
            [[0.0] * len(self.target_input_ids) for _ in range(max(1, self.params.delay_steps))],
            maxlen=max(1, self.params.delay_steps + 1),
        )
        self.integrity = self.params.integrity

    def _sample_weight(self) -> float:
        if self.params.allow_negative_weights:
            return random.uniform(-1.0, 1.0)
        return random.uniform(0.2, 1.0)

    def _normalise_rows(self, matrix: List[List[float]]) -> List[List[float]]:
        if not self.params.normalize_rows:
            return matrix
        out = []
        for row in matrix:
            scale = sum(abs(v) for v in row)
            if scale <= 0.0:
                out.append(row[:])
            else:
                out.append([float(v) / scale for v in row])
        return out

    def _init_weight_matrix(self, provided: Optional[List[List[float]]]) -> List[List[float]]:
        n_tgt = len(self.target_input_ids)
        n_src = len(self.source_output_ids)
        if provided and len(provided) == n_tgt and all(len(r) == n_src for r in provided):
            return self._normalise_rows([row[:] for row in provided])
        
        topo = self.params.topology
        matrix = []
        for ti in range(n_tgt):
            row = []
            for si in range(n_src):
                if topo == "one_to_one":
                    row.append(self._sample_weight() if si == ti else 0.0)
                elif topo == "topographic":
                    dist = abs(si / max(n_src - 1, 1) - ti / max(n_tgt - 1, 1))
                    row.append(self._sample_weight() * max(0.0, 1.0 - 2.0 * dist))
                elif topo == "sparse":
                    row.append(self._sample_weight() if random.random() < self.params.sparse_prob else 0.0)
                else:  # dense
                    row.append(self._sample_weight())
            matrix.append(row)
        return self._normalise_rows(matrix)

    def _communication_gate(self, source_net, target_net) -> float:
        """Paper §7 Eq. 12: Frequency-dependent gate g_p(t).
        
        BIOINSPIRED: Communication-through-coherence (Fries, 2015).
        """
        c = self.params.coherence_strength
        if c <= 0.0:
            return 1.0
        band = self.params.coherence_band
        # Try oscillator_bank (Game of Life NeuraxonNetwork) or oscillators (Neuraxon2Multi)
        src_osc = getattr(source_net, 'oscillator_bank', None) or getattr(source_net, 'oscillators', None)
        tgt_osc = getattr(target_net, 'oscillator_bank', None) or getattr(target_net, 'oscillators', None)
        if not src_osc or not tgt_osc:
            return 1.0
        src_bands = getattr(src_osc, 'bands', {})
        tgt_bands = getattr(tgt_osc, 'bands', {})
        src_band = src_bands.get(band)
        tgt_band = tgt_bands.get(band)
        if not src_band or not tgt_band:
            return 1.0
        phase_diff = float(src_band.get('phase', 0.0)) - float(tgt_band.get('phase', 0.0))
        phase_gate = 0.5 * (1.0 + math.cos(phase_diff))
        return (1.0 - c) + c * phase_gate

    def project(self, source_sphere: 'NeuraxonSphere', target_sphere: 'NeuraxonSphere') -> Dict[int, float]:
        """Compute the projection signal from source to target (Paper §7 Eq. 11).

        v4.52 PERF (#15a): reuses the source network's pre-built `_neuron_map`
        instead of rebuilding `{n.id: n for n in all_neurons}` on every call.
        With 2-3 links and 3 spheres per NxEr, this was being rebuilt 6+ times
        per agent per step. Science-neutral: same dict, same semantics.
        """
        src_net = source_sphere.network
        neuron_map = getattr(src_net, '_neuron_map', None)
        if neuron_map is None:
            neuron_map = {n.id: n for n in src_net.all_neurons}
        source_states = [
            neuron_map[nid].trinary_state if nid in neuron_map else 0
            for nid in self.source_output_ids
        ]
        gate = self._communication_gate(src_net, target_sphere.network)
        gain_gate = float(self.params.gain) * gate
        bias = self.params.bias
        payload = []
        for row in self.weight_matrix:
            total = bias + sum(w * s for w, s in zip(row, source_states))
            payload.append(gain_gate * total)

        if self.params.delay_steps <= 0:
            delayed = payload
        else:
            self.delay_buffer.append(payload)
            delayed = self.delay_buffer.popleft()

        return {nid: delayed[i] for i, nid in enumerate(self.target_input_ids) if i < len(delayed)}

    def update_plasticity(self, source_sphere: 'NeuraxonSphere', target_sphere: 'NeuraxonSphere',
                          global_da: float = 0.15):
        """Paper §7 Eq. 13: Three-factor projection plasticity (STDP + DA gating).

        v4.52 PERF (#15a): reuse network._neuron_map for both source and target
        spheres. Also hoist cached params (plasticity_rate, weight_decay, weight_clip)
        to locals — previously each attribute was re-read per matrix cell.
        """
        if self.params.plasticity_rate <= 0.0:
            return
        src_net = source_sphere.network
        tgt_net = target_sphere.network
        src_map = getattr(src_net, '_neuron_map', None) or {n.id: n for n in src_net.all_neurons}
        tgt_map = getattr(tgt_net, '_neuron_map', None) or {n.id: n for n in tgt_net.all_neurons}
        src = [src_map[nid].trinary_state if nid in src_map else 0 for nid in self.source_output_ids]
        tgt = [tgt_map[nid].trinary_state if nid in tgt_map else 0 for nid in self.target_input_ids]

        da_gate = 0.5 + global_da  # Paper Eq. 13: (0.5 + [DA]^global)
        plast = self.params.plasticity_rate
        decay = self.params.weight_decay
        clip = self.params.weight_clip
        wm = self.weight_matrix
        for ti in range(len(wm)):
            row = wm[ti]
            t_state = float(tgt[ti])
            for si in range(len(row)):
                hebb = float(src[si]) * t_state
                v = row[si] + plast * hebb * da_gate
                if decay > 0.0:
                    v *= (1.0 - decay)
                if v > clip: v = clip
                elif v < -clip: v = -clip
                row[si] = v
        self.weight_matrix = self._normalise_rows(wm)

    def update_structural_plasticity(self):
        """Structural plasticity: decay integrity, mark for pruning."""
        # Compute mean absolute weight
        mean_w = 0.0
        count = 0
        for row in self.weight_matrix:
            for w in row:
                mean_w += abs(w)
                count += 1
        mean_w = mean_w / max(count, 1)
        
        if mean_w < 0.01:
            self.integrity -= self.params.integrity_decay_rate
        else:
            self.integrity = min(1.0, self.integrity + 0.0001)

    def to_dict(self) -> dict:
        return {
            'link_id': self.link_id,
            'source_sphere_id': self.source_sphere_id,
            'target_sphere_id': self.target_sphere_id,
            'source_output_ids': list(self.source_output_ids),
            'target_input_ids': list(self.target_input_ids),
            'params': asdict(self.params),
            'weight_matrix': [row[:] for row in self.weight_matrix],
            'delay_buffer': [list(row) for row in self.delay_buffer],
            'integrity': self.integrity,
        }


# =============================================================================
# NEURAXON MULTI-SPHERE — The modular brain graph
# =============================================================================

class NeuraxonMultiSphere:
    """Graph of Neuraxon v2.0 spheres with arbitrary non-linear connectivity.
    
    Paper §7: "Multi-Sphere architecture groups multiple, independently instantiated
    Neuraxon v2.0 networks (termed Spheres) and connects them through biologically-
    grounded inter-Sphere projections."
    
    Each sphere is an unchanged NeuraxonNetwork. Multi-Sphere only adds:
    - Graph-level composition
    - Interface-port routing
    - Delayed bundles with frequency-gated transmission
    - Optional grouping into layers
    - Volume-transmission neuromodulation
    - Convenience methods for modular brain assembly
    """

    ZERO_EPS = 1e-12

    def __init__(self, name: str = "Neuraxon Multi-Sphere"):
        self.name = name
        self.spheres: Dict[str, NeuraxonSphere] = {}
        self.links: Dict[str, SphereLink] = {}
        self.layers: Dict[str, SphereLayer] = {}
        self.time = 0.0
        self.step_count = 0
        # Global neuromodulator field (volume transmission — Paper §7 Principle 5)
        self.global_neuromodulators: Dict[str, float] = {
            'dopamine': 0.15, 'serotonin': 0.12,
            'acetylcholine': 0.12, 'norepinephrine': 0.12
        }

    # --- Layer management ---
    def register_layer(self, layer_id: str, depth: int = 0, description: str = ""):
        if layer_id not in self.layers:
            self.layers[layer_id] = SphereLayer(layer_id=layer_id, depth=depth, description=description)
        else:
            self.layers[layer_id].depth = depth
            if description:
                self.layers[layer_id].description = description

    # --- Sphere management ---
    def add_sphere(self, sphere_id: str, network=None, params=None,
                   interface: Optional[SphereInterface] = None,
                   label: str = "", layer_id: str = "L0", layer_depth: int = 0,
                   modality_tags: Optional[List[str]] = None,
                   description: str = "") -> NeuraxonSphere:
        if sphere_id in self.spheres:
            raise ValueError(f"Sphere '{sphere_id}' already exists")
        self.register_layer(layer_id, depth=layer_depth)
        
        if network is None:
            from .network import NeuraxonNetwork
            net = NeuraxonNetwork(params or NetworkParameters(
                network_name=f"{sphere_id} Sphere"))
        else:
            net = network
            
        sphere = NeuraxonSphere(
            sphere_id=sphere_id, network=net, interface=interface,
            label=label or sphere_id, layer_id=layer_id,
            modality_tags=modality_tags, description=description,
        )
        self.spheres[sphere_id] = sphere
        self.layers[layer_id].add_sphere(sphere_id)
        return sphere

    def remove_sphere(self, sphere_id: str):
        """Remove a sphere and all its links (for brain surgery / transplant)."""
        if sphere_id not in self.spheres:
            return
        # Remove all links involving this sphere
        links_to_remove = [lid for lid, link in self.links.items()
                          if link.source_sphere_id == sphere_id or link.target_sphere_id == sphere_id]
        for lid in links_to_remove:
            del self.links[lid]
        # Remove from layer
        sphere = self.spheres[sphere_id]
        if sphere.layer_id in self.layers:
            self.layers[sphere.layer_id].remove_sphere(sphere_id)
        del self.spheres[sphere_id]

    def replace_sphere(self, sphere_id: str, new_network, new_interface: Optional[SphereInterface] = None):
        """Hot-swap a sphere's network (brain surgery). Preserves links."""
        if sphere_id not in self.spheres:
            raise KeyError(f"Sphere '{sphere_id}' not found")
        old = self.spheres[sphere_id]
        old.network = new_network
        if new_interface:
            old.interface = new_interface
            old._validate_interface()

    # --- Link management ---
    def connect_spheres(self, source_sphere_id: str, target_sphere_id: str,
                        source_output_ids: Optional[List[int]] = None,
                        target_input_ids: Optional[List[int]] = None,
                        params: Optional[SphereLinkParameters] = None,
                        weight_matrix: Optional[List[List[float]]] = None,
                        link_id: Optional[str] = None,
                        bidirectional: bool = False) -> str:
        if source_sphere_id not in self.spheres or target_sphere_id not in self.spheres:
            raise KeyError("Both spheres must be added before linking")
        src = self.spheres[source_sphere_id]
        tgt = self.spheres[target_sphere_id]
        src_ids = list(source_output_ids or src.interface.relay_output_ids)
        tgt_ids = list(target_input_ids or tgt.interface.relay_input_ids)
        link_name = link_id or f"{source_sphere_id}__to__{target_sphere_id}__{len(self.links)}"
        if link_name in self.links:
            raise ValueError(f"Link '{link_name}' already exists")
        self.links[link_name] = SphereLink(
            link_id=link_name,
            source_sphere_id=source_sphere_id,
            target_sphere_id=target_sphere_id,
            source_output_ids=src_ids,
            target_input_ids=tgt_ids,
            params=params,
            weight_matrix=weight_matrix,
        )
        if bidirectional:
            rev_params = copy.deepcopy(params) if params else SphereLinkParameters()
            rev_id = f"{link_name}__rev"
            self.connect_spheres(target_sphere_id, source_sphere_id,
                                 source_output_ids=tgt.interface.relay_output_ids,
                                 target_input_ids=src.interface.relay_input_ids,
                                 params=rev_params, link_id=rev_id, bidirectional=False)
        return link_name

    def connect_layers(self, source_layer_id: str, target_layer_id: str,
                       params: Optional[SphereLinkParameters] = None,
                       bidirectional: bool = False):
        for src_id in self.layers[source_layer_id].sphere_ids:
            for tgt_id in self.layers[target_layer_id].sphere_ids:
                self.connect_spheres(src_id, tgt_id, params=copy.deepcopy(params),
                                     bidirectional=bidirectional)

    def spheres_in_layer(self, layer_id: str) -> List[str]:
        return list(self.layers.get(layer_id, SphereLayer()).sphere_ids)

    # --- Simulation Pipeline (Paper §8) ---
    def _aggregate_link_inputs(self) -> Dict[str, Dict[int, float]]:
        """Step 1: Inter-Sphere propagation with frequency gating."""
        aggregated: Dict[str, Dict[int, float]] = {sid: {} for sid in self.spheres}
        for link in self.links.values():
            if link.integrity <= 0.0:
                continue
            source = self.spheres.get(link.source_sphere_id)
            target = self.spheres.get(link.target_sphere_id)
            if not source or not target:
                continue
            payload = link.project(source, target)
            for nid, value in payload.items():
                aggregated[target.sphere_id][nid] = aggregated[target.sphere_id].get(nid, 0.0) + float(value)
        return aggregated

    def _prepare_external_inputs(self, sphere: NeuraxonSphere,
                                 user_inputs: Dict[int, float],
                                 link_inputs: Dict[int, float]) -> Dict[int, float]:
        """Combine user (environmental) and inter-sphere link inputs."""
        user_inputs = dict(user_inputs or {})
        link_inputs = dict(link_inputs or {})
        prepared: Dict[int, float] = {}
        input_port_ids = set(sphere.interface.all_input_ids())
        all_input_ids = set(n.id for n in sphere.network.input_neurons)
        managed_ids = sorted(input_port_ids | set(link_inputs.keys()) | {nid for nid in user_inputs if nid in all_input_ids})

        combined_trinary: Dict[int, int] = {}
        for nid in managed_ids:
            total = float(user_inputs.get(nid, 0.0)) + float(link_inputs.get(nid, 0.0))
            state = _continuous_to_trinary(total)
            combined_trinary[nid] = state
            prepared[nid] = float(state) if state != 0 else self.ZERO_EPS

        for nid, value in user_inputs.items():
            if nid not in all_input_ids:
                prepared[nid] = float(value)

        sphere.last_user_inputs = {int(k): float(v) for k, v in user_inputs.items()}
        sphere.last_link_inputs = {int(k): float(v) for k, v in link_inputs.items()}
        sphere.last_combined_inputs = combined_trinary
        return prepared

    def _diffuse_global_neuromodulators(self, rate: float = 0.1):
        """Step 2: Volume-transmission neuromodulation (Paper §7 Principle 5).
        
        BIOINSPIRED: Brainstem nuclei (VTA, raphe, LC, basal forebrain) project diffusely
        to all cortical areas. Global neuromodulator concentrations influence each sphere.
        """
        for sphere in self.spheres.values():
            net = sphere.network
            for mod, global_level in self.global_neuromodulators.items():
                current = net.neuromodulators.get(mod, 0.12)
                # Slow diffusion toward global level
                net.neuromodulators[mod] = current + rate * (global_level - current)

    def simulate_step(self, external_inputs_by_sphere: Optional[Dict[str, Dict[int, float]]] = None):
        """Execute one full multi-sphere simulation step (Paper §8 Pipeline).
        
        Steps:
        1. Inter-Sphere propagation (frequency-gated)
        2. Global neuromodulation (volume transmission)
        3. Intra-Sphere simulation (each sphere runs Algorithm 1)
        4. Projection plasticity (three-factor rule)
        5. Structural plasticity (pruning/sprouting)
        """
        external_inputs_by_sphere = external_inputs_by_sphere or {}
        
        # Step 1: Inter-Sphere propagation
        routed_inputs = self._aggregate_link_inputs()

        # Step 2: Global neuromodulation
        self._diffuse_global_neuromodulators()

        # Step 3: Intra-Sphere simulation
        for sphere_id, sphere in self.spheres.items():
            prepared = self._prepare_external_inputs(
                sphere,
                external_inputs_by_sphere.get(sphere_id, {}),
                routed_inputs.get(sphere_id, {}),
            )
            sphere.network.simulate_step(prepared)

        # Step 4: Projection plasticity
        global_da = self.global_neuromodulators.get('dopamine', 0.15)
        for link in self.links.values():
            if link.integrity <= 0.0:
                continue
            src = self.spheres.get(link.source_sphere_id)
            tgt = self.spheres.get(link.target_sphere_id)
            if src and tgt:
                link.update_plasticity(src, tgt, global_da)

        # Step 5: Structural plasticity
        links_to_prune = []
        for lid, link in self.links.items():
            link.update_structural_plasticity()
            if link.integrity <= 0.0:
                links_to_prune.append(lid)
        for lid in links_to_prune:
            del self.links[lid]

        self.step_count += 1
        self.time = max((s.network.time for s in self.spheres.values()), default=0.0)

        # Update global neuromodulators from sphere averages
        if self.spheres:
            for mod in self.global_neuromodulators:
                avg = sum(s.network.neuromodulators.get(mod, 0.12) for s in self.spheres.values()) / len(self.spheres)
                self.global_neuromodulators[mod] = avg

    # --- Training helpers ---
    def train_sphere_independently(self, sphere_id: str, patterns: List[Any],
                                   steps_per_pattern: int = 10, repetitions: int = 1):
        if sphere_id not in self.spheres:
            raise KeyError(f"Unknown sphere '{sphere_id}'")
        sphere = self.spheres[sphere_id]
        port_ids = sphere.interface.sensory_input_ids or [n.id for n in sphere.network.input_neurons]
        for _ in range(int(repetitions)):
            for pattern in patterns:
                if isinstance(pattern, dict):
                    payload = dict(pattern)
                else:
                    values = list(pattern)
                    payload = {port_ids[i]: values[i] for i in range(min(len(values), len(port_ids)))}
                prepared = self._prepare_external_inputs(sphere, payload, {})
                for _ in range(int(steps_per_pattern)):
                    sphere.network.simulate_step(prepared)

    # --- Global modulation ---
    def set_global_modulator(self, neuromodulator: str, level: float,
                             sphere_ids: Optional[List[str]] = None):
        self.global_neuromodulators[neuromodulator] = level
        targets = sphere_ids or list(self.spheres.keys())
        for sid in targets:
            self.spheres[sid].network.neuromodulators[neuromodulator] = level

    # --- Readout ---
    def get_sphere_outputs(self, sphere_id: str, port: str = 'readout') -> Dict[int, int]:
        sphere = self.spheres[sphere_id]
        if port == 'relay':
            return sphere.relay_outputs()
        return sphere.readout_outputs()

    def get_global_state(self) -> Dict[str, Any]:
        return {
            sid: {
                'layer_id': sphere.layer_id,
                'label': sphere.label,
                'modality_tags': list(sphere.modality_tags),
                'states': sphere.all_states(),
                'last_combined_inputs': dict(sphere.last_combined_inputs),
            }
            for sid, sphere in self.spheres.items()
        }

    def get_energy(self) -> float:
        total = 0.0
        for sphere in self.spheres.values():
            if hasattr(sphere.network, 'get_energy_status'):
                status = sphere.network.get_energy_status()
                total += status.get('total', 0.0)
            elif hasattr(sphere.network, 'all_neurons'):
                total += sum(getattr(n, 'energy_level', 0.0) for n in sphere.network.all_neurons)
        return total

    # --- Proxy properties for backward compatibility with single-network NxEr ---
    @property
    def motor_sphere(self) -> Optional[NeuraxonSphere]:
        """Get the motor sphere (behavior output)."""
        return self.spheres.get('motor')

    @property
    def sensory_sphere(self) -> Optional[NeuraxonSphere]:
        """Get the primary sensory sphere."""
        return self.spheres.get('sensory')

    @property
    def association_sphere(self) -> Optional[NeuraxonSphere]:
        """Get the association sphere."""
        return self.spheres.get('association')

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'time': self.time,
            'step_count': self.step_count,
            'total_energy': self.get_energy(),
            'global_neuromodulators': dict(self.global_neuromodulators),
            'layers': {lid: layer.to_dict() for lid, layer in self.layers.items()},
            'spheres': {sid: sphere.to_dict() for sid, sphere in self.spheres.items()},
            'links': {lid: link.to_dict() for lid, link in self.links.items()},
        }


# =============================================================================
# DEFAULT TOPOLOGY BUILDERS
# =============================================================================

def build_default_multisphere(params: NetworkParameters, 
                               topology: str = "sensory_association_motor") -> NeuraxonMultiSphere:
    """Build a default multi-sphere brain for the Game of Life.
    
    The default 3-sphere hierarchy (v4.5: 10 inputs / 7 outputs):
    - Sensory Sphere (Layer 0): Receives environmental inputs (10 channels, incl. Song)
    - Association Sphere (Layer 1): Integrates, learns context
    - Motor Sphere (Layer 2): Produces behavioral outputs (7 channels, incl. Sing)
    
    This mirrors the paper's "canonical sensory-association-motor hierarchy" (§7.4).
    """
    from .network import NeuraxonNetwork
    
    brain = NeuraxonMultiSphere(f"NxEr Brain ({topology})")
    
    # --- Sensory Sphere ---
    sensory_params = NetworkParameters(
        network_name="Sensory Sphere",
        num_input_neurons=params.num_input_neurons,   # v4.5: 10 environmental inputs (incl. Song)
        num_hidden_neurons=max(3, params.num_hidden_neurons // 2),
        num_output_neurons=4,  # Relay outputs to association
    )
    # Copy key parameters from parent config
    for attr in ['dt', 'membrane_time_constant', 'firing_threshold_excitatory',
                 'firing_threshold_inhibitory', 'spontaneous_firing_rate',
                 'dsn_enabled', 'ctsn_enabled', 'chrono_enabled', 'agmp_enabled',
                 'connection_probability', 'small_world_k', 'small_world_rewire_prob']:
        if hasattr(params, attr):
            setattr(sensory_params, attr, getattr(params, attr))
    
    sensory_net = NeuraxonNetwork(sensory_params)
    sensory_in = [n.id for n in sensory_net.input_neurons]
    sensory_out = [n.id for n in sensory_net.output_neurons]
    brain.add_sphere(
        "sensory", network=sensory_net,
        interface=SphereInterface(
            sensory_input_ids=sensory_in,
            relay_input_ids=sensory_in,
            relay_output_ids=sensory_out,
            readout_output_ids=sensory_out,
        ),
        layer_id="sensory", layer_depth=0,
        modality_tags=["environmental", "perception"],
        description="Sensory cortex-like sphere receiving 10-channel environmental input",
    )

    # --- Association Sphere ---
    assoc_params = NetworkParameters(
        network_name="Association Sphere",
        num_input_neurons=6,  # 4 from sensory relay + 2 context
        num_hidden_neurons=max(4, params.num_hidden_neurons),
        num_output_neurons=5,  # Relay outputs to motor + feedback
    )
    for attr in ['dt', 'membrane_time_constant', 'firing_threshold_excitatory',
                 'firing_threshold_inhibitory', 'spontaneous_firing_rate',
                 'dsn_enabled', 'ctsn_enabled', 'chrono_enabled', 'agmp_enabled',
                 'connection_probability', 'small_world_k', 'small_world_rewire_prob']:
        if hasattr(params, attr):
            setattr(assoc_params, attr, getattr(params, attr))
    
    assoc_net = NeuraxonNetwork(assoc_params)
    assoc_in = [n.id for n in assoc_net.input_neurons]
    assoc_out = [n.id for n in assoc_net.output_neurons]
    brain.add_sphere(
        "association", network=assoc_net,
        interface=SphereInterface(
            sensory_input_ids=assoc_in[:2],     # 2 context inputs
            relay_input_ids=assoc_in[2:],        # 4 relay inputs from sensory
            relay_output_ids=assoc_out[:4],       # 4 relay outputs to motor
            readout_output_ids=assoc_out,
        ),
        layer_id="association", layer_depth=1,
        modality_tags=["multimodal", "context", "integration"],
        description="Association cortex-like sphere for context integration",
    )

    # --- Motor Sphere ---
    motor_params = NetworkParameters(
        network_name="Motor Sphere",
        num_input_neurons=6,  # From association relay + direct proprioception
        num_hidden_neurons=max(3, params.num_hidden_neurons // 2),
        num_output_neurons=params.num_output_neurons,  # v4.5: 7 behavioral outputs (incl. Sing)
    )
    for attr in ['dt', 'membrane_time_constant', 'firing_threshold_excitatory',
                 'firing_threshold_inhibitory', 'spontaneous_firing_rate',
                 'dsn_enabled', 'ctsn_enabled', 'chrono_enabled', 'agmp_enabled',
                 'connection_probability', 'small_world_k', 'small_world_rewire_prob']:
        if hasattr(params, attr):
            setattr(motor_params, attr, getattr(params, attr))

    motor_net = NeuraxonNetwork(motor_params)
    motor_in = [n.id for n in motor_net.input_neurons]
    motor_out = [n.id for n in motor_net.output_neurons]
    brain.add_sphere(
        "motor", network=motor_net,
        interface=SphereInterface(
            sensory_input_ids=motor_in[:2],       # Direct proprioception/urgency
            relay_input_ids=motor_in[2:],          # 4 from association relay
            relay_output_ids=motor_out[:3],         # Feedback to association
            readout_output_ids=motor_out,           # All 6 behavioral outputs
        ),
        layer_id="motor", layer_depth=2,
        modality_tags=["action", "behavior"],
        description="Motor cortex-like sphere producing 7-channel behavioral output",
    )

    # --- Inter-Sphere Links ---
    ff_params = SphereLinkParameters(
        kind="feedforward", topology="topographic", delay_steps=1,
        gain=1.0, coherence_strength=0.25, plasticity_rate=0.001,
    )
    fb_params = SphereLinkParameters(
        kind="feedback", topology="sparse", sparse_prob=0.6, delay_steps=2,
        gain=0.8, coherence_strength=0.2, plasticity_rate=0.0005,
    )
    ctx_params = SphereLinkParameters(
        kind="thalamic_like", topology="dense", delay_steps=1,
        gain=0.6, coherence_strength=0.35, plasticity_rate=0.0005,
    )

    # Feedforward: sensory → association
    brain.connect_spheres("sensory", "association",
                          source_output_ids=sensory_out,
                          target_input_ids=assoc_in[2:],  # relay inputs
                          params=copy.deepcopy(ff_params))
    # Feedforward: association → motor
    brain.connect_spheres("association", "motor",
                          source_output_ids=assoc_out[:4],
                          target_input_ids=motor_in[2:],  # relay inputs
                          params=copy.deepcopy(ff_params))
    # Feedback: motor → association
    brain.connect_spheres("motor", "association",
                          source_output_ids=motor_out[:3],
                          target_input_ids=assoc_in[:3] if len(assoc_in) >= 3 else assoc_in,
                          params=copy.deepcopy(fb_params))
    # Context/Thalamic: association → sensory (top-down)
    brain.connect_spheres("association", "sensory",
                          source_output_ids=assoc_out[:3] if len(assoc_out) >= 3 else assoc_out,
                          target_input_ids=sensory_in[:3] if len(sensory_in) >= 3 else sensory_in,
                          params=copy.deepcopy(ctx_params))

    return brain


# =============================================================================
# MULTI-SPHERE SAVE / LOAD
# =============================================================================

def save_multisphere_to_dict(multi: NeuraxonMultiSphere) -> dict:
    """Serialize a NeuraxonMultiSphere to a dict (for embedding in save files)."""
    return multi.to_dict()


def load_multisphere_from_dict(data: dict, rebuild_net_fn=None) -> NeuraxonMultiSphere:
    """Reconstruct a NeuraxonMultiSphere from a dict.
    
    Args:
        data: Serialized multi-sphere dict
        rebuild_net_fn: Callable that takes a network dict and returns a NeuraxonNetwork.
                       If None, uses the default from network module.
    """
    model = NeuraxonMultiSphere(name=data.get('name', 'Neuraxon Multi-Sphere'))
    model.global_neuromodulators = data.get('global_neuromodulators', {
        'dopamine': 0.15, 'serotonin': 0.12,
        'acetylcholine': 0.12, 'norepinephrine': 0.12
    })

    # Restore layers
    for layer_id, layer_data in data.get('layers', {}).items():
        model.register_layer(
            layer_id=layer_id,
            depth=int(layer_data.get('depth', 0)),
            description=layer_data.get('description', ''),
        )

    # Restore spheres
    for sphere_id, sphere_data in data.get('spheres', {}).items():
        interface = SphereInterface(**sphere_data.get('interface', {}))
        if rebuild_net_fn:
            network = rebuild_net_fn(sphere_data['network'])
        else:
            # Fallback: try to import and use the game of life network loader
            try:
                from .network import NeuraxonNetwork
                network = NeuraxonNetwork._from_dict(sphere_data['network'])
            except Exception:
                network = None
        
        if network is None:
            continue
            
        sphere = model.add_sphere(
            sphere_id=sphere_id,
            network=network,
            interface=interface,
            label=sphere_data.get('label', sphere_id),
            layer_id=sphere_data.get('layer_id', 'L0'),
            modality_tags=sphere_data.get('modality_tags', []),
            description=sphere_data.get('description', ''),
        )
        sphere.last_user_inputs = {int(k): float(v) for k, v in sphere_data.get('last_user_inputs', {}).items()}
        sphere.last_link_inputs = {int(k): float(v) for k, v in sphere_data.get('last_link_inputs', {}).items()}
        sphere.last_combined_inputs = {int(k): int(v) for k, v in sphere_data.get('last_combined_inputs', {}).items()}

    # Restore links
    for link_id, link_data in data.get('links', {}).items():
        params = SphereLinkParameters(**{k: v for k, v in link_data.get('params', {}).items()
                                         if k in SphereLinkParameters.__dataclass_fields__})
        link = SphereLink(
            link_id=link_id,
            source_sphere_id=link_data['source_sphere_id'],
            target_sphere_id=link_data['target_sphere_id'],
            source_output_ids=link_data['source_output_ids'],
            target_input_ids=link_data['target_input_ids'],
            params=params,
            weight_matrix=link_data.get('weight_matrix'),
        )
        link.integrity = link_data.get('integrity', 1.0)
        saved_buffer = link_data.get('delay_buffer', [])
        if isinstance(saved_buffer, list) and saved_buffer:
            link.delay_buffer = deque(
                [list(map(float, row)) for row in saved_buffer],
                maxlen=max(1, params.delay_steps + 1),
            )
        model.links[link_id] = link

    model.time = float(data.get('time', 0.0))
    model.step_count = int(data.get('step_count', 0))
    return model


# =============================================================================
# SPHERE-LEVEL SAVE/LOAD (for individual sphere transplant)
# =============================================================================

def save_sphere_to_dict(sphere: NeuraxonSphere) -> dict:
    """Save a single sphere (for transplanting into another brain)."""
    return sphere.to_dict()


def load_sphere_from_dict(data: dict, rebuild_net_fn=None) -> NeuraxonSphere:
    """Load a single sphere from dict."""
    interface = SphereInterface(**data.get('interface', {}))
    if rebuild_net_fn:
        network = rebuild_net_fn(data['network'])
    else:
        network = None
    
    return NeuraxonSphere(
        sphere_id=data.get('sphere_id', 'unknown'),
        network=network,
        interface=interface,
        label=data.get('label', ''),
        layer_id=data.get('layer_id', 'L0'),
        modality_tags=data.get('modality_tags', []),
        description=data.get('description', ''),
    )
