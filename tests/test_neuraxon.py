"""
Basic tests for Neuraxon neural network
"""

import pytest
from neuraxon import (
    Neuraxon, Synapse, NeuraxonNetwork, NetworkParameters,
    NeuronType, TrinaryState
)


def test_network_creation():
    """Test basic network creation"""
    params = NetworkParameters(
        num_input_neurons=3,
        num_hidden_neurons=5,
        num_output_neurons=2
    )
    network = NeuraxonNetwork(params)
    
    assert len(network.input_neurons) == 3
    assert len(network.hidden_neurons) == 5
    assert len(network.output_neurons) == 2
    assert len(network.all_neurons) == 10


def test_trinary_states():
    """Test trinary state functionality"""
    params = NetworkParameters()
    neuron = Neuraxon(0, NeuronType.HIDDEN, params)
    
    # Test state setting
    neuron.set_state(1)
    assert neuron.trinary_state == 1
    
    neuron.set_state(-1)
    assert neuron.trinary_state == -1
    
    neuron.set_state(0)
    assert neuron.trinary_state == 0


def test_synapse_computation():
    """Test synaptic input computation"""
    params = NetworkParameters()
    synapse = Synapse(0, 1, params)
    
    # Test with different presynaptic states
    input_excitatory = synapse.compute_input(1)
    input_neutral = synapse.compute_input(0)
    input_inhibitory = synapse.compute_input(-1)
    
    assert input_neutral == 0.0
    assert isinstance(input_excitatory, float)
    assert isinstance(input_inhibitory, float)


def test_network_simulation():
    """Test network simulation step"""
    params = NetworkParameters(
        num_input_neurons=2,
        num_hidden_neurons=3,
        num_output_neurons=1
    )
    network = NeuraxonNetwork(params)
    
    initial_step = network.step_count
    network.simulate_step()
    
    assert network.step_count == initial_step + 1
    assert network.time > 0


def test_input_output():
    """Test setting inputs and getting outputs"""
    params = NetworkParameters(
        num_input_neurons=3,
        num_hidden_neurons=5,
        num_output_neurons=2
    )
    network = NeuraxonNetwork(params)
    
    # Set input states
    network.set_input_states([1, -1, 0])
    
    # Verify input states were set
    assert network.input_neurons[0].trinary_state == 1
    assert network.input_neurons[1].trinary_state == -1
    assert network.input_neurons[2].trinary_state == 0
    
    # Get output states
    outputs = network.get_output_states()
    assert len(outputs) <= 2  # May be fewer if neurons died
    assert all(o in [-1, 0, 1] for o in outputs)


def test_neuromodulation():
    """Test neuromodulator adjustment"""
    params = NetworkParameters()
    network = NeuraxonNetwork(params)
    
    # Test dopamine modulation
    network.modulate('dopamine', 0.7)
    assert network.neuromodulators['dopamine'] == 0.7
    
    # Test clamping
    network.modulate('serotonin', 1.5)
    assert network.neuromodulators['serotonin'] == 1.0  # Clamped to max
    
    network.modulate('acetylcholine', -0.5)
    assert network.neuromodulators['acetylcholine'] == 0.0  # Clamped to min


def test_neuron_health():
    """Test neuron health mechanics"""
    params = NetworkParameters()
    neuron = Neuraxon(0, NeuronType.HIDDEN, params)
    
    assert neuron.health == 1.0
    assert neuron.is_active == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])