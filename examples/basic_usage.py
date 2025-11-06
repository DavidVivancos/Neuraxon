"""
Basic usage example for Neuraxon neural network
"""

from neuraxon import NeuraxonNetwork, NetworkParameters, save_network, load_network


def main():
    print("=" * 70)
    print("NEURAXON - Basic Usage Example")
    print("=" * 70)
    
    # 1. Create network
    print("\n1. Creating network...")
    params = NetworkParameters(
        num_input_neurons=5,
        num_hidden_neurons=20,
        num_output_neurons=5,
        connection_probability=0.05
    )
    network = NeuraxonNetwork(params)
    print(f"   Created network: {len(network.input_neurons)} input, "
          f"{len(network.hidden_neurons)} hidden, {len(network.output_neurons)} output")
    
    # 2. Set input pattern
    print("\n2. Setting input pattern...")
    input_pattern = [1, -1, 0, 1, -1]
    network.set_input_states(input_pattern)
    print(f"   Input: {input_pattern}")
    
    # 3. Run simulation
    print("\n3. Running simulation...")
    for step in range(100):
        network.simulate_step()
        
        if step % 20 == 0:
            outputs = network.get_output_states()
            print(f"   Step {step:3d}: Outputs = {outputs}")
    
    # 4. Test neuromodulation
    print("\n4. Testing neuromodulation...")
    network.modulate('dopamine', 0.8)
    print("   Dopamine increased to 0.8 (enhanced learning)")
    
    for step in range(20):
        network.simulate_step()
    
    outputs = network.get_output_states()
    print(f"   After modulation: Outputs = {outputs}")
    
    # 5. Save network
    print("\n5. Saving network...")
    save_network(network, "example_network.json")
    
    # 6. Load network
    print("\n6. Loading network...")
    loaded_network = load_network("example_network.json")
    print(f"   Network loaded: {loaded_network.step_count} steps recorded")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()