#!/usr/bin/env python3
"""
Demo: Google FlyBrain Integration

This script demonstrates how to:
1. Connect to NeuPrint API (hemibrain dataset)
2. Fetch visual-motor pathway neurons
3. Convert to our ConnectomeData format
4. Build a network for game playing

Requirements:
- NEUPRINT_TOKEN environment variable (get from https://neuprint.janelia.org/)
- Or run without token for public access (rate limited)

Usage:
    export NEUPRINT_TOKEN="your_token_here"
    python demo_flybrain.py
"""

import os
import sys
import logging
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.connectome.loader import ConnectomeLoader
from src.connectome.google_flybrain import (
    FlyBrainConfig, FlyBrainIntegrator, 
    create_flybrain_config_from_env, fetch_hemibrain_visual_motor
)
from src.networks.assembly import NetworkBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_neuprint_connection():
    """Test basic NeuPrint connection."""
    logger.info("=" * 60)
    logger.info("DEMO 1: NeuPrint Connection Test")
    logger.info("=" * 60)
    
    config = create_flybrain_config_from_env()
    integrator = FlyBrainIntegrator(config)
    
    # Get available ROIs
    logger.info("Fetching available brain regions (ROIs)...")
    try:
        roi_info = integrator.neuprint.get_roi_info()
        logger.info(f"Found {len(roi_info)} ROIs")
        for roi in roi_info[:10]:
            logger.info(f"  {roi['name']}: volume={roi.get('volume', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to fetch ROIs: {e}")
        return False
    
    return True


def demo_fetch_visual_neurons():
    """Fetch visual system neurons."""
    logger.info("=" * 60)
    logger.info("DEMO 2: Fetch Visual System Neurons")
    logger.info("=" * 60)
    
    config = create_flybrain_config_from_env()
    integrator = FlyBrainIntegrator(config)
    
    # Fetch LC neurons (lobula columnar - visual projection)
    logger.info("Fetching LC neurons...")
    try:
        neurons = integrator.fetch_cell_type_neurons("LC", max_neurons=100)
        logger.info(f"Found {len(neurons)} LC neurons")
        
        # Show cell types
        cell_types = {}
        for n in neurons:
            ct = n.get('type', 'unknown')
            cell_types[ct] = cell_types.get(ct, 0) + 1
        
        logger.info("Top cell types:")
        for ct, count in sorted(cell_types.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {ct}: {count}")
        
        return neurons
    except Exception as e:
        logger.error(f"Failed to fetch neurons: {e}")
        return []


def demo_fetch_subcircuit():
    """Fetch a subcircuit between cell types."""
    logger.info("=" * 60)
    logger.info("DEMO 3: Fetch Visual-Motor Subcircuit")
    logger.info("=" * 60)
    
    config = create_flybrain_config_from_env()
    integrator = FlyBrainIntegrator(config)
    
    # Define visual-motor pathway cell types
    pre_regions = ["LC", "LPLC", "LPTC", "Tm", "TmY", "mALC", "AVLP"]  # Optic lobes
    post_regions = ["PEN", "EPG", "PEG", "PFN", "PFL", "FR", "FC", "FS", "FB", "hDelta", "vDelta",  # Central complex
                    "KC", "APL", "DAN", "MBON", "PPL", "PAM",  # Mushroom body
                    "LH", "LHPV",  # Lateral horn
                    "DNa", "DNb", "DNc", "DNd", "DNg", "DNp", "DN", "MDN"]  # Descending neurons
    
    logger.info(f"Pre-regions (input): {pre_regions}")
    logger.info(f"Post-regions (output): {post_regions}")
    
    try:
        neurons, synapses = integrator.fetch_subcircuit(
            pre_regions=pre_regions,
            post_regions=post_regions,
            max_neurons_per_region=200
        )
        
        logger.info(f"Total neurons: {len(neurons)}")
        logger.info(f"Total synapses: {len(synapses)}")
        
        # Count by cell type
        type_counts = {}
        for n in neurons:
            ct = n.get('type', 'unknown')
            type_counts[ct] = type_counts.get(ct, 0) + 1
        
        logger.info("Neurons per cell type:")
        for ct, count in sorted(type_counts.items()):
            logger.info(f"  {ct}: {count}")
        
        return neurons, synapses
    except Exception as e:
        logger.error(f"Failed to fetch subcircuit: {e}")
        return [], []


def demo_convert_to_connectome_data(neurons, synapses):
    """Convert NeuPrint data to our format."""
    logger.info("=" * 60)
    logger.info("DEMO 4: Convert to ConnectomeData")
    logger.info("=" * 60)
    
    config = create_flybrain_config_from_env()
    integrator = FlyBrainIntegrator(config)
    
    try:
        connectome_data = integrator.export_to_connectome_data(neurons, synapses)
        
        logger.info(f"ConnectomeData created:")
        logger.info(f"  Neurons: {len(connectome_data.neurons)}")
        logger.info(f"  Synapses: {len(connectome_data.synapses)}")
        logger.info(f"  Regions: {connectome_data.regions}")
        logger.info(f"  Cell types: {len(connectome_data.cell_types)}")
        
        # Show adjacency matrix info
        for region in connectome_data.regions:
            if region != 'unknown':
                W = connectome_data.get_adjacency_matrix(region=region, sparse=True)
                logger.info(f"  {region}: {W.shape[0]} neurons, {W.nnz} synapses")
        
        return connectome_data
    except Exception as e:
        logger.error(f"Failed to convert: {e}")
        return None


def demo_build_network(connectome_data):
    """Build network from connectome data."""
    logger.info("=" * 60)
    logger.info("DEMO 5: Build Network from Connectome")
    logger.info("=" * 60)
    
    if connectome_data is None:
        logger.warning("No connectome data, skipping network build")
        return None
    
    try:
        # Create network config
        from src.networks.assembly import NetworkConfig
        config = NetworkConfig()
        # Create network builder
        builder = NetworkBuilder(config)
        
        # Build populations and synapses
        builder.build_from_connectome(connectome_data)
        
        logger.info(f"Network built:")
        logger.info(f"  Populations: {list(builder.populations.keys())}")
        logger.info(f"  Synapses: {list(builder.synapses.keys())}")
        
        for name, pop in builder.populations.items():
            logger.info(f"  {name}: {pop.n_neurons} neurons")
        
        for name, syn in builder.synapses.items():
            logger.info(f"  {name}: {syn.weights.shape} weights")
        
        return builder
    except Exception as e:
        logger.error(f"Failed to build network: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_network_step(builder):
    """Test network step with synthetic input."""
    logger.info("=" * 60)
    logger.info("DEMO 6: Test Network Step")
    logger.info("=" * 60)
    
    if builder is None:
        logger.warning("No network, skipping step test")
        return
    
    try:
        # Create synthetic optic lobe input (visual stimulus)
        n_optic = builder.populations.get('optic_lobes', None)
        if n_optic is None:
            # Find optic lobe population
            for name in builder.populations:
                if 'optic' in name.lower() or 'me' in name.lower() or 'lo' in name.lower():
                    n_optic = builder.populations[name]
                    break
        
        if n_optic is None:
            logger.warning("No optic lobe population found")
            return
        
        # Create input spike pattern (simulate moving object)
        optic_input = np.zeros(n_optic.n_neurons)
        # Activate a subset of neurons (simulate receptive field)
        active_indices = np.random.choice(n_optic.n_neurons, size=min(50, n_optic.n_neurons), replace=False)
        optic_input[active_indices] = 1.0
        
        logger.info(f"Injecting input into {len(active_indices)} optic lobe neurons")
        
        # Step network
        spikes = builder.step(external_input={'optic_lobes': optic_input})
        
        # Report activity
        total_spikes = 0
        for region, region_spikes in spikes.items():
            n_spikes = region_spikes.sum()
            total_spikes += n_spikes
            if n_spikes > 0:
                logger.info(f"  {region}: {n_spikes:.0f} spikes")
        
        logger.info(f"Total spikes: {total_spikes:.0f}")
        
        # Test multiple steps
        logger.info("Running 10 steps...")
        for step in range(10):
            spikes = builder.step(external_input={'optic_lobes': optic_input})
            total = sum(s.sum() for s in spikes.values())
            if step % 3 == 0:
                logger.info(f"  Step {step}: {total:.0f} total spikes")
        
    except Exception as e:
        logger.error(f"Failed to step network: {e}")
        import traceback
        traceback.print_exc()


def demo_full_pipeline():
    """Run the full pipeline: fetch -> convert -> build -> test."""
    logger.info("=" * 60)
    logger.info("DEMO 7: Full Pipeline (Cached)")
    logger.info("=" * 60)
    
    config = create_flybrain_config_from_env()
    integrator = FlyBrainIntegrator(config)
    
    # Try to load cached connectome first
    cache_name = "hemibrain_visual_motor"
    try:
        connectome_data = integrator.load_cached_connectome(cache_name)
        logger.info(f"Loaded cached connectome: {cache_name}")
    except FileNotFoundError:
        logger.info("No cache found, fetching from NeuPrint...")
        connectome_data = integrator.build_game_ready_connectome(
            max_neurons_per_region=500
        )
        integrator.cache_connectome(connectome_data, cache_name)
        logger.info(f"Cached connectome as: {cache_name}")
    
    # Build network
    from src.networks.assembly import NetworkConfig
    config = NetworkConfig()
    builder = NetworkBuilder(config)
    builder.build_from_connectome(connectome_data)
    
    logger.info(f"Network ready: {sum(p.n_neurons for p in builder.populations.values())} neurons")
    
    # Test with game-like input
    logger.info("Testing with game-like visual input...")
    
    # Simulate a moving ball trajectory
    for frame in range(20):
        # Create moving stimulus
        optic_pop = None
        for name, pop in builder.populations.items():
            if 'me' in name.lower() or 'lo' in name.lower() or 'optic' in name.lower():
                optic_pop = pop
                break
        
        if optic_pop:
            # Moving Gaussian blob
            center = int(optic_pop.n_neurons * (0.3 + 0.4 * np.sin(frame * 0.3)))
            width = max(1, int(optic_pop.n_neurons * 0.05))
            input_spikes = np.zeros(optic_pop.n_neurons)
            for i in range(max(0, center-width), min(optic_pop.n_neurons, center+width)):
                input_spikes[i] = np.exp(-(i-center)**2 / (2*width**2))
            
            spikes = builder.step(external_input={'optic_lobes': input_spikes})
            total = sum(s.sum() for s in spikes.values())
            if frame % 5 == 0:
                logger.info(f"  Frame {frame}: {total:.0f} spikes")


def demo_without_token():
    """Demo using synthetic data when no token available."""
    logger.info("=" * 60)
    logger.info("DEMO 8: Synthetic Fallback (No Token)")
    logger.info("=" * 60)
    
    # Use our existing synthetic connectome
    loader = ConnectomeLoader("config/connectome_test.yaml")
    connectome_data = loader.create_synthetic(max_synapses=100000)
    
    logger.info(f"Synthetic connectome:")
    logger.info(f"  Neurons: {len(connectome_data.neurons)}")
    logger.info(f"  Synapses: {len(connectome_data.synapses)}")
    logger.info(f"  Regions: {connectome_data.regions}")
    
    # Build network with proper config
    from src.networks.assembly import NetworkBuilder, NetworkConfig
    config = NetworkConfig(
        regions=connectome_data.regions,
        use_neuromodulation=True,
        enforce_dale=False
    )
    builder = NetworkBuilder(config)
    builder.build_from_connectome(connectome_data)
    
    logger.info(f"Network built with {sum(p.n_neurons for p in builder.populations.values())} neurons")
    
    # Test
    optic_pop = builder.populations.get('optic_lobes')
    if optic_pop:
        input_spikes = np.zeros(optic_pop.n_neurons_neurons)
        input_spikes[:50] = 1.0
        spikes = builder.step(external_input={'optic_lobes': input_spikes})
        total = sum(s.sum() for s in spikes.values())
        logger.info(f"Test step: {total:.0f} total spikes")


def main():
    """Run all demos."""
    logger.info("Google FlyBrain Integration Demo")
    logger.info("=" * 60)
    
    # Check for token
    has_token = bool(os.getenv('NEUPRINT_TOKEN'))
    if has_token:
        logger.info("NEUPRINT_TOKEN found - will use live API")
    else:
        logger.info("No NEUPRINT_TOKEN - will use synthetic fallback")
        logger.info("Get token from: https://neuprint.janelia.org/")
    
    # Run demos
    if has_token:
        demo_neuprint_connection()
        neurons = demo_fetch_visual_neurons()
        neurons, synapses = demo_fetch_subcircuit()
        connectome_data = demo_convert_to_connectome_data(neurons, synapses)
        builder = demo_build_network(connectome_data)
        demo_network_step(builder)
        demo_full_pipeline()
    else:
        demo_without_token()
    
    logger.info("=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()