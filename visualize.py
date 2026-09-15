#!/usr/bin/env python3
"""
Visualization dashboard for fruit fly brain activity.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from networks.assembly import NetworkBuilder, NetworkConfig
from neurons.lif import LIFParams
from synapses.stdp import STDPParams
from neuromod.modulator import UnifiedNeuromodulation
from connectome.loader import ConnectomeLoader
import yaml


class BrainVisualizer:
    """Real-time visualization of network activity."""
    
    def __init__(self, network: NetworkBuilder, connectome=None):
        self.network = network
        self.connectome = connectome
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = GridSpec(3, 4, figure=self.fig)
        
        # Raster plot axes
        self.ax_raster = self.fig.add_subplot(self.gs[0, :2])
        self.ax_raster.set_title("Spike Raster (All Regions)")
        self.ax_raster.set_xlabel("Time (ms)")
        self.ax_raster.set_ylabel("Neuron Index")
        
        # Voltage traces
        self.ax_voltage = self.fig.add_subplot(self.gs[0, 2:])
        self.ax_voltage.set_title("Membrane Potentials (Sample)")
        self.ax_voltage.set_xlabel("Time (ms)")
        self.ax_voltage.set_ylabel("V (mV)")
        
        # Weight matrices
        self.ax_weights = self.fig.add_subplot(self.gs[1, :2])
        self.ax_weights.set_title("Synaptic Weights")
        
        # Neuromodulation
        self.ax_neuromod = self.fig.add_subplot(self.gs[1, 2:])
        self.ax_neuromod.set_title("Neuromodulator Concentrations")
        
        # Firing rates
        self.ax_rates = self.fig.add_subplot(self.gs[2, :2])
        self.ax_rates.set_title("Population Firing Rates")
        self.ax_rates.set_xlabel("Time (ms)")
        self.ax_rates.set_ylabel("Rate (Hz)")
        
        # Connectivity graph
        self.ax_connect = self.fig.add_subplot(self.gs[2, 2:])
        self.ax_connect.set_title("Region Connectivity")
        
        # Data buffers
        self.max_history = 1000
        self.time_buffer = []
        self.spike_buffer = {r: [] for r in network.config.regions}
        self.voltage_buffer = {r: [] for r in network.config.regions}
        self.rate_buffer = {r: [] for r in network.config.regions}
        self.neuromod_buffer = {"dopamine": [], "octopamine": [], "serotonin": []}
        
        # Sample neurons for voltage traces (local indices within region)
        self.sample_neurons = {}
        for region in network.config.regions:
            n = len(network.get_neuron_ids(region))
            if n > 0:
                self.sample_neurons[region] = list(range(min(5, n)))
        
        self.step_count = 0
    
    def update(self, frame):
        """Update visualization with one network step."""
        # Run network step with random input
        external_input = {}
        for region in self.network.config.regions:
            n = len(self.network.get_neuron_ids(region))
            if n > 0:
                # Random Poisson input
                rates = np.random.exponential(5.0, n)
                spikes = np.random.random(n) < rates * self.network.config.dt / 1000.0
                external_input[region] = spikes.astype(float) * 50.0
        
        spikes = self.network.step(external_input=external_input)
        state = self.network.get_state()
        
        # Update buffers
        t = self.step_count * self.network.config.dt
        self.time_buffer.append(t)
        if len(self.time_buffer) > self.max_history:
            self.time_buffer.pop(0)
        
        for region in self.network.config.regions:
            # Spikes
            region_spikes = spikes.get(region, np.array([]))
            spike_indices = np.where(region_spikes)[0]
            for idx in spike_indices:
                self.spike_buffer[region].append((t, idx))
            
            # Trim spike buffer
            cutoff = t - self.max_history * self.network.config.dt
            self.spike_buffer[region] = [(tt, ii) for tt, ii in self.spike_buffer[region] if tt > cutoff]
            
            # Voltages
            if region in state["neurons"]:
                v = state["neurons"][region]["v"]
                if region in self.sample_neurons:
                    sample_v = v[self.sample_neurons[region]]
                    self.voltage_buffer[region].append(sample_v)
                    if len(self.voltage_buffer[region]) > self.max_history:
                        self.voltage_buffer[region].pop(0)
            
            # Firing rates (sliding window)
            recent_spikes = [ii for tt, ii in self.spike_buffer[region] if tt > t - 100]
            rate = len(recent_spikes) / max(1, len(self.network.get_neuron_ids(region))) * 1000 / 100
            self.rate_buffer[region].append(rate)
            if len(self.rate_buffer[region]) > self.max_history:
                self.rate_buffer[region].pop(0)
        
        # Neuromodulation
        if self.network.neuromod:
            for mod in ["dopamine", "octopamine", "serotonin"]:
                concs = []
                for region in self.network.config.regions:
                    concs.append(getattr(self.network.neuromod, mod).get_region_concentration(region))
                self.neuromod_buffer[mod].append(np.mean(concs))
                if len(self.neuromod_buffer[mod]) > self.max_history:
                    self.neuromod_buffer[mod].pop(0)
        
        self.step_count += 1
        
        # Redraw
        self._redraw()
        return []
    
    def _redraw(self):
        """Redraw all plots."""
        # Raster
        self.ax_raster.clear()
        self.ax_raster.set_title("Spike Raster")
        self.ax_raster.set_xlabel("Time (ms)")
        self.ax_raster.set_ylabel("Neuron Index")
        
        y_offset = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.network.config.regions)))
        for i, region in enumerate(self.network.config.regions):
            if self.spike_buffer[region]:
                times, indices = zip(*self.spike_buffer[region])
                self.ax_raster.scatter(times, [y_offset + idx for idx in indices], 
                                     s=1, c=[colors[i]], alpha=0.6, label=region)
            y_offset += len(self.network.get_neuron_ids(region))
        if any(self.spike_buffer[r] for r in self.network.config.regions):
            self.ax_raster.legend(loc='upper right', fontsize=8)
        
        # Voltages
        self.ax_voltage.clear()
        self.ax_voltage.set_title("Membrane Potentials")
        self.ax_voltage.set_xlabel("Time (ms)")
        self.ax_voltage.set_ylabel("V (mV)")
        for region in self.network.config.regions:
            if self.voltage_buffer[region]:
                voltages = np.array(self.voltage_buffer[region]).T
                times = self.time_buffer[-len(voltages[0]):]
                for j, v in enumerate(voltages):
                    self.ax_voltage.plot(times, v, alpha=0.7, label=f"{region}_{j}")
        self.ax_voltage.legend(loc='upper right', fontsize=6)
        
        # Weights
        self.ax_weights.clear()
        self.ax_weights.set_title("Synaptic Weight Matrices")
        if self.network.synapses:
            # Show first synapse group
            key, syn = list(self.network.synapses.items())[0]
            W = syn.get_dense_weights()
            im = self.ax_weights.imshow(W, aspect='auto', cmap='viridis', vmin=0, vmax=1)
            self.ax_weights.set_xlabel("Pre-synaptic")
            self.ax_weights.set_ylabel("Post-synaptic")
            plt.colorbar(im, ax=self.ax_weights)
        
        # Neuromodulation
        self.ax_neuromod.clear()
        self.ax_neuromod.set_title("Neuromodulator Concentrations")
        self.ax_neuromod.set_xlabel("Time (ms)")
        self.ax_neuromod.set_ylabel("Concentration")
        times = self.time_buffer
        for mod in ["dopamine", "octopamine", "serotonin"]:
            if self.neuromod_buffer[mod]:
                self.ax_neuromod.plot(times[-len(self.neuromod_buffer[mod]):], 
                                    self.neuromod_buffer[mod], label=mod)
        self.ax_neuromod.legend()
        
        # Firing rates
        self.ax_rates.clear()
        self.ax_rates.set_title("Population Firing Rates")
        self.ax_rates.set_xlabel("Time (ms)")
        self.ax_rates.set_ylabel("Rate (Hz)")
        for region in self.network.config.regions:
            if self.rate_buffer[region]:
                self.ax_rates.plot(times[-len(self.rate_buffer[region]):], 
                                 self.rate_buffer[region], label=region)
        self.ax_rates.legend(fontsize=8)
        
        # Connectivity
        self.ax_connect.clear()
        self.ax_connect.set_title("Region Connectivity")
        regions = self.network.config.regions
        n = len(regions)
        conn_matrix = np.zeros((n, n))
        for i, pre in enumerate(regions):
            for j, post in enumerate(regions):
                syn = self.network.get_synapse_group(pre, post)
                if syn:
                    conn_matrix[j, i] = np.mean(syn.weights)
        im = self.ax_connect.imshow(conn_matrix, cmap='RdBu_r', vmin=0, vmax=1)
        self.ax_connect.set_xticks(range(n))
        self.ax_connect.set_yticks(range(n))
        self.ax_connect.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
        self.ax_connect.set_yticklabels(regions, fontsize=8)
        plt.colorbar(im, ax=self.ax_connect)
        
        self.fig.tight_layout()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize fruit fly brain activity")
    parser.add_argument("--config", default="config/connectome_test.yaml", help="Config file")
    parser.add_argument("--duration", type=int, default=1000, help="Duration in ms")
    parser.add_argument("--max-synapses", type=int, default=10000, help="Max synapses for synthetic connectome")
    args = parser.parse_args()
    
    # Create network
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    neuron_params = {}
    for region_name, region_cfg in config.get('regions', {}).items():
        neuron_params_cfg = region_cfg.get('neuron_params', {})
        neuron_params[region_name] = LIFParams(
            C_m=neuron_params_cfg.get('C_m', 200.0),
            g_L=neuron_params_cfg.get('g_L', 10.0),
            E_L=neuron_params_cfg.get('E_L', -60.0),
            V_th=neuron_params_cfg.get('V_th', -50.0),
            V_reset=neuron_params_cfg.get('V_reset', -65.0),
            t_ref=neuron_params_cfg.get('t_ref', 2.0),
            E_rev=neuron_params_cfg.get('E_rev', -70.0),
            g_syn_max=neuron_params_cfg.get('g_syn_max', 1.0),
            tau_syn=neuron_params_cfg.get('tau_syn', 5.0),
        )
    
    stdp_params = {}
    for conn_name, conn_cfg in config.get('stdp_params', {}).items():
        stdp_params[conn_name] = STDPParams(
            tau_pre=conn_cfg.get('tau_pre', 20.0),
            tau_post=conn_cfg.get('tau_post', 20.0),
            A_pre=conn_cfg.get('A_pre', 0.01),
            A_post=conn_cfg.get('A_post', -0.012),
        )
    
    net_config = NetworkConfig(
        dt=config.get('simulation', {}).get('dt', 0.1),
        regions=list(config.get('regions', {}).keys()),
        neuron_params=neuron_params,
        stdp_params=stdp_params,
        use_neuromodulation=True,
    )
    
    builder = NetworkBuilder(net_config)
    loader = ConnectomeLoader(args.config)
    connectome = loader.create_synthetic(config, max_synapses=args.max_synapses)
    builder.build_from_connectome(connectome)
    
    # Visualize
    viz = BrainVisualizer(builder, connectome)
    ani = animation.FuncAnimation(viz.fig, viz.update, interval=50, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
