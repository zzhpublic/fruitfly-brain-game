#!/usr/bin/env python3
"""
Demo: Real-time Fruit Fly Brain Activation Visualization

Shows which brain regions are active during closed-loop game playing.
Displays spike rasters, firing rates, and 3D brain region activation.

Usage:
    python demo_brain_activation.py --game pinball --steps 500
    python demo_brain_activation.py --game pong --steps 500
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.connectome.loader import ConnectomeLoader
from src.networks.assembly import NetworkBuilder, NetworkConfig
from src.games.pinball import PinballEnv
from src.games.pong import PongEnv
from src.games.maze import MazeEnv
from src.games.odor import OdorNavigationEnv
from src.games.looming import LoomingEscapeEnv


class BrainActivationVisualizer:
    """Real-time visualization of fruit fly brain activation."""
    
    def __init__(self, network_builder, game_env, game_name="game"):
        self.network = network_builder
        self.game = game_env
        self.game_name = game_name
        self.step_count = 0
        self.max_steps = 1000
        
        # History for plotting
        self.history_len = 200
        self.spike_history = {region: [] for region in network_builder.populations.keys()}
        self.rate_history = {region: [] for region in network_builder.populations.keys()}
        self.voltage_history = {region: [] for region in network_builder.populations.keys()}
        
        # Brain region positions for 3D visualization (approximate Drosophila brain coordinates)
        self.region_positions = {
            'optic_lobes': (0.2, 0.8, 0.5),      # Lateral, anterior
            'mushroom_body': (0.5, 0.5, 0.3),    # Central, middle
            'central_complex': (0.5, 0.3, 0.5),  # Central, posterior
            'lateral_horn': (0.8, 0.6, 0.4),     # Lateral, anterior
            'neuromodulatory': (0.5, 0.5, 0.7),  # Central, dorsal
            'descending_neurons': (0.5, 0.1, 0.5), # Central, ventral
        }
        
        # Region colors
        self.region_colors = {
            'optic_lobes': '#FF6B6B',      # Red - visual input
            'mushroom_body': '#4ECDC4',    # Teal - learning
            'central_complex': '#45B7D1',  # Blue - navigation
            'lateral_horn': '#FFA07A',     # Light salmon - olfactory
            'neuromodulatory': '#98D8C8',  # Mint - modulation
            'descending_neurons': '#F7DC6F', # Yellow - motor output
        }
        
        # Setup figure
        self.fig = plt.figure(figsize=(18, 10))
        self.gs = gridspec.GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 3D brain view (top-left)
        self.ax_brain = self.fig.add_subplot(self.gs[0:2, 0:2], projection='3d')
        
        # Spike raster (top-right)
        self.ax_raster = self.fig.add_subplot(self.gs[0, 2:])
        
        # Firing rates over time (middle-right)
        self.ax_rates = self.fig.add_subplot(self.gs[1, 2:])
        
        # Voltage traces (bottom-left)
        self.ax_voltage = self.fig.add_subplot(self.gs[2, 0:2])
        
        # Game state (bottom-right)
        self.ax_game = self.fig.add_subplot(self.gs[2, 2:])
        
        # Initialize plots
        self._init_plots()
        
        # Game frame for rendering
        self.game_frame = None
    
    def _init_plots(self):
        """Initialize all subplots."""
        # 3D Brain
        self.ax_brain.set_title('Fruit Fly Brain - Region Activation', fontsize=12, fontweight='bold')
        self.ax_brain.set_xlabel('Lateral ← → Medial')
        self.ax_brain.set_ylabel('Anterior ← → Posterior')
        self.ax_brain.set_zlabel('Ventral ← → Dorsal')
        self.ax_brain.set_xlim(0, 1)
        self.ax_brain.set_ylim(0, 1)
        self.ax_brain.set_zlim(0, 1)
        
        # Draw brain regions as spheres
        self.region_spheres = {}
        for region, (x, y, z) in self.region_positions.items():
            if region in self.network.populations:
                color = self.region_colors.get(region, '#888888')
                # Create sphere
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                r = 0.05
                xs = x + r * np.outer(np.cos(u), np.sin(v))
                ys = y + r * np.outer(np.sin(u), np.sin(v))
                zs = z + r * np.outer(np.ones_like(u), np.cos(v))
                sphere = self.ax_brain.plot_surface(xs, ys, zs, alpha=0.3, color=color, 
                                                     edgecolor=color, linewidth=0.5)
                self.region_spheres[region] = (sphere, (x, y, z), color)
        
        # Spike raster
        self.ax_raster.set_title('Spike Raster (Last 200 Steps)', fontsize=11)
        self.ax_raster.set_xlabel('Time Step')
        self.ax_raster.set_ylabel('Region')
        self.ax_raster.set_xlim(0, self.history_len)
        
        # Firing rates
        self.ax_rates.set_title('Population Firing Rates', fontsize=11)
        self.ax_rates.set_xlabel('Time Step')
        self.ax_rates.set_ylabel('Rate (Hz)')
        self.ax_rates.set_xlim(0, self.history_len)
        self.rate_lines = {}
        
        # Voltage traces
        self.ax_voltage.set_title('Sample Neuron Voltages', fontsize=11)
        self.ax_voltage.set_xlabel('Time Step')
        self.ax_voltage.set_ylabel('Voltage (mV)')
        self.ax_voltage.set_xlim(0, self.history_len)
        self.ax_voltage.set_ylim(-80, 20)
        self.voltage_lines = {}
        
        # Game state
        self.ax_game.set_title(f'Game: {self.game_name}', fontsize=11)
        self.ax_game.axis('off')
    
    def update_brain_3d(self, spikes_dict):
        """Update 3D brain visualization with current activation."""
        # Clear previous activation spheres
        for artist in self.ax_brain.collections[:]:
            if hasattr(artist, '_is_activation'):
                artist.remove()
        
        # Draw activation for each region
        for region, spikes in spikes_dict.items():
            if region not in self.region_positions:
                continue
            
            x, y, z = self.region_positions[region]
            base_color = self.region_colors.get(region, '#888888')
            
            # Calculate activation level (0-1)
            n_neurons = self.network.populations[region].n_neurons
            if n_neurons > 0:
                activation = min(spikes.sum() / n_neurons * 10, 1.0)  # Scale for visibility
            else:
                activation = 0
            
            if activation > 0.01:
                # Draw activation sphere
                r = 0.03 + 0.07 * activation
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                xs = x + r * np.outer(np.cos(u), np.sin(v))
                ys = y + r * np.outer(np.sin(u), np.sin(v))
                zs = z + r * np.outer(np.ones_like(u), np.cos(v))
                
                # Color intensity based on activation
                alpha = 0.3 + 0.5 * activation
                sphere = self.ax_brain.plot_surface(xs, ys, zs, alpha=alpha, color=base_color,
                                                     edgecolor='none')
                sphere._is_activation = True
                
                # Add text label with spike count
                self.ax_brain.text(x, y, z + 0.12, f'{region}\n{spikes.sum():.0f} spikes',
                                   fontsize=8, ha='center', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=base_color, alpha=0.7))
    
    def update_raster(self, spikes_dict):
        """Update spike raster plot."""
        self.ax_raster.clear()
        self.ax_raster.set_title('Spike Raster (Last 200 Steps)', fontsize=11)
        self.ax_raster.set_xlabel('Time Step')
        self.ax_raster.set_ylabel('Region')
        
        if self.step_count > 1:
            self.ax_raster.set_xlim(max(0, self.step_count - self.history_len), self.step_count)
        
        y_pos = 0
        y_labels = []
        y_ticks = []
        
        for region, spikes in spikes_dict.items():
            if region not in self.network.populations:
                continue
            
            # Get spike indices
            spike_indices = np.where(spikes > 0)[0]
            if len(spike_indices) > 0:
                # Plot spikes at current time step
                times = np.full(len(spike_indices), self.step_count)
                self.ax_raster.scatter(times, np.full(len(spike_indices), y_pos) + 
                                       np.random.uniform(-0.3, 0.3, len(spike_indices)),
                                       c=self.region_colors.get(region, '#888888'), 
                                       s=10, alpha=0.6, marker='|')
            
            y_labels.append(region)
            y_ticks.append(y_pos)
            y_pos += 1
        
        if y_ticks:
            self.ax_raster.set_yticks(y_ticks)
            self.ax_raster.set_yticklabels(y_labels)
            self.ax_raster.set_ylim(-0.5, y_pos - 0.5)
    
    def update_rates(self, spikes_dict):
        """Update firing rate history."""
        for region, spikes in spikes_dict.items():
            if region not in self.rate_history:
                continue
            
            n_neurons = self.network.populations[region].n_neurons
            rate = spikes.sum() / n_neurons * 1000  # Convert to Hz (assuming 1ms dt)
            self.rate_history[region].append(rate)
            
            # Keep history length
            if len(self.rate_history[region]) > self.history_len:
                self.rate_history[region].pop(0)
        
        # Plot
        self.ax_rates.clear()
        self.ax_rates.set_title('Population Firing Rates', fontsize=11)
        self.ax_rates.set_xlabel('Time Step')
        self.ax_rates.set_ylabel('Rate (Hz)')
        
        for region, history in self.rate_history.items():
            if len(history) > 0:
                x = range(max(0, self.step_count - len(history)), self.step_count)
                if len(x) == len(history):
                    self.ax_rates.plot(x, history, label=region, 
                                       color=self.region_colors.get(region, '#888888'), linewidth=1.5)
        
        handles, labels = self.ax_rates.get_legend_handles_labels()
        if handles:
            self.ax_rates.legend(loc='upper right', fontsize=8)
        if self.step_count > 1:
            self.ax_rates.set_xlim(max(0, self.step_count - self.history_len), self.step_count)
    
    def update_voltages(self):
        """Update voltage traces for sample neurons."""
        self.ax_voltage.clear()
        self.ax_voltage.set_title('Sample Neuron Voltages', fontsize=11)
        self.ax_voltage.set_xlabel('Time Step')
        self.ax_voltage.set_ylabel('Voltage (mV)')
        self.ax_voltage.set_ylim(-80, 20)
        
        for region, pop in self.network.populations.items():
            if pop.n_neurons == 0:
                continue
            
            # Sample a few neurons
            sample_idx = min(3, pop.n_neurons)
            for i in range(sample_idx):
                # We need to track voltage history - for now show current
                pass
        
        # Show current voltages as bar
        regions = list(self.network.populations.keys())
        voltages = [np.mean(pop.V) for pop in self.network.populations.values()]
        colors = [self.region_colors.get(r, '#888888') for r in regions]
        
        bars = self.ax_voltage.bar(regions, voltages, color=colors, alpha=0.7, edgecolor='black')
        self.ax_voltage.axhline(y=-40, color='red', linestyle='--', alpha=0.5, label='Threshold (-40mV)')
        self.ax_voltage.axhline(y=-60, color='gray', linestyle='--', alpha=0.5, label='Rest (-60mV)')
        handles, labels = self.ax_voltage.get_legend_handles_labels()
        if handles:
            self.ax_voltage.legend(fontsize=8)
        self.ax_voltage.set_xticks(range(len(regions)))
        self.ax_voltage.set_xticklabels(regions, rotation=45, ha='right')
    
    def update_game(self):
        """Update game state visualization."""
        self.ax_game.clear()
        self.ax_game.set_title(f'Game: {self.game_name} | Step: {self.step_count}', fontsize=11)
        
        # Render game frame
        frame = self.game.render()
        if frame is not None:
            self.ax_game.imshow(frame)
        self.ax_game.axis('off')
    
    def step(self):
        """Run one simulation step and update visualization."""
        # Get game observation
        obs = self.game._encode_observation()
        
        # Resize observation to match optic_lobes population size
        optic_pop = self.network.populations.get('optic_lobes')
        if optic_pop is not None and len(obs) != optic_pop.n_neurons:
            # Repeat or truncate to match
            if len(obs) < optic_pop.n_neurons:
                # Tile the observation
                repeats = int(np.ceil(optic_pop.n_neurons / len(obs)))
                obs = np.tile(obs, repeats)[:optic_pop.n_neurons]
            else:
                obs = obs[:optic_pop.n_neurons]
        
        # Scale input: spike rates (0-1) → current (pA)
        # Need ~500,000 pA to spike, so scale by ~5000
        INPUT_SCALE = 5000.0
        obs = obs * INPUT_SCALE
        
        # Network step
        spikes = self.network.step(external_input={'optic_lobes': obs})
        
        # Decode action
        dn_spikes = spikes.get('descending_neurons', np.zeros(50))
        left_spikes = dn_spikes[:5].sum() if len(dn_spikes) >= 5 else 0
        right_spikes = dn_spikes[5:10].sum() if len(dn_spikes) >= 10 else 0
        stay_spikes = dn_spikes[10:15].sum() if len(dn_spikes) >= 15 else 0
        
        rates = [left_spikes/5, stay_spikes/5, right_spikes/5]
        action = np.argmax(rates)
        
        # Game step
        obs, reward, terminated, truncated, info = self.game.step(np.zeros(10))
        
        # Update visualizations
        self.update_brain_3d(spikes)
        self.update_raster(spikes)
        self.update_rates(spikes)
        self.update_voltages()
        self.update_game()
        
        self.step_count += 1
        
        return spikes, action, reward, terminated, truncated, info
    
    def save_frame(self, filepath):
        """Save current figure to file."""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    def close(self):
        plt.close(self.fig)


def create_game(game_name, auto_paddle=True):
    """Create game environment by name."""
    games = {
        'pinball': lambda: PinballEnv(auto_paddle=auto_paddle, render_mode='rgb_array'),
        'pong': lambda: PongEnv(render_mode='rgb_array'),
        'maze': lambda: MazeEnv(render_mode='rgb_array'),
        'odor': lambda: OdorNavigationEnv(render_mode='rgb_array'),
        'looming': lambda: LoomingEscapeEnv(render_mode='rgb_array'),
    }
    
    if game_name not in games:
        raise ValueError(f"Unknown game: {game_name}. Options: {list(games.keys())}")
    
    return games[game_name]()


def main():
    parser = argparse.ArgumentParser(description='Fruit Fly Brain Activation Demo')
    parser.add_argument('--game', type=str, default='pinball', 
                        choices=['pinball', 'pong', 'maze', 'odor', 'looming'],
                        help='Game to play')
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--save-frames', action='store_true', help='Save frames as images')
    parser.add_argument('--output-dir', type=str, default='./brain_activation_frames', help='Output directory')
    parser.add_argument('--auto-paddle', action='store_true', default=True, help='Auto-paddle for pinball')
    parser.add_argument('--headless', action='store_true', default=True, help='Run in headless mode (no display)')
    args = parser.parse_args()
    
    # Create output directory
    if args.save_frames:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Fruit Fly Brain Activation Demo")
    print(f"Game: {args.game}")
    print(f"Steps: {args.steps}")
    print(f"=" * 60)
    
    # Load connectome and build network
    print("Loading connectome...")
    loader = ConnectomeLoader("config/connectome_test.yaml")
    connectome_data = loader.create_synthetic(max_synapses=100000)
    
    print(f"Connectome: {len(connectome_data.neurons)} neurons, {len(connectome_data.synapses)} synapses")
    print(f"Regions: {connectome_data.regions}")
    
    # Build network
    print("Building network...")
    config = NetworkConfig(
        regions=connectome_data.regions,
        use_neuromodulation=True,
        enforce_dale=False
    )
    builder = NetworkBuilder(config)
    builder.build_from_connectome(connectome_data)
    
    print(f"Network: {sum(p.n_neurons for p in builder.populations.values())} neurons")
    for name, pop in builder.populations.items():
        print(f"  {name}: {pop.n_neurons} neurons")
    
    # Create game
    print(f"Creating game: {args.game}")
    game = create_game(args.game, auto_paddle=args.auto_paddle)
    obs, _ = game.reset()
    
    # Create visualizer
    print("Starting visualization...")
    visualizer = BrainActivationVisualizer(builder, game, args.game)
    
    # Run simulation
    print("Running simulation...")
    for step in range(args.steps):
        spikes, action, reward, terminated, truncated, info = visualizer.step()
        
        # Print progress
        total_spikes = sum(s.sum() for s in spikes.values())
        active_regions = [r for r, s in spikes.items() if s.sum() > 0]
        
        if step % 20 == 0:
            print(f"Step {step:4d}: {total_spikes:6.0f} spikes | "
                  f"Active: {', '.join(active_regions) if active_regions else 'none'} | "
                  f"Action: {['LEFT','STAY','RIGHT'][action]} | "
                  f"Reward: {reward:.2f}")
        
        # Save frame
        if args.save_frames and step % 5 == 0:
            filepath = os.path.join(args.output_dir, f'frame_{step:04d}.png')
            visualizer.save_frame(filepath)
        
        if terminated or truncated:
            print(f"Game ended at step {step}")
            break
    
    # Save final frame
    if args.save_frames:
        filepath = os.path.join(args.output_dir, f'frame_final.png')
        visualizer.save_frame(filepath)
        print(f"Frames saved to {args.output_dir}")
    
    visualizer.close()
    print("Done!")


if __name__ == "__main__":
    main()