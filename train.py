#!/usr/bin/env python3
"""
Training script for fruit fly brain game playing.
"""
import argparse
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from networks.assembly import NetworkBuilder, NetworkConfig
from neurons.lif import LIFParams
from synapses.stdp import STDPParams
from neuromod.modulator import UnifiedNeuromodulation
from connectome.loader import ConnectomeLoader
from games.pong import PongEnv
from games.maze import MazeEnv
from games.odor import OdorNavigationEnv
from games.looming import LoomingEscapeEnv


def create_network(config_path: str = "config/connectome.yaml", max_synapses: int = 10000) -> NetworkBuilder:
    """Create network from config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Neuron params per region
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
    
    # STDP params
    stdp_params = {}
    for conn_name, conn_cfg in config.get('stdp_params', {}).items():
        stdp_params[conn_name] = STDPParams(
            tau_pre=conn_cfg.get('tau_pre', 20.0),
            tau_post=conn_cfg.get('tau_post', 20.0),
            A_pre=conn_cfg.get('A_pre', 0.01),
            A_post=conn_cfg.get('A_post', -0.012),
            w_min=conn_cfg.get('w_min', 0.0),
            w_max=conn_cfg.get('w_max', 1.0),
        )
    
    net_config = NetworkConfig(
        dt=config.get('simulation', {}).get('dt', 0.1),
        regions=list(config.get('regions', {}).keys()),
        neuron_params=neuron_params,
        stdp_params=stdp_params,
        conn_scale=config.get('simulation', {}).get('conn_scale', 1.0),
        enforce_dale=config.get('simulation', {}).get('enforce_dale', True),
        use_neuromodulation=config.get('simulation', {}).get('use_neuromodulation', True),
    )
    
    builder = NetworkBuilder(net_config)
    
    # Load or create connectome
    loader = ConnectomeLoader(config_path)
    connectome = loader.create_synthetic(config, max_synapses=max_synapses)
    
    builder.build_from_connectome(connectome)
    
    return builder, connectome


def train_pong(network: NetworkBuilder, n_episodes: int = 100, render: bool = False):
    """Train on Pong."""
    env = PongEnv()
    
    optic_ids = network.get_neuron_ids("optic_lobes")
    descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
    
    print(f"Optic lobe neurons: {len(optic_ids)}")
    print(f"Descending neurons: {len(descending_ids)}")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            I_optic = np.zeros(len(optic_ids))
            if len(optic_ids) > 0:
                scale = len(optic_ids) / 64
                for i, spike in enumerate(obs):
                    if spike > 0:
                        idx = int(i * scale) % len(optic_ids)
                        I_optic[idx] += 50.0
            
            external_input = {"optic_lobes": I_optic}
            spikes = network.step(external_input=external_input)
            
            action = np.zeros(10)
            if len(descending_ids) > 0:
                desc_spikes = spikes.get("central_complex", np.array([]))
                if len(desc_spikes) > 0:
                    for i in range(10):
                        idx = int(i * len(desc_spikes) / 10)
                        if idx < len(desc_spikes):
                            action[i] = desc_spikes[idx].astype(float) * 10
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Score={info['score']}, Reward={total_reward:.2f}")
    
    env.close()


def train_maze(network: NetworkBuilder, n_episodes: int = 100, render: bool = False):
    """Train on Maze."""
    env = MazeEnv()
    
    optic_ids = network.get_neuron_ids("optic_lobes")
    central_ids = network.get_neuron_ids("central_complex")
    descending_ids = network.get_neuron_ids("descending_neurons") or central_ids
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            I_optic = np.zeros(len(optic_ids))
            I_central = np.zeros(len(central_ids))
            
            visual_obs = obs[:36]
            for i, spike in enumerate(visual_obs):
                if spike > 0 and len(optic_ids) > 0:
                    idx = int(i * len(optic_ids) / 36) % len(optic_ids)
                    I_optic[idx] += 30.0
            
            compass_obs = obs[36:52]
            for i, spike in enumerate(compass_obs):
                if spike > 0 and len(central_ids) > 0:
                    idx = int(i * len(central_ids) / 16) % len(central_ids)
                    I_central[idx] += 40.0
            
            external_input = {"optic_lobes": I_optic, "central_complex": I_central}
            spikes = network.step(external_input=external_input)
            
            action = np.zeros(12)
            desc_spikes = spikes.get("central_complex", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(12):
                    idx = int(i * len(desc_spikes) / 12)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Dist={info['dist_to_goal']:.1f}, Reward={total_reward:.2f}")
    
    env.close()


def train_odor(network: NetworkBuilder, n_episodes: int = 100, render: bool = False):
    """Train on Odor Navigation."""
    env = OdorNavigationEnv()
    
    mb_ids = network.get_neuron_ids("mushroom_body")
    lh_ids = network.get_neuron_ids("lateral_horn")
    descending_ids = network.get_neuron_ids("descending_neurons") or lh_ids
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            I_mb = np.zeros(len(mb_ids))
            I_lh = np.zeros(len(lh_ids))
            
            orn_obs = obs[:200]
            for i, spike in enumerate(orn_obs):
                if spike > 0 and len(mb_ids) > 0:
                    idx = int(i * len(mb_ids) / 200) % len(mb_ids)
                    I_mb[idx] += 20.0
            
            wind_obs = obs[200:]
            for i, spike in enumerate(wind_obs):
                if spike > 0 and len(lh_ids) > 0:
                    idx = int(i * len(lh_ids) / 8) % len(lh_ids)
                    I_lh[idx] += 30.0
            
            external_input = {"mushroom_body": I_mb, "lateral_horn": I_lh}
            spikes = network.step(external_input=external_input)
            
            action = np.zeros(12)
            desc_spikes = spikes.get("lateral_horn", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(12):
                    idx = int(i * len(desc_spikes) / 12)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Dist={info['dist']:.1f}, Conc={info['conc']:.1f}, Reward={total_reward:.2f}")
    
    env.close()


def train_looming(network: NetworkBuilder, n_episodes: int = 100, render: bool = False):
    """Train on Looming Escape."""
    env = LoomingEscapeEnv()
    
    optic_ids = network.get_neuron_ids("optic_lobes")
    descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            I_optic = np.zeros(len(optic_ids))
            for i, spike in enumerate(obs):
                if spike > 0 and len(optic_ids) > 0:
                    idx = int(i * len(optic_ids) / 20) % len(optic_ids)
                    I_optic[idx] += 100.0
            
            external_input = {"optic_lobes": I_optic}
            spikes = network.step(external_input=external_input)
            
            action = np.zeros(10)
            desc_spikes = spikes.get("central_complex", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(10):
                    idx = int(i * len(desc_spikes) / 10)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Escaped={info['escaped']}, Reward={total_reward:.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train fruit fly brain on games")
    parser.add_argument("--game", choices=["pong", "maze", "odor", "looming"], default="pong")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--config", default="config/connectome.yaml")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print(f"Creating network from {args.config}...")
    network, connectome = create_network(args.config, max_synapses=10000)
    print(f"Network: {len(network.neurons)} neurons, {len(network.synapses)} synapse groups")
    
    if args.game == "pong":
        train_pong(network, args.episodes, args.render)
    elif args.game == "maze":
        train_maze(network, args.episodes, args.render)
    elif args.game == "odor":
        train_odor(network, args.episodes, args.render)
    elif args.game == "looming":
        train_looming(network, args.episodes, args.render)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
