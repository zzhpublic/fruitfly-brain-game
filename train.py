#!/usr/bin/env python3
"""
Training script for fruit fly brain game playing.
"""
import argparse
import numpy as np
import yaml
import pickle
import json
from pathlib import Path
import sys
from datetime import datetime

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
from games.pinball import PinballEnv


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


def save_checkpoint(network: NetworkBuilder, episode: int, metrics: dict, 
                    save_dir: Path, game: str, config_name: str):
    """Save model checkpoint and metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save synaptic weights
    weights_data = {}
    for (pre_r, post_r), syn in network.synapses.items():
        weights_data[f"{pre_r}->{post_r}"] = {
            'weights': syn.weights.copy(),
            'pre_indices': syn.pre_indices.copy(),
            'post_indices': syn.post_indices.copy(),
            'pre_trace': syn.pre_trace.copy(),
            'post_trace': syn.post_trace.copy(),
        }
    
    # Save neuron states (population states)
    neuron_states = {}
    for region, pop in network.populations.items():
        neuron_states[region] = {
            'V': pop.V.copy(),
            'refractory': pop.refractory.copy(),
            'g_syn': pop.g_syn.copy(),
        }
    
    # Save neuromodulator states
    neuromod_states = {}
    if network.neuromod:
        for mod_name in ['dopamine', 'octopamine', 'serotonin']:
            mod = getattr(network.neuromod, mod_name)
            neuromod_states[mod_name] = {
                'concentrations': mod.concentrations.copy() if hasattr(mod, 'concentrations') else {},
            }
    
    checkpoint = {
        'episode': episode,
        'game': game,
        'config': config_name,
        'timestamp': datetime.now().isoformat(),
        'weights': weights_data,
        'neuron_states': neuron_states,
        'neuromod_states': neuromod_states,
        'metrics': metrics,
    }
    
    # Save as pickle (for full state) and JSON (for metrics)
    checkpoint_path = save_dir / f"{game}_ep{episode:06d}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    metrics_path = save_dir / f"{game}_metrics.json"
    # Load existing metrics
    all_metrics = []
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
    all_metrics.append(metrics)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"  Checkpoint saved: {checkpoint_path}")


def load_checkpoint(network: NetworkBuilder, checkpoint_path: Path):
    """Load model checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Restore synaptic weights
    for (pre_r, post_r), syn in network.synapses.items():
        key = f"{pre_r}->{post_r}"
        if key in checkpoint['weights']:
            data = checkpoint['weights'][key]
            syn.weights[:] = data['weights']
            syn.pre_trace[:] = data['pre_trace']
            syn.post_trace[:] = data['post_trace']
    
    # Restore neuron states
    for region, pop in network.populations.items():
        if region in checkpoint['neuron_states']:
            data = checkpoint['neuron_states'][region]
            pop.V[:] = data['V']
            pop.refractory[:] = data['refractory']
            pop.g_syn[:] = data['g_syn']
    
    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    return checkpoint


def train_pong(network: NetworkBuilder, n_episodes: int = 100, render: bool = False,
               save_dir: Path = None, save_interval: int = 100, config_name: str = "default"):
    """Train on Pong."""
    env = PongEnv()
    
    optic_ids = network.get_neuron_ids("optic_lobes")
    descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
    
    print(f"Optic lobe neurons: {len(optic_ids)}")
    print(f"Descending neurons: {len(descending_ids)}")
    
    episode_rewards = []
    episode_scores = []
    
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
        
        episode_rewards.append(total_reward)
        episode_scores.append(info['score'])
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Score={info['score']}, Reward={total_reward:.2f}")
        
        # Save checkpoint
        if save_dir and (episode + 1) % save_interval == 0:
            metrics = {
                'episode': episode,
                'score': info['score'],
                'reward': total_reward,
                'avg_reward_100': np.mean(episode_rewards[-100:]),
                'avg_score_100': np.mean(episode_scores[-100:]),
            }
            save_checkpoint(network, episode, metrics, save_dir, "pong", config_name)
    
    env.close()
    return episode_rewards, episode_scores


def train_maze(network: NetworkBuilder, n_episodes: int = 100, render: bool = False,
               save_dir: Path = None, save_interval: int = 100, config_name: str = "default"):
    """Train on Maze."""
    env = MazeEnv()
    
    optic_ids = network.get_neuron_ids("optic_lobes")
    central_ids = network.get_neuron_ids("central_complex")
    descending_ids = network.get_neuron_ids("descending_neurons") or central_ids
    
    episode_rewards = []
    episode_dists = []
    
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
        
        episode_rewards.append(total_reward)
        episode_dists.append(info['dist_to_goal'])
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Dist={info['dist_to_goal']:.1f}, Reward={total_reward:.2f}")
        
        if save_dir and (episode + 1) % save_interval == 0:
            metrics = {
                'episode': episode,
                'dist_to_goal': info['dist_to_goal'],
                'reward': total_reward,
                'avg_reward_100': np.mean(episode_rewards[-100:]),
                'avg_dist_100': np.mean(episode_dists[-100:]),
            }
            save_checkpoint(network, episode, metrics, save_dir, "maze", config_name)
    
    env.close()
    return episode_rewards, episode_dists


def train_odor(network: NetworkBuilder, n_episodes: int = 100, render: bool = False,
               save_dir: Path = None, save_interval: int = 100, config_name: str = "default"):
    """Train on Odor Navigation."""
    env = OdorNavigationEnv()
    
    mb_ids = network.get_neuron_ids("mushroom_body")
    lh_ids = network.get_neuron_ids("lateral_horn")
    descending_ids = network.get_neuron_ids("descending_neurons") or lh_ids
    
    episode_rewards = []
    episode_dists = []
    
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
        
        episode_rewards.append(total_reward)
        episode_dists.append(info['dist'])
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Dist={info['dist']:.1f}, Conc={info['conc']:.1f}, Reward={total_reward:.2f}")
        
        if save_dir and (episode + 1) % save_interval == 0:
            metrics = {
                'episode': episode,
                'dist': info['dist'],
                'conc': info['conc'],
                'reward': total_reward,
                'avg_reward_100': np.mean(episode_rewards[-100:]),
                'avg_dist_100': np.mean(episode_dists[-100:]),
            }
            save_checkpoint(network, episode, metrics, save_dir, "odor", config_name)
    
    env.close()
    return episode_rewards, episode_dists


def train_looming(network: NetworkBuilder, n_episodes: int = 100, render: bool = False,
                  save_dir: Path = None, save_interval: int = 100, config_name: str = "default"):
    """Train on Looming Escape."""
    env = LoomingEscapeEnv()
    
    optic_ids = network.get_neuron_ids("optic_lobes")
    descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
    
    episode_rewards = []
    episode_escaped = []
    
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
        
        episode_rewards.append(total_reward)
        episode_escaped.append(info['escaped'])
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Escaped={info['escaped']}, Reward={total_reward:.2f}")
        
        if save_dir and (episode + 1) % save_interval == 0:
            metrics = {
                'episode': episode,
                'escaped': info['escaped'],
                'reward': total_reward,
                'avg_reward_100': np.mean(episode_rewards[-100:]),
                'escape_rate_100': np.mean(episode_escaped[-100:]),
            }
            save_checkpoint(network, episode, metrics, save_dir, "looming", config_name)
    
    env.close()
    return episode_rewards, episode_escaped


def train_pinball(network: NetworkBuilder, n_episodes: int = 100, render: bool = False,
                  save_dir: Path = None, save_interval: int = 100, config_name: str = "default"):
    """Train on Pinball."""
    env = PinballEnv()
    
    optic_ids = network.get_neuron_ids("optic_lobes")
    descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
    
    print(f"Optic lobe neurons: {len(optic_ids)}")
    print(f"Descending neurons: {len(descending_ids)}")
    
    episode_rewards = []
    episode_scores = []
    episode_hits = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Map observation to network input
            external_input = {}
            if len(optic_ids) > 0:
                I_optic = np.zeros(len(optic_ids))
                scale = len(optic_ids) / len(obs)
                for i, spike in enumerate(obs):
                    if spike > 0:
                        idx = int(i * scale) % len(optic_ids)
                        I_optic[idx] += 50.0
                external_input["optic_lobes"] = I_optic
            
            # Network step
            spikes = network.step(external_input=external_input)
            
            # Map network output to action
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
        
        episode_rewards.append(total_reward)
        episode_scores.append(info['score'])
        episode_hits.append(info['hits'])
        
        print(f"Episode {episode}: Score={info['score']}, Hits={info['hits']}, Reward={total_reward:.2f}")
        
        # Save checkpoint
        if save_dir and (episode + 1) % save_interval == 0:
            metrics = {
                'episode': episode,
                'score': info['score'],
                'hits': info['hits'],
                'reward': total_reward,
                'avg_reward_100': np.mean(episode_rewards[-100:]),
                'avg_score_100': np.mean(episode_scores[-100:]),
                'avg_hits_100': np.mean(episode_hits[-100:]),
            }
            save_checkpoint(network, episode, metrics, save_dir, "pinball", config_name)
    
    env.close()
    return episode_rewards, episode_scores, episode_hits


def main():
    parser = argparse.ArgumentParser(description="Train fruit fly brain on games")
    parser.add_argument("--game", choices=["pong", "maze", "odor", "looming", "pinball"], default="pong")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--config", default="config/connectome.yaml")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print(f"Creating network from {args.config}...")
    network, connectome = create_network(args.config, max_synapses=10000)
    print(f"Network: {len(network.neurons)} neurons, {len(network.synapses)} synapse groups")
    
    save_dir = Path(args.save_dir)
    config_name = Path(args.config).stem
    
    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(network, Path(args.resume))
    
    if args.game == "pong":
        train_pong(network, args.episodes, args.render, save_dir, args.save_interval, config_name)
    elif args.game == "maze":
        train_maze(network, args.episodes, args.render, save_dir, args.save_interval, config_name)
    elif args.game == "odor":
        train_odor(network, args.episodes, args.render, save_dir, args.save_interval, config_name)
    elif args.game == "looming":
        train_looming(network, args.episodes, args.render, save_dir, args.save_interval, config_name)
    elif args.game == "pinball":
        train_pinball(network, args.episodes, args.render, save_dir, args.save_interval, config_name)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
