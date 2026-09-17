#!/usr/bin/env python3
"""
Inference/evaluation script for fruit fly brain game playing.
Loads a trained checkpoint and runs the network on games.
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
    """Create network from config (same as train.py)."""
    with open(config_path) as f:
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
    loader = ConnectomeLoader(config_path)
    connectome = loader.create_synthetic(config, max_synapses=max_synapses)
    builder.build_from_connectome(connectome)
    
    return builder, connectome


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
    
    print(f"Loaded checkpoint from episode {checkpoint['episode']} (game: {checkpoint['game']})")
    return checkpoint


def map_obs_to_input(obs, network, game: str):
    """Map game observation to network input currents."""
    external_input = {}
    
    if game == "pong":
        optic_ids = network.get_neuron_ids("optic_lobes")
        I_optic = np.zeros(len(optic_ids))
        if len(optic_ids) > 0:
            scale = len(optic_ids) / 64
            for i, spike in enumerate(obs):
                if spike > 0:
                    idx = int(i * scale) % len(optic_ids)
                    I_optic[idx] += 50.0
        external_input["optic_lobes"] = I_optic
        
    elif game == "maze":
        optic_ids = network.get_neuron_ids("optic_lobes")
        central_ids = network.get_neuron_ids("central_complex")
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
        
    elif game == "odor":
        mb_ids = network.get_neuron_ids("mushroom_body")
        lh_ids = network.get_neuron_ids("lateral_horn")
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
        
    elif game == "looming":
        optic_ids = network.get_neuron_ids("optic_lobes")
        I_optic = np.zeros(len(optic_ids))
        for i, spike in enumerate(obs):
            if spike > 0 and len(optic_ids) > 0:
                idx = int(i * len(optic_ids) / 20) % len(optic_ids)
                I_optic[idx] += 100.0
        external_input["optic_lobes"] = I_optic
        
    elif game == "pinball":
        optic_ids = network.get_neuron_ids("optic_lobes")
        I_optic = np.zeros(len(optic_ids))
        if len(optic_ids) > 0:
            scale = len(optic_ids) / len(obs)
            for i, spike in enumerate(obs):
                if spike > 0:
                    idx = int(i * scale) % len(optic_ids)
                    I_optic[idx] += 50.0
        external_input["optic_lobes"] = I_optic
    
    return external_input


def map_spikes_to_action(spikes, network, game: str):
    """Map network output spikes to game action."""
    if game == "pong":
        descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
        action = np.zeros(10)
        if len(descending_ids) > 0:
            desc_spikes = spikes.get("central_complex", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(10):
                    idx = int(i * len(desc_spikes) / 10)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
        return action
        
    elif game == "maze":
        central_ids = network.get_neuron_ids("central_complex")
        action = np.zeros(12)
        desc_spikes = spikes.get("central_complex", np.array([]))
        if len(desc_spikes) > 0:
            for i in range(12):
                idx = int(i * len(desc_spikes) / 12)
                if idx < len(desc_spikes):
                    action[i] = desc_spikes[idx].astype(float) * 10
        return action
        
    elif game == "odor":
        lh_ids = network.get_neuron_ids("lateral_horn")
        action = np.zeros(12)
        desc_spikes = spikes.get("lateral_horn", np.array([]))
        if len(desc_spikes) > 0:
            for i in range(12):
                idx = int(i * len(desc_spikes) / 12)
                if idx < len(desc_spikes):
                    action[i] = desc_spikes[idx].astype(float) * 10
        return action
        
    elif game == "looming":
        descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
        action = np.zeros(10)
        desc_spikes = spikes.get("central_complex", np.array([]))
        if len(desc_spikes) > 0:
            for i in range(10):
                idx = int(i * len(desc_spikes) / 10)
                if idx < len(desc_spikes):
                    action[i] = desc_spikes[idx].astype(float) * 10
        return action
        
    elif game == "pinball":
        descending_ids = network.get_neuron_ids("descending_neurons") or network.get_neuron_ids("central_complex")
        action = np.zeros(10)
        desc_spikes = spikes.get("central_complex", np.array([]))
        if len(desc_spikes) > 0:
            for i in range(10):
                idx = int(i * len(desc_spikes) / 10)
                if idx < len(desc_spikes):
                    action[i] = desc_spikes[idx].astype(float) * 10
        return action
    
    return np.array([])


def evaluate_pong(network: NetworkBuilder, n_episodes: int = 10, render: bool = False):
    """Evaluate on Pong."""
    env = PongEnv(render_mode="human" if render else None)
    
    scores = []
    rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            external_input = map_obs_to_input(obs, network, "pong")
            spikes = network.step(external_input=external_input)
            action = map_spikes_to_action(spikes, network, "pong")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        scores.append(info['score'])
        rewards.append(total_reward)
        print(f"Episode {episode}: Score={info['score']}, Reward={total_reward:.2f}")
    
    env.close()
    return scores, rewards


def evaluate_maze(network: NetworkBuilder, n_episodes: int = 10, render: bool = False):
    """Evaluate on Maze."""
    env = MazeEnv(render_mode="human" if render else None)
    
    dists = []
    rewards = []
    success = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            external_input = map_obs_to_input(obs, network, "maze")
            spikes = network.step(external_input=external_input)
            action = map_spikes_to_action(spikes, network, "maze")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        dists.append(info['dist_to_goal'])
        rewards.append(total_reward)
        success.append(info['dist_to_goal'] < 1.0)
        print(f"Episode {episode}: Dist={info['dist_to_goal']:.1f}, Reward={total_reward:.2f}, Success={info['dist_to_goal'] < 1.0}")
    
    env.close()
    return dists, rewards, success


def evaluate_odor(network: NetworkBuilder, n_episodes: int = 10, render: bool = False):
    """Evaluate on Odor Navigation."""
    env = OdorNavigationEnv(render_mode="human" if render else None)
    
    dists = []
    rewards = []
    concs = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            external_input = map_obs_to_input(obs, network, "odor")
            spikes = network.step(external_input=external_input)
            action = map_spikes_to_action(spikes, network, "odor")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        dists.append(info['dist'])
        rewards.append(total_reward)
        concs.append(info['conc'])
        print(f"Episode {episode}: Dist={info['dist']:.1f}, Conc={info['conc']:.1f}, Reward={total_reward:.2f}")
    
    env.close()
    return dists, rewards, concs


def evaluate_looming(network: NetworkBuilder, n_episodes: int = 10, render: bool = False):
    """Evaluate on Looming Escape."""
    env = LoomingEscapeEnv(render_mode="human" if render else None)
    
    escaped = []
    rewards = []
    survival_times = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            external_input = map_obs_to_input(obs, network, "looming")
            spikes = network.step(external_input=external_input)
            action = map_spikes_to_action(spikes, network, "looming")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        escaped.append(info['escaped'])
        rewards.append(total_reward)
        survival_times.append(steps)
        print(f"Episode {episode}: Escaped={info['escaped']}, Steps={steps}, Reward={total_reward:.2f}")
    
    env.close()
    return escaped, rewards, survival_times


def evaluate_pinball(network: NetworkBuilder, n_episodes: int = 10, render: bool = False):
    """Evaluate on Pinball."""
    env = PinballEnv(render_mode="human" if render else None)
    
    scores = []
    rewards = []
    hits = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            external_input = map_obs_to_input(obs, network, "pinball")
            spikes = network.step(external_input=external_input)
            action = map_spikes_to_action(spikes, network, "pinball")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        scores.append(info['score'])
        rewards.append(total_reward)
        hits.append(info['hits'])
        print(f"Episode {episode}: Score={info['score']}, Hits={info['hits']}, Reward={total_reward:.2f}")
    
    env.close()
    return scores, rewards, hits


def main():
    parser = argparse.ArgumentParser(description="Evaluate fruit fly brain on games")
    parser.add_argument("--game", choices=["pong", "maze", "odor", "looming", "pinball"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pkl file")
    parser.add_argument("--config", default="config/connectome.yaml", help="Config used for training")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render game visually")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print(f"Creating network from {args.config}...")
    network, connectome = create_network(args.config, max_synapses=10000)
    print(f"Network: {len(network.neurons)} neurons, {len(network.synapses)} synapse groups")
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(network, Path(args.checkpoint))
    
    # Disable learning during evaluation
    for syn in network.synapses.values():
        syn.A_pre = 0.0
        syn.A_post = 0.0
    
    if args.game == "pong":
        scores, rewards = evaluate_pong(network, args.episodes, args.render)
        results = {
            'game': 'pong',
            'checkpoint': args.checkpoint,
            'episodes': args.episodes,
            'scores': scores,
            'rewards': rewards,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
        }
    elif args.game == "maze":
        dists, rewards, success = evaluate_maze(network, args.episodes, args.render)
        results = {
            'game': 'maze',
            'checkpoint': args.checkpoint,
            'episodes': args.episodes,
            'distances': dists,
            'rewards': rewards,
            'success': success,
            'mean_dist': float(np.mean(dists)),
            'std_dist': float(np.std(dists)),
            'mean_reward': float(np.mean(rewards)),
            'success_rate': float(np.mean(success)),
        }
    elif args.game == "odor":
        dists, rewards, concs = evaluate_odor(network, args.episodes, args.render)
        results = {
            'game': 'odor',
            'checkpoint': args.checkpoint,
            'episodes': args.episodes,
            'distances': dists,
            'rewards': rewards,
            'concentrations': concs,
            'mean_dist': float(np.mean(dists)),
            'std_dist': float(np.std(dists)),
            'mean_reward': float(np.mean(rewards)),
            'mean_conc': float(np.mean(concs)),
        }
    elif args.game == "looming":
        escaped, rewards, survival = evaluate_looming(network, args.episodes, args.render)
        results = {
            'game': 'looming',
            'checkpoint': args.checkpoint,
            'episodes': args.episodes,
            'escaped': escaped,
            'rewards': rewards,
            'survival_times': survival,
            'escape_rate': float(np.mean(escaped)),
            'mean_reward': float(np.mean(rewards)),
            'mean_survival': float(np.mean(survival)),
        }
    elif args.game == "pinball":
        scores, rewards, hits = evaluate_pinball(network, args.episodes, args.render)
        results = {
            'game': 'pinball',
            'checkpoint': args.checkpoint,
            'episodes': args.episodes,
            'scores': scores,
            'rewards': rewards,
            'hits': hits,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'mean_reward': float(np.mean(rewards)),
            'mean_hits': float(np.mean(hits)),
        }
    
    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            print(f"  {k}: mean={np.mean(v):.2f}, std={np.std(v):.2f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()