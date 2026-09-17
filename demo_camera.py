#!/usr/bin/env python3
"""
Camera-based demo for fruit fly brain game.
Uses webcam input to drive the neural network in real-time.
"""
import argparse
import numpy as np
import cv2
import yaml
import sys
from pathlib import Path

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


class CameraProcessor:
    """Process camera frames into spike trains for network input."""
    
    def __init__(self, n_neurons: int = 64, grid_size: tuple = (8, 8)):
        self.n_neurons = n_neurons
        self.grid_size = grid_size
        self.prev_frame = None
        self.motion_history = []
        self.max_history = 5
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert camera frame to spike rates for optic lobe neurons.
        
        Args:
            frame: BGR camera frame
            
        Returns:
            Spike rates for n_neurons (Hz)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.grid_size[0] * 8, self.grid_size[1] * 8))
        
        # Compute motion (frame difference)
        if self.prev_frame is not None:
            diff = cv2.absdiff(gray, self.prev_frame)
            # Threshold to get motion regions
            _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        else:
            motion = np.zeros_like(gray)
        
        self.prev_frame = gray.copy()
        self.motion_history.append(motion)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)
        
        # Accumulate motion over history
        if len(self.motion_history) > 0:
            accumulated = np.sum(self.motion_history, axis=0).astype(np.float32)
        else:
            accumulated = motion.astype(np.float32)
        
        # Divide into grid and compute average motion per cell
        cell_h = accumulated.shape[0] // self.grid_size[1]
        cell_w = accumulated.shape[1] // self.grid_size[0]
        
        spike_rates = np.zeros(self.n_neurons)
        for i in range(self.grid_size[1]):
            for j in range(self.grid_size[0]):
                idx = i * self.grid_size[0] + j
                if idx >= self.n_neurons:
                    break
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_motion = accumulated[y1:y2, x1:x2].mean()
                # Convert to spike rate (0-100 Hz)
                spike_rates[idx] = min(cell_motion / 10.0, 100.0)
        
        return spike_rates
    
    def process_optical_flow(self, frame: np.ndarray) -> np.ndarray:
        """Alternative: use optical flow for motion detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        if self.prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        else:
            magnitude = np.zeros_like(gray)
        
        self.prev_frame = gray.copy()
        
        # Pool into grid
        cell_h = magnitude.shape[0] // self.grid_size[1]
        cell_w = magnitude.shape[1] // self.grid_size[0]
        
        spike_rates = np.zeros(self.n_neurons)
        for i in range(self.grid_size[1]):
            for j in range(self.grid_size[0]):
                idx = i * self.grid_size[0] + j
                if idx >= self.n_neurons:
                    break
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                spike_rates[idx] = min(magnitude[y1:y2, x1:x2].mean() * 5, 100.0)
        
        return spike_rates


class CameraDemo:
    """Main demo class integrating camera, network, and game."""
    
    def __init__(self, config_path: str, checkpoint_path: str = None, 
                 game: str = "pong", camera_id: int = 0, use_flow: bool = False):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.game_name = game
        self.camera_id = camera_id
        self.use_flow = use_flow
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Build network
        self._build_network()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # Create game environment
        self._create_game()
        
        # Camera processor
        n_optic = len(self.network.get_neuron_ids("optic_lobes"))
        self.camera_processor = CameraProcessor(n_neurons=n_optic)
        
        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # State
        self.running = True
        self.step_count = 0
        self.total_reward = 0.0
        
    def _build_network(self):
        """Build network from config."""
        neuron_params = {}
        for region_name, region_cfg in self.config.get('regions', {}).items():
            np_cfg = region_cfg.get('neuron_params', {})
            neuron_params[region_name] = LIFParams(
                C_m=np_cfg.get('C_m', 200.0), g_L=np_cfg.get('g_L', 10.0),
                E_L=np_cfg.get('E_L', -60.0), V_th=np_cfg.get('V_th', -50.0),
                V_reset=np_cfg.get('V_reset', -65.0), t_ref=np_cfg.get('t_ref', 2.0),
                E_rev=np_cfg.get('E_rev', -70.0), g_syn_max=np_cfg.get('g_syn_max', 1.0),
                tau_syn=np_cfg.get('tau_syn', 5.0),
            )
        
        stdp_params = {}
        for conn_name, conn_cfg in self.config.get('stdp_params', {}).items():
            stdp_params[conn_name] = STDPParams(
                tau_pre=conn_cfg.get('tau_pre', 20.0),
                tau_post=conn_cfg.get('tau_post', 20.0),
                A_pre=conn_cfg.get('A_pre', 0.01),
                A_post=conn_cfg.get('A_post', -0.012),
                w_min=conn_cfg.get('w_min', 0.0),
                w_max=conn_cfg.get('w_max', 1.0),
            )
        
        net_config = NetworkConfig(
            dt=self.config.get('simulation', {}).get('dt', 0.1),
            regions=list(self.config.get('regions', {}).keys()),
            neuron_params=neuron_params,
            stdp_params=stdp_params,
            conn_scale=self.config.get('simulation', {}).get('conn_scale', 1.0),
            enforce_dale=self.config.get('simulation', {}).get('enforce_dale', True),
            use_neuromodulation=self.config.get('simulation', {}).get('use_neuromodulation', True),
        )
        
        self.network = NetworkBuilder(net_config)
        loader = ConnectomeLoader(self.config_path)
        connectome = loader.create_synthetic(self.config, max_synapses=10000)
        self.network.build_from_connectome(connectome)
        
        print(f"Network: {len(self.network.neurons)} neurons, {len(self.network.synapses)} synapse groups")
        for region in self.network.config.regions:
            n = len(self.network.get_neuron_ids(region))
            print(f"  {region}: {n} neurons")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore synaptic weights
        for (pre_r, post_r), syn in self.network.synapses.items():
            key = f"{pre_r}->{post_r}"
            if key in checkpoint['weights']:
                data = checkpoint['weights'][key]
                syn.weights[:] = data['weights']
                syn.pre_trace[:] = data['pre_trace']
                syn.post_trace[:] = data['post_trace']
        
        # Restore neuron states
        for region, pop in self.network.populations.items():
            if region in checkpoint['neuron_states']:
                data = checkpoint['neuron_states'][region]
                pop.V[:] = data['V']
                pop.refractory[:] = data['refractory']
                pop.g_syn[:] = data['g_syn']
        
        print(f"Loaded checkpoint from episode {checkpoint['episode']} (game: {checkpoint['game']})")
        
        # Disable learning during demo
        for syn in self.network.synapses.values():
            syn.A_pre = 0.0
            syn.A_post = 0.0
    
    def _create_game(self):
        """Create game environment."""
        if self.game_name == "pong":
            self.env = PongEnv(render_mode="human")
        elif self.game_name == "maze":
            self.env = MazeEnv(render_mode="human")
        elif self.game_name == "odor":
            self.env = OdorNavigationEnv(render_mode="human")
        elif self.game_name == "looming":
            self.env = LoomingEscapeEnv(render_mode="human")
        elif self.game_name == "pinball":
            self.env = PinballEnv(render_mode="human")
        else:
            raise ValueError(f"Unknown game: {self.game_name}")
        
        self.obs, _ = self.env.reset()
        print(f"Game: {self.game_name}, obs shape: {self.obs.shape}")
    
    def _map_camera_to_input(self, spike_rates: np.ndarray) -> dict:
        """Map camera spike rates to network external input."""
        external_input = {}
        
        # Primary: optic lobes get camera input
        optic_ids = self.network.get_neuron_ids("optic_lobes")
        if len(optic_ids) > 0:
            I_optic = np.zeros(len(optic_ids))
            scale = len(optic_ids) / len(spike_rates)
            for i, rate in enumerate(spike_rates):
                if rate > 0:
                    idx = int(i * scale) % len(optic_ids)
                    # Convert rate (Hz) to current (pA)
                    I_optic[idx] += rate * 0.5
            external_input["optic_lobes"] = I_optic
        
        # For maze: also provide compass input from horizontal motion
        if self.game_name == "maze":
            central_ids = self.network.get_neuron_ids("central_complex")
            if len(central_ids) > 0:
                I_central = np.zeros(len(central_ids))
                # Use left-right motion difference as compass
                left_motion = spike_rates[:len(spike_rates)//2].sum()
                right_motion = spike_rates[len(spike_rates)//2:].sum()
                direction = (right_motion - left_motion) / (left_motion + right_motion + 1e-6)
                idx = int((direction + 1) / 2 * len(central_ids)) % len(central_ids)
                I_central[idx] += abs(direction) * 50
                external_input["central_complex"] = I_central
        
        # For odor: mushroom body gets odor-like input
        if self.game_name == "odor":
            mb_ids = self.network.get_neuron_ids("mushroom_body")
            if len(mb_ids) > 0:
                I_mb = np.zeros(len(mb_ids))
                # Use overall motion as odor concentration
                total_motion = spike_rates.sum()
                for i in range(len(mb_ids)):
                    I_mb[i] += total_motion * 0.1
                external_input["mushroom_body"] = I_mb
        
        # For looming: optic lobes get looming-like input
        if self.game_name == "looming":
            optic_ids = self.network.get_neuron_ids("optic_lobes")
            if len(optic_ids) > 0:
                I_optic = np.zeros(len(optic_ids))
                # Expanding motion pattern
                for i, rate in enumerate(spike_rates):
                    if rate > 10:  # Threshold for "looming"
                        idx = int(i * len(optic_ids) / len(spike_rates)) % len(optic_ids)
                        I_optic[idx] += rate * 2.0
                external_input["optic_lobes"] = I_optic
        
        # For pinball: optic lobes get ball/paddle motion
        if self.game_name == "pinball":
            optic_ids = self.network.get_neuron_ids("optic_lobes")
            if len(optic_ids) > 0:
                I_optic = np.zeros(len(optic_ids))
                # Use motion for ball tracking
                for i, rate in enumerate(spike_rates):
                    if rate > 5:
                        idx = int(i * len(optic_ids) / len(spike_rates)) % len(optic_ids)
                        I_optic[idx] += rate * 1.0
                external_input["optic_lobes"] = I_optic
        
        return external_input
    
    def _map_spikes_to_action(self, spikes: dict) -> np.ndarray:
        """Map network output spikes to game action."""
        if self.game_name == "pong":
            action = np.zeros(10)
            desc_spikes = spikes.get("central_complex", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(10):
                    idx = int(i * len(desc_spikes) / 10)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            return action
            
        elif self.game_name == "maze":
            action = np.zeros(12)
            desc_spikes = spikes.get("central_complex", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(12):
                    idx = int(i * len(desc_spikes) / 12)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            return action
            
        elif self.game_name == "odor":
            action = np.zeros(12)
            desc_spikes = spikes.get("lateral_horn", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(12):
                    idx = int(i * len(desc_spikes) / 12)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            return action
            
        elif self.game_name == "looming":
            action = np.zeros(10)
            desc_spikes = spikes.get("central_complex", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(10):
                    idx = int(i * len(desc_spikes) / 10)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            return action
            
        elif self.game_name == "pinball":
            action = np.zeros(10)
            desc_spikes = spikes.get("central_complex", np.array([]))
            if len(desc_spikes) > 0:
                for i in range(10):
                    idx = int(i * len(desc_spikes) / 10)
                    if idx < len(desc_spikes):
                        action[i] = desc_spikes[idx].astype(float) * 10
            return action
        
        return np.array([])
    
    def _draw_overlay(self, frame: np.ndarray, spike_rates: np.ndarray, 
                      action: np.ndarray, reward: float, info: dict) -> np.ndarray:
        """Draw info overlay on camera frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw grid visualization of spike rates
        grid_size = self.camera_processor.grid_size
        cell_h = h // grid_size[1]
        cell_w = w // grid_size[0]
        
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                idx = i * grid_size[0] + j
                if idx >= len(spike_rates):
                    break
                rate = spike_rates[idx]
                intensity = int(min(rate / 100.0 * 255, 255))
                color = (0, intensity, 255 - intensity)  # Blue to red
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Blend overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text info
        cv2.putText(frame, f"Game: {self.game_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Step: {self.step_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Reward: {reward:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Reward: {self.total_reward:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Game-specific info
        if self.game_name == "pong":
            cv2.putText(frame, f"Score: {info.get('score', 0)}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif self.game_name == "maze":
            cv2.putText(frame, f"Dist to goal: {info.get('dist_to_goal', 0):.1f}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif self.game_name == "odor":
            cv2.putText(frame, f"Dist: {info.get('dist', 0):.1f}, Conc: {info.get('conc', 0):.1f}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif self.game_name == "looming":
            cv2.putText(frame, f"Escaped: {info.get('escaped', False)}, Steps: {self.step_count}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif self.game_name == "pinball":
            cv2.putText(frame, f"Score: {info.get('score', 0)}, Hits: {info.get('hits', 0)}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Action visualization
        if len(action) > 0:
            action_str = "Action: " + " ".join(f"{a:.1f}" for a in action[:5])
            cv2.putText(frame, action_str, (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main demo loop."""
        print("\n=== Camera Demo Started ===")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset game")
        print("  's' - Save screenshot")
        print("  'f' - Toggle optical flow mode")
        print("============================\n")
        
        while self.running:
            # Read camera frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame to spike rates
            if self.use_flow:
                spike_rates = self.camera_processor.process_optical_flow(frame)
            else:
                spike_rates = self.camera_processor.process_frame(frame)
            
            # Map to network input
            external_input = self._map_camera_to_input(spike_rates)
            
            # Step network
            spikes = self.network.step(external_input=external_input)
            
            # Map to action
            action = self._map_spikes_to_action(spikes)
            
            # Step game
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            self.step_count += 1
            
            # Draw overlay
            display_frame = self._draw_overlay(frame.copy(), spike_rates, action, reward, info)
            
            # Show camera feed
            cv2.imshow('Fruit Fly Brain - Camera Input', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.obs, _ = self.env.reset()
                self.total_reward = 0.0
                self.step_count = 0
                print("Game reset!")
            elif key == ord('s'):
                cv2.imwrite(f'demo_screenshot_{self.step_count}.png', display_frame)
                print(f"Screenshot saved!")
            elif key == ord('f'):
                self.use_flow = not self.use_flow
                print(f"Optical flow mode: {'ON' if self.use_flow else 'OFF'}")
            
            # Check game over
            if terminated or truncated:
                print(f"Game over! Score: {info.get('score', 'N/A')}, Total reward: {self.total_reward:.2f}")
                self.obs, _ = self.env.reset()
                self.total_reward = 0.0
                self.step_count = 0
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.env.close()
        print("Demo ended.")


def main():
    parser = argparse.ArgumentParser(description="Camera demo for fruit fly brain")
    parser.add_argument("--config", default="config/connectome_test.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load")
    parser.add_argument("--game", choices=["pong", "maze", "odor", "looming", "pinball"], default="pong")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--flow", action="store_true", help="Use optical flow instead of frame diff")
    args = parser.parse_args()
    
    demo = CameraDemo(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        game=args.game,
        camera_id=args.camera,
        use_flow=args.flow
    )
    
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()