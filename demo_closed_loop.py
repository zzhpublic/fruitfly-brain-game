#!/usr/bin/env python3
"""
Closed-loop demo: Game screen capture → Fruit fly brain → Game input.
The game renders itself, we capture the frame, process it through the network,
and feed the network's output back as actions. Fully autonomous agent.
"""
import argparse
import numpy as np
import cv2
import yaml
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from networks.assembly import NetworkBuilder, NetworkConfig
from neurons.lif import LIFParams
from synapses.stdp import STDPParams
from connectome.loader import ConnectomeLoader
from games.pong import PongEnv
from games.maze import MazeEnv
from games.odor import OdorNavigationEnv
from games.looming import LoomingEscapeEnv
from games.pinball import PinballEnv


class GameScreenProcessor:
    """Process game screen frames into spike trains for network input."""
    
    def __init__(self, n_neurons: int = 64, grid_size: tuple = (8, 8), 
                 method: str = "frame_diff"):
        self.n_neurons = n_neurons
        self.grid_size = grid_size
        self.method = method
        self.prev_frame = None
        self.prev_gray = None
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert game screen frame to spike rates.
        
        Args:
            frame: RGB game frame from env.render()
            
        Returns:
            Spike rates for n_neurons (Hz)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Resize to grid
        gray = cv2.resize(gray, (self.grid_size[0] * 8, self.grid_size[1] * 8))
        
        if self.method == "frame_diff":
            return self._process_frame_diff(gray)
        elif self.method == "optical_flow":
            return self._process_optical_flow(gray)
        elif self.method == "intensity":
            return self._process_intensity(gray)
        elif self.method == "edges":
            return self._process_edges(gray)
        else:
            return self._process_frame_diff(gray)
    
    def _process_frame_diff(self, gray: np.ndarray) -> np.ndarray:
        """Frame difference for motion detection."""
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        else:
            motion = np.zeros_like(gray)
        
        self.prev_gray = gray.copy()
        
        # Pool into grid
        return self._pool_to_grid(motion.astype(np.float32), scale=0.5)
    
    def _process_optical_flow(self, gray: np.ndarray) -> np.ndarray:
        """Optical flow for motion vectors."""
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        else:
            magnitude = np.zeros_like(gray, dtype=np.float32)
        
        self.prev_gray = gray.copy()
        return self._pool_to_grid(magnitude, scale=5.0)
    
    def _process_intensity(self, gray: np.ndarray) -> np.ndarray:
        """Direct intensity mapping (for static objects)."""
        return self._pool_to_grid(gray.astype(np.float32), scale=0.4)
    
    def _process_edges(self, gray: np.ndarray) -> np.ndarray:
        """Edge detection for object boundaries."""
        edges = cv2.Canny(gray, 50, 150)
        return self._pool_to_grid(edges.astype(np.float32), scale=1.0)
    
    def _pool_to_grid(self, img: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Pool image into grid of neurons."""
        cell_h = img.shape[0] // self.grid_size[1]
        cell_w = img.shape[1] // self.grid_size[0]
        
        spike_rates = np.zeros(self.n_neurons)
        for i in range(self.grid_size[1]):
            for j in range(self.grid_size[0]):
                idx = i * self.grid_size[0] + j
                if idx >= self.n_neurons:
                    break
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_val = img[y1:y2, x1:x2].mean()
                spike_rates[idx] = min(cell_val * scale, 100.0)
        
        return spike_rates
    
    def reset(self):
        """Reset processor state."""
        self.prev_frame = None
        self.prev_gray = None


class ClosedLoopDemo:
    """Closed-loop: Game → Screen → Network → Action → Game"""
    
    def __init__(self, config_path: str, checkpoint_path: str = None,
                 game: str = "pong", method: str = "frame_diff",
                 render_scale: float = 2.0, fps_limit: int = 30):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.game_name = game
        self.method = method
        self.render_scale = render_scale
        self.fps_limit = fps_limit
        self.frame_time = 1.0 / fps_limit
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Build network
        self._build_network()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # Create game environment (no render_mode, we'll render manually)
        self._create_game()
        
        # Screen processor
        n_optic = len(self.network.get_neuron_ids("optic_lobes"))
        self.screen_processor = GameScreenProcessor(
            n_neurons=n_optic, 
            grid_size=(8, 8),
            method=method
        )
        
        # Display window
        self.window_name = f"Fruit Fly Brain - {game.upper()} (Closed Loop)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 
                        int(640 * render_scale), int(480 * render_scale))
        
        # Stats
        self.step_count = 0
        self.total_reward = 0.0
        self.episode = 0
        self.running = True
        self.paused = False
        self.show_overlay = True
        
        # For visualization
        self.spike_history = []
        self.action_history = []
        self.reward_history = []
        
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
            self.env = PongEnv()
        elif self.game_name == "maze":
            self.env = MazeEnv()
        elif self.game_name == "odor":
            self.env = OdorNavigationEnv()
        elif self.game_name == "looming":
            self.env = LoomingEscapeEnv()
        elif self.game_name == "pinball":
            self.env = PinballEnv()
        else:
            raise ValueError(f"Unknown game: {self.game_name}")
        
        self.obs, _ = self.env.reset()
        print(f"Game: {self.game_name}, obs shape: {self.obs.shape}")
    
    def _get_game_frame(self) -> np.ndarray:
        """Get game screen as RGB array."""
        # Render game to RGB array
        frame = self.env.render()
        if frame is None:
            # Fallback: create a simple visualization
            frame = self._create_fallback_frame()
        return frame
    
    def _create_fallback_frame(self) -> np.ndarray:
        """Create a visual representation from game state."""
        if self.game_name == "pong":
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Draw ball
            bx = int(self.env.ball_x * 320 / self.env.screen_width)
            by = int(self.env.ball_y * 240 / self.env.screen_height)
            cv2.circle(frame, (bx, by), 5, (255, 255, 255), -1)
            # Draw paddle
            py = int(self.env.paddle_y * 240 / self.env.screen_height)
            ph = int(self.env.paddle_height * 240 / self.env.screen_height)
            cv2.rectangle(frame, (10, py - ph//2), (20, py + ph//2), (0, 255, 0), -1)
            return frame
        elif self.game_name == "maze":
            frame = np.zeros((300, 300, 3), dtype=np.uint8)
            # Draw maze walls
            for y in range(self.env.maze_size):
                for x in range(self.env.maze_size):
                    if self.env.maze[y, x] == 1:
                        cv2.rectangle(frame, (x*20, y*20), ((x+1)*20, (y+1)*20), (100, 100, 100), -1)
            # Draw agent
            ax = int(self.env.agent_pos[0] * 20)
            ay = int(self.env.agent_pos[1] * 20)
            cv2.circle(frame, (ax, ay), 8, (0, 255, 0), -1)
            # Draw goal
            gx = int(self.env.goal_pos[0] * 20)
            gy = int(self.env.goal_pos[1] * 20)
            cv2.circle(frame, (gx, gy), 8, (0, 0, 255), -1)
            return frame
        else:
            return np.zeros((240, 320, 3), dtype=np.uint8)
    
    def _map_screen_to_input(self, spike_rates: np.ndarray) -> dict:
        """Map screen spike rates to network external input."""
        external_input = {}
        
        # Primary: optic lobes get screen input
        # Scale factor: spike rates (0-100 Hz) → current (pA)
        # Need ~500,000 pA to spike, so scale by ~5000
        INPUT_SCALE = 5000.0
        
        optic_ids = self.network.get_neuron_ids("optic_lobes")
        if len(optic_ids) > 0:
            I_optic = np.zeros(len(optic_ids))
            scale = len(optic_ids) / len(spike_rates)
            for i, rate in enumerate(spike_rates):
                if rate > 0:
                    idx = int(i * scale) % len(optic_ids)
                    I_optic[idx] += rate * INPUT_SCALE
            external_input["optic_lobes"] = I_optic
        
        # Game-specific additional inputs
        if self.game_name == "maze":
            central_ids = self.network.get_neuron_ids("central_complex")
            if len(central_ids) > 0:
                I_central = np.zeros(len(central_ids))
                # Use horizontal motion as compass
                left = spike_rates[:len(spike_rates)//2].sum()
                right = spike_rates[len(spike_rates)//2:].sum()
                direction = (right - left) / (left + right + 1e-6)
                idx = int((direction + 1) / 2 * len(central_ids)) % len(central_ids)
                I_central[idx] += abs(direction) * 50000
                external_input["central_complex"] = I_central
        
        elif self.game_name == "odor":
            mb_ids = self.network.get_neuron_ids("mushroom_body")
            if len(mb_ids) > 0:
                I_mb = np.zeros(len(mb_ids))
                total = spike_rates.sum()
                for i in range(len(mb_ids)):
                    I_mb[i] += total * INPUT_SCALE * 0.1
                external_input["mushroom_body"] = I_mb
        
        elif self.game_name == "looming":
            optic_ids = self.network.get_neuron_ids("optic_lobes")
            if len(optic_ids) > 0:
                I_optic = np.zeros(len(optic_ids))
                for i, rate in enumerate(spike_rates):
                    if rate > 10:
                        idx = int(i * len(optic_ids) / len(spike_rates)) % len(optic_ids)
                        I_optic[idx] += rate * INPUT_SCALE * 2.0
                external_input["optic_lobes"] = I_optic
        
        elif self.game_name == "pinball":
            optic_ids = self.network.get_neuron_ids("optic_lobes")
            if len(optic_ids) > 0:
                I_optic = np.zeros(len(optic_ids))
                for i, rate in enumerate(spike_rates):
                    if rate > 5:
                        idx = int(i * len(optic_ids) / len(spike_rates)) % len(optic_ids)
                        I_optic[idx] += rate * INPUT_SCALE * 1.0
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
        """Draw info overlay on game frame."""
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # Draw spike rate grid overlay (top-left)
        grid_size = self.screen_processor.grid_size
        cell_h = min(120, h // grid_size[1])
        cell_w = min(160, w // grid_size[0])
        
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                idx = i * grid_size[0] + j
                if idx >= len(spike_rates):
                    break
                rate = spike_rates[idx]
                intensity = int(min(rate / 100.0 * 255, 255))
                color = (0, intensity, 255 - intensity)
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cv2.rectangle(display, (x1, y1), (x2, y2), color, -1)
        
        # Blend
        alpha = 0.4
        cv2.addWeighted(display, alpha, frame, 1 - alpha, 0, display)
        
        # Text info panel (right side)
        panel_x = w - 250
        if panel_x < 0:
            panel_x = 10
        
        y = 30
        cv2.putText(display, f"GAME: {self.game_name.upper()}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(display, f"Episode: {self.episode}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(display, f"Step: {self.step_count}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(display, f"Reward: {reward:.2f}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if reward > 0 else (0, 0, 255), 2)
        y += 30
        cv2.putText(display, f"Total: {self.total_reward:.2f}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(display, f"Method: {self.method}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        cv2.putText(display, f"FPS Limit: {self.fps_limit}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Game-specific info
        y += 30
        if self.game_name == "pong":
            cv2.putText(display, f"Score: {info.get('score', 0)}", (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif self.game_name == "maze":
            cv2.putText(display, f"Dist: {info.get('dist_to_goal', 0):.1f}", (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif self.game_name == "odor":
            cv2.putText(display, f"Dist: {info.get('dist', 0):.1f}", (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 25
            cv2.putText(display, f"Conc: {info.get('conc', 0):.1f}", (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif self.game_name == "looming":
            cv2.putText(display, f"Escaped: {info.get('escaped', False)}", (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif self.game_name == "pinball":
            cv2.putText(display, f"Score: {info.get('score', 0)}, Hits: {info.get('hits', 0)}", (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Action visualization
        y += 40
        if len(action) > 0:
            cv2.putText(display, "ACTION:", (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
            for i in range(min(10, len(action))):
                bar_len = int(abs(action[i]) * 2)
                color = (0, 255, 0) if action[i] > 0 else (0, 0, 255)
                cv2.rectangle(display, (panel_x, y), (panel_x + bar_len, y + 10), color, -1)
                y += 12
        
        # Spike activity indicator
        y += 10
        # Get spikes from the last network step (stored in self.last_spikes)
        total_spikes = sum(len(s) for s in self.last_spikes.values()) if hasattr(self, 'last_spikes') else 0
        cv2.putText(display, f"Network Spikes: {total_spikes}", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls help
        y += 30
        cv2.putText(display, "CONTROLS:", (panel_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        for ctrl in ["Q: Quit", "P: Pause", "R: Reset", "S: Screenshot", 
                     "O: Toggle Overlay", "M: Change Method"]:
            cv2.putText(display, ctrl, (panel_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y += 18
        
        return display
    
    def run(self):
        """Main closed-loop demo."""
        print("\n=== CLOSED-LOOP DEMO ===")
        print("Game Screen → Network → Action → Game")
        print("==========================\n")
        print("Controls:")
        print("  Q - Quit")
        print("  P - Pause/Resume")
        print("  R - Reset game")
        print("  S - Save screenshot")
        print("  O - Toggle overlay")
        print("  M - Change processing method")
        print("  +/- - Adjust FPS limit")
        print("==========================\n")
        
        last_time = time.time()
        
        while self.running:
            loop_start = time.time()
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Resumed'}")
            elif key == ord('r'):
                self._reset_game()
            elif key == ord('s'):
                self._save_screenshot()
            elif key == ord('o'):
                self.show_overlay = not self.show_overlay
            elif key == ord('m'):
                self._cycle_method()
            elif key == ord('+') or key == ord('='):
                self.fps_limit = min(120, self.fps_limit + 5)
                self.frame_time = 1.0 / self.fps_limit
                print(f"FPS limit: {self.fps_limit}")
            elif key == ord('-') or key == ord('_'):
                self.fps_limit = max(5, self.fps_limit - 5)
                self.frame_time = 1.0 / self.fps_limit
                print(f"FPS limit: {self.fps_limit}")
            
            if self.paused:
                # Still show frame when paused
                frame = self._get_game_frame()
                if self.show_overlay:
                    frame = self._draw_overlay(frame, np.zeros(self.screen_processor.n_neurons), 
                                             np.array([]), 0, {})
                cv2.imshow(self.window_name, frame)
                continue
            
            # 1. GET GAME SCREEN
            frame = self._get_game_frame()
            
            # 2. PROCESS SCREEN → SPIKE RATES
            spike_rates = self.screen_processor.process_frame(frame)
            
            # 3. MAP TO NETWORK INPUT
            external_input = self._map_screen_to_input(spike_rates)
            
            # 4. NETWORK STEP
            spikes = self.network.step(external_input=external_input)
            self.last_spikes = spikes  # Store for overlay
            
            # 5. MAP NETWORK OUTPUT → ACTION
            action = self._map_spikes_to_action(spikes)
            
            # 6. GAME STEP
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            self.step_count += 1
            
            # 7. DRAW OVERLAY
            if self.show_overlay:
                display_frame = self._draw_overlay(frame, spike_rates, action, reward, info)
            else:
                display_frame = frame
            
            # 8. SHOW
            cv2.imshow(self.window_name, display_frame)
            
            # 9. CHECK GAME OVER
            if terminated or truncated:
                self.episode += 1
                print(f"Episode {self.episode} done! Score: {info.get('score', 'N/A')}, "
                      f"Total reward: {self.total_reward:.2f}, Steps: {self.step_count}")
                self._reset_game()
            
            # FPS limiting
            elapsed = time.time() - loop_start
            sleep_time = self.frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Cleanup
        cv2.destroyAllWindows()
        self.env.close()
        print(f"\nDemo ended. Episodes: {self.episode}, Total steps: {self.step_count}")
    
    def _reset_game(self):
        """Reset game and network state."""
        self.obs, _ = self.env.reset()
        self.screen_processor.reset()
        self.total_reward = 0.0
        self.step_count = 0
        # Optionally reset network state
        for pop in self.network.populations.values():
            pop.V[:] = pop.params.E_L
            pop.refractory[:] = 0.0
            pop.g_syn[:] = 0.0
        print("Game reset!")
    
    def _save_screenshot(self):
        """Save current frame."""
        frame = self._get_game_frame()
        if self.show_overlay:
            spike_rates = self.screen_processor.process_frame(frame)
            external_input = self._map_screen_to_input(spike_rates)
            # Use stored spikes from last step to avoid advancing network
            spikes = getattr(self, 'last_spikes', {})
            action = self._map_spikes_to_action(spikes)
            frame = self._draw_overlay(frame, spike_rates, action, 0, {})
        filename = f"closedloop_{self.game_name}_ep{self.episode}_step{self.step_count}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def _cycle_method(self):
        """Cycle through processing methods."""
        methods = ["frame_diff", "optical_flow", "intensity", "edges"]
        idx = methods.index(self.method)
        self.method = methods[(idx + 1) % len(methods)]
        self.screen_processor.method = self.method
        self.screen_processor.reset()
        print(f"Method changed to: {self.method}")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop: Game screen → Network → Game")
    parser.add_argument("--config", default="config/connectome_test.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load")
    parser.add_argument("--game", choices=["pong", "maze", "odor", "looming", "pinball"], default="pong")
    parser.add_argument("--method", choices=["frame_diff", "optical_flow", "intensity", "edges"], 
                        default="frame_diff", help="Screen processing method")
    parser.add_argument("--scale", type=float, default=2.0, help="Display scale factor")
    parser.add_argument("--fps", type=int, default=30, help="FPS limit")
    args = parser.parse_args()
    
    demo = ClosedLoopDemo(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        game=args.game,
        method=args.method,
        render_scale=args.scale,
        fps_limit=args.fps
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