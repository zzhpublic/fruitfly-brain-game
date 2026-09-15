"""Looming escape environment (optic lobes -> descending neurons)."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

class LoomingEscapeEnv(gym.Env):
    """
    Looming stimulus escape for optic lobe -> giant fiber -> descending neurons.
    
    Observations: Looming-sensitive neuron spikes (LPLC2-like)
    Actions: Jump/fly away (descending neurons)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self,
                 n_looming_neurons: int = 20,
                 max_loom_time: float = 1000.0,  # ms
                 dt: float = 0.1,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        self.n_looming_neurons = n_looming_neurons
        self.max_loom_time = max_loom_time
        self.dt = dt
        
        # Looming parameters
        self.l_over_v = 50.0  # l/|v| ratio (size/speed)
        self.threshold = 0.5  # Angular size threshold for escape
        
        # Looming neuron tuning (different l/|v| preferences)
        self.neuron_l_over_v = np.logspace(np.log10(10), np.log10(200), n_looming_neurons)
        
        # Observation: looming neuron spikes
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(n_looming_neurons,), dtype=np.float32
        )
        
        # Action: escape jump (1) or not (0) - population code
        self.n_action_neurons = 10
        self.action_space = spaces.Box(
            low=0, high=50, shape=(self.n_action_neurons,), dtype=np.float32
        )
        
        self.loom_time = 0.0
        self.loom_active = False
        self.escaped = False
        self.step_count = 0
    
    def _angular_size(self, t: float) -> float:
        """Angular size of looming object at time t."""
        # theta(t) = 2 * arctan(l / (2 * (v*t)))
        # For small angles: theta ≈ l / (v*t)
        if t <= 0:
            return 0.0
        return 2 * np.arctan(self.l_over_v / (2 * t))
    
    def _angular_velocity(self, t: float) -> float:
        """Angular expansion velocity."""
        if t <= 0:
            return 0.0
        return self.l_over_v / (t**2 + (self.l_over_v/2)**2)
    
    def _encode_observation(self) -> np.ndarray:
        """LPLC2-like looming detection."""
        if not self.loom_active:
            return np.zeros(self.n_looming_neurons, dtype=np.float32)
        
        theta = self._angular_size(self.loom_time)
        omega = self._angular_velocity(self.loom_time)
        
        # LPLC2 model: responds to angular velocity * size
        # Each neuron tuned to different l/|v|
        rates = np.zeros(self.n_looming_neurons)
        for i, lv in enumerate(self.neuron_l_over_v):
            # Response peaks when stimulus l/|v| matches neuron preference
            # Simplified: response ~ omega * exp(-(log(l/v) - log(lv))^2)
            log_ratio = np.log(self.l_over_v / (self.loom_time + 1e-6))
            log_pref = np.log(lv)
            tuning = np.exp(-(log_ratio - log_pref)**2 / 0.5)
            rates[i] = omega * tuning * 1000
        
        # Poisson spikes
        spike_probs = 1 - np.exp(-rates * self.dt / 1000.0)
        spikes = (np.random.random(self.n_looming_neurons) < spike_probs).astype(np.float32)
        
        return spikes
    
    def _decode_action(self, action: np.ndarray) -> bool:
        """Population vote for escape."""
        escape_votes = action.sum()
        return escape_votes > 5 * 25  # Threshold
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        escape = self._decode_action(action)
        
        if self.loom_active:
            self.loom_time += self.dt
        
        # Check collision (loom reaches threshold)
        theta = self._angular_size(self.loom_time) if self.loom_active else 0
        collision = theta > self.threshold
        
        terminated = False
        reward = 0.0
        
        if collision:
            if escape and not self.escaped:
                # Successful escape
                reward = 10.0
                self.escaped = True
                terminated = True
            elif not escape:
                # Failed to escape
                reward = -10.0
                terminated = True
        
        # Randomly start new loom
        if not self.loom_active and np.random.random() < 0.001:
            self.loom_active = True
            self.loom_time = 1.0  # Start at 1ms before collision
            self.escaped = False
        
        if self.escaped:
            self.loom_active = False
            self.loom_time = 0.0
        
        self.step_count += 1
        truncated = self.step_count >= 10000
        
        obs = self._encode_observation()
        info = {"theta": theta, "loom_active": self.loom_active, "escaped": self.escaped}
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.loom_time = 0.0
        self.loom_active = False
        self.escaped = False
        self.step_count = 0
        return self._encode_observation(), {}
    
    def render(self):
        if self.render_mode == "human":
            if self.loom_active:
                theta = self._angular_size(self.loom_time)
                size = int(theta * 50)
                print(" " * 20 + "█" * size + f" θ={theta:.3f} rad")
            else:
                print("Waiting for loom...")
            print(f"Escaped: {self.escaped}")
