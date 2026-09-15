"""Odor navigation environment (mushroom body / lateral horn)."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

class OdorNavigationEnv(gym.Env):
    """
    Odor plume navigation for mushroom body / lateral horn.
    
    Observations: Olfactory receptor neuron (ORN) spikes + wind direction
    Actions: Motor commands for upwind casting
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    
    def __init__(self,
                 arena_size: float = 100.0,
                 source_pos: Tuple[float, float] = None,
                 wind_speed: float = 1.0,
                 wind_dir: float = 0.0,  # radians
                 plume_sigma: float = 5.0,
                 max_steps: int = 5000,
                 dt: float = 0.1,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        self.arena_size = arena_size
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        self.plume_sigma = plume_sigma
        self.max_steps = max_steps
        self.dt = dt
        
        # Odor source
        self.source_pos = np.array(source_pos) if source_pos else np.array([arena_size * 0.8, arena_size * 0.5])
        
        # Agent
        self.agent_pos = np.array([arena_size * 0.1, arena_size * 0.5])
        self.agent_dir = 0.0
        
        # Olfactory system: 50 ORN types (different odorants)
        self.n_orn_types = 50
        self.n_orn_per_type = 4  # 4 neurons per glomerulus
        self.n_orn_total = self.n_orn_types * self.n_orn_per_type
        
        # Wind sensing (antennae)
        self.n_wind_neurons = 8
        
        self.observation_space = spaces.Box(
            low=0, high=100, 
            shape=(self.n_orn_total + self.n_wind_neurons,), 
            dtype=np.float32
        )
        
        # Action: turn + forward
        self.n_action_neurons = 12
        self.action_space = spaces.Box(
            low=0, high=50, shape=(self.n_action_neurons,), dtype=np.float32
        )
        
        self.step_count = 0
        self.trajectory = []
    
    def _odor_concentration(self, pos: np.ndarray) -> float:
        """Gaussian plume model with wind advection."""
        # Downwind distance
        wind_vec = np.array([np.cos(self.wind_dir), np.sin(self.wind_dir)])
        to_source = self.source_pos - pos
        downwind = np.dot(to_source, wind_vec)
        crosswind = np.linalg.norm(to_source - downwind * wind_vec)
        
        if downwind < 0:
            return 0.0  # Upwind of source
        
        # Gaussian plume
        conc = np.exp(-crosswind**2 / (2 * self.plume_sigma**2))
        conc *= np.exp(-downwind / (self.wind_speed * 10.0))  # Decay downwind
        return conc * 100.0
    
    def _encode_observation(self) -> np.ndarray:
        """Encode odor concentration as ORN spikes + wind direction."""
        conc = self._odor_concentration(self.agent_pos)
        
        # ORN responses: different tuning curves
        orn_rates = np.zeros(self.n_orn_total)
        for i in range(self.n_orn_types):
            # Each ORN type has different sensitivity
            sensitivity = 0.5 + i * 0.01
            threshold = i * 0.5
            rate = max(0, (conc - threshold) * sensitivity)
            # 4 neurons per type with noise
            base_idx = i * self.n_orn_per_type
            orn_rates[base_idx:base_idx+4] = rate + np.random.exponential(rate * 0.1, 4)
        
        # Wind direction encoding (antennae)
        wind_rates = np.zeros(self.n_wind_neurons)
        wind_angle = (self.wind_dir - self.agent_dir + np.pi) % (2 * np.pi) - np.pi
        idx = int((wind_angle + np.pi) / (2 * np.pi) * self.n_wind_neurons)
        if 0 <= idx < self.n_wind_neurons:
            wind_rates[idx] = self.wind_speed * 50
        
        obs = np.concatenate([orn_rates, wind_rates])
        
        # Poisson spikes
        spike_probs = 1 - np.exp(-obs * self.dt / 1000.0)
        spikes = (np.random.random(len(obs)) < spike_probs).astype(np.float32)
        
        return spikes
    
    def _decode_action(self, action: np.ndarray) -> Tuple[float, float]:
        turn_left = action[:4].sum() / 4.0
        turn_right = action[4:8].sum() / 4.0
        forward = action[8:].sum() / 4.0
        
        turn_rate = (turn_right - turn_left) * 0.3
        speed = forward * 0.3
        
        return turn_rate, speed
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        turn_rate, speed = self._decode_action(action)
        
        self.agent_dir += turn_rate
        self.agent_dir = (self.agent_dir + np.pi) % (2 * np.pi) - np.pi
        
        new_pos = self.agent_pos + np.array([np.cos(self.agent_dir), np.sin(self.agent_dir)]) * speed
        
        # Boundary
        new_pos = np.clip(new_pos, 0, self.arena_size)
        self.agent_pos = new_pos
        
        self.trajectory.append(self.agent_pos.copy())
        
        # Check source
        dist = np.linalg.norm(self.agent_pos - self.source_pos)
        terminated = dist < 2.0
        
        if terminated:
            reward = 10.0
        else:
            # Reward for upwind progress
            wind_vec = np.array([np.cos(self.wind_dir), np.sin(self.wind_dir)])
            to_source = self.source_pos - self.agent_pos
            upwind_progress = np.dot(to_source, wind_vec)
            reward = -upwind_progress * 0.01
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        obs = self._encode_observation()
        info = {"pos": self.agent_pos.copy(), "dist": dist, "conc": self._odor_concentration(self.agent_pos)}
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.agent_pos = np.array([self.arena_size * 0.1, self.arena_size * 0.5])
        self.agent_dir = 0.0
        self.step_count = 0
        self.trajectory = []
        return self._encode_observation(), {}
    
    def render(self):
        if self.render_mode == "human":
            grid_size = 40
            grid = np.zeros((grid_size, grid_size), dtype=str)
            grid[:, :] = " "
            
            # Source
            sx = int(self.source_pos[0] * grid_size / self.arena_size)
            sy = int(self.source_pos[1] * grid_size / self.arena_size)
            if 0 <= sx < grid_size and 0 <= sy < grid_size:
                grid[sy, sx] = "★"
            
            # Agent
            ax = int(self.agent_pos[0] * grid_size / self.arena_size)
            ay = int(self.agent_pos[1] * grid_size / self.arena_size)
            if 0 <= ax < grid_size and 0 <= ay < grid_size:
                grid[ay, ax] = "●"
            
            # Wind arrow
            wx = int(grid_size/2 + np.cos(self.wind_dir) * 5)
            wy = int(grid_size/2 + np.sin(self.wind_dir) * 5)
            if 0 <= wx < grid_size and 0 <= wy < grid_size:
                grid[wy, wx] = "→"
            
            for row in grid:
                print("".join(row))
            print(f"Pos: {self.agent_pos}, Conc: {self._odor_concentration(self.agent_pos):.1f}")
