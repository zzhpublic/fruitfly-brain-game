"""Pong game environment with spike-based observations."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

class PongEnv(gym.Env):
    """
    Pong environment for Drosophila brain.
    
    Observations: Spike trains from optic lobe neurons (motion detection)
    Actions: Spike trains to descending neurons (up/down)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, 
                 screen_width: int = 160,
                 screen_height: int = 120,
                 paddle_height: int = 20,
                 ball_speed: float = 3.0,
                 max_steps: int = 10000,
                 dt: float = 0.1):  # ms per step
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_height = paddle_height
        self.ball_speed = ball_speed
        self.max_steps = max_steps
        self.dt = dt
        
        # Game state
        self.paddle_y = screen_height // 2
        self.ball_x = screen_width // 2
        self.ball_y = screen_height // 2
        self.ball_vx = ball_speed
        self.ball_vy = np.random.uniform(-1, 1)
        
        # Optic lobe encoding (motion detection)
        self.n_motion_neurons = 64  # 8x8 grid
        self.motion_receptive_fields = self._create_receptive_fields()
        
        # Descending neuron decoding
        self.n_action_neurons = 10  # 5 up, 5 down
        
        # Observation space: spike counts per motion neuron over window
        self.obs_window = 50  # ms
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.n_motion_neurons,), dtype=np.float32
        )
        
        # Action space: spike counts for up/down neurons
        self.action_space = spaces.Box(
            low=0, high=50, shape=(self.n_action_neurons,), dtype=np.float32
        )
        
        self.step_count = 0
        self.score = 0
    
    def _create_receptive_fields(self):
        """Create motion detector receptive fields."""
        fields = []
        grid_x, grid_y = 8, 8
        for i in range(grid_x):
            for j in range(grid_y):
                cx = (i + 0.5) * self.screen_width / grid_x
                cy = (j + 0.5) * self.screen_height / grid_y
                fields.append((cx, cy))
        return np.array(fields)
    
    def _encode_observation(self) -> np.ndarray:
        """Encode game state as optic lobe spike rates."""
        # Ball position relative to receptive fields
        distances = np.sqrt(
            (self.motion_receptive_fields[:, 0] - self.ball_x)**2 +
            (self.motion_receptive_fields[:, 1] - self.ball_y)**2
        )
        
        # Gaussian tuning
        sigma = 15.0
        rates = np.exp(-distances**2 / (2 * sigma**2)) * 100  # max 100 Hz
        
        # Add motion direction sensitivity (Reichardt detector style)
        # Simplified: just use ball velocity
        motion_signal = np.abs(self.ball_vx) + np.abs(self.ball_vy)
        rates *= (1 + motion_signal * 0.1)
        
        # Poisson spike generation over observation window
        spike_probs = 1 - np.exp(-rates * self.obs_window / 1000.0)
        spikes = np.random.random(self.n_motion_neurons) < spike_probs
        
        return spikes.astype(np.float32)
    
    def _decode_action(self, action: np.ndarray) -> float:
        """Decode descending neuron spikes to paddle movement."""
        up_spikes = action[:5].sum()
        down_spikes = action[5:].sum()
        
        # Rate coding
        up_rate = up_spikes / 5.0
        down_rate = down_spikes / 5.0
        
        # Movement proportional to rate difference
        move = (up_rate - down_rate) * 10.0  # pixels per step
        return move
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        # Decode action
        move = self._decode_action(action)
        self.paddle_y = np.clip(self.paddle_y + move, 
                                self.paddle_height // 2, 
                                self.screen_height - self.paddle_height // 2)
        
        # Update ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Wall collisions
        if self.ball_y <= 0 or self.ball_y >= self.screen_height:
            self.ball_vy *= -1
            self.ball_y = np.clip(self.ball_y, 0, self.screen_height)
        
        # Paddle collision
        paddle_left = 10
        paddle_right = 20
        if (self.ball_x <= paddle_right and 
            abs(self.ball_y - self.paddle_y) <= self.paddle_height // 2):
            self.ball_vx *= -1.1  # Speed up slightly
            self.ball_x = paddle_right
            self.score += 1
        
        # Missed ball
        terminated = False
        if self.ball_x < 0:
            terminated = True
            reward = -1.0
        elif self.ball_x > self.screen_width:
            # Ball passed opponent (simplified)
            self.ball_x = self.screen_width // 2
            self.ball_y = self.screen_height // 2
            self.ball_vx = -self.ball_speed
            self.ball_vy = np.random.uniform(-1, 1)
            reward = 0.0
        else:
            reward = 0.01  # Small survival reward
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        obs = self._encode_observation()
        info = {"score": self.score, "ball_pos": (self.ball_x, self.ball_y)}
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.paddle_y = self.screen_height // 2
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        self.ball_vx = self.ball_speed * np.random.choice([-1, 1])
        self.ball_vy = np.random.uniform(-1, 1)
        self.step_count = 0
        self.score = 0
        return self._encode_observation(), {}
    
    def render(self):
        """Simple text render."""
        if self.render_mode == "human":
            grid = np.zeros((20, 40), dtype=str)
            grid[:, :] = " "
            # Ball
            bx = int(self.ball_x * 40 / self.screen_width)
            by = int(self.ball_y * 20 / self.screen_height)
            if 0 <= bx < 40 and 0 <= by < 20:
                grid[by, bx] = "●"
            # Paddle
            py = int(self.paddle_y * 20 / self.screen_height)
            ph = max(1, int(self.paddle_height * 20 / self.screen_height))
            for i in range(py - ph//2, py + ph//2):
                if 0 <= i < 20:
                    grid[i, 1] = "█"
            print("\n".join("".join(row) for row in grid))
            print(f"Score: {self.score}")
