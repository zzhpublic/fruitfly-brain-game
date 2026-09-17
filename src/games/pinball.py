"""Pinball/Breakout game environment with spike-based observations."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

try:
    import cv2
except ImportError:
    cv2 = None


class PinballEnv(gym.Env):
    """
    Pinball/Breakout environment for Drosophila brain.
    
    Observations: Spike trains from optic lobe neurons (ball/paddle position, motion)
    Actions: Spike trains to descending neurons (left/right paddle movement)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self,
                 screen_width: int = 160,
                 screen_height: int = 120,
                 paddle_width: int = 20,  # Shorter paddle
                 paddle_height: int = 5,
                 ball_radius: int = 2,  # Smaller ball
                 ball_speed: float = 4.0,
                 max_steps: int = 10000,
                 dt: float = 0.1,  # ms per step
                 render_mode: Optional[str] = None,
                 auto_paddle: bool = True):  # Auto-paddle prediction
        super().__init__()
        self.render_mode = render_mode
        self.auto_paddle = auto_paddle
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_radius = ball_radius
        self.ball_speed = ball_speed
        self.max_steps = max_steps
        self.dt = dt
        
        # Game state
        self.paddle_x = screen_width // 2
        self.ball_x = screen_width // 2
        self.ball_y = screen_height // 2
        self.ball_vx = ball_speed * np.random.choice([-1, 1])
        self.ball_vy = -ball_speed * np.random.uniform(0.5, 1.5)  # Start going up
        
        # Optic lobe encoding (motion + position detection)
        self.n_motion_neurons = 64  # 8x8 grid for motion
        self.n_position_neurons = 32  # Position encoding
        self.motion_receptive_fields = self._create_receptive_fields()
        
        # Descending neuron decoding
        self.n_action_neurons = 10  # 5 left, 5 right
        
        # Observation space: motion spikes + position spikes
        self.obs_window = 50  # ms
        self.observation_space = spaces.Box(
            low=0, high=100, 
            shape=(self.n_motion_neurons + self.n_position_neurons,), 
            dtype=np.float32
        )
        
        # Action space: spike counts for left/right neurons
        self.action_space = spaces.Box(
            low=0, high=50, shape=(self.n_action_neurons,), dtype=np.float32
        )
        
        self.step_count = 0
        self.score = 0
        self.hits = 0
        self.prev_ball_pos = None
    
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
        # Motion detection (frame difference style)
        if self.prev_ball_pos is not None:
            dx = self.ball_x - self.prev_ball_pos[0]
            dy = self.ball_y - self.prev_ball_pos[1]
        else:
            dx, dy = 0, 0
        self.prev_ball_pos = (self.ball_x, self.ball_y)
        
        # Motion neurons: respond to ball movement in receptive field
        motion_rates = np.zeros(self.n_motion_neurons)
        for i, (cx, cy) in enumerate(self.motion_receptive_fields):
            dist = np.sqrt((cx - self.ball_x)**2 + (cy - self.ball_y)**2)
            # Gaussian tuning for position
            pos_rate = np.exp(-dist**2 / (2 * 20**2)) * 50
            # Motion sensitivity
            motion_rate = (abs(dx) + abs(dy)) * 10
            motion_rates[i] = min(pos_rate + motion_rate, 100)
        
        # Position neurons: encode ball and paddle position
        pos_rates = np.zeros(self.n_position_neurons)
        # Ball x position (16 neurons)
        ball_x_bin = int(self.ball_x / self.screen_width * 16)
        if 0 <= ball_x_bin < 16:
            pos_rates[ball_x_bin] = 80
        # Ball y position (8 neurons)
        ball_y_bin = int(self.ball_y / self.screen_height * 8)
        if 0 <= ball_y_bin < 8:
            pos_rates[16 + ball_y_bin] = 80
        # Paddle x position (8 neurons)
        paddle_x_bin = int(self.paddle_x / self.screen_width * 8)
        if 0 <= paddle_x_bin < 8:
            pos_rates[24 + paddle_x_bin] = 60
        
        # Poisson spike generation
        all_rates = np.concatenate([motion_rates, pos_rates])
        spike_probs = 1 - np.exp(-all_rates * self.obs_window / 1000.0)
        spikes = np.random.random(len(all_rates)) < spike_probs
        
        return spikes.astype(np.float32)
    
    def _decode_action(self, action: np.ndarray) -> float:
        """Decode descending neuron spikes to paddle movement."""
        left_spikes = action[:5].sum()
        right_spikes = action[5:].sum()
        
        left_rate = left_spikes / 5.0
        right_rate = right_spikes / 5.0
        
        # Movement proportional to rate difference
        move = (right_rate - left_rate) * 15.0  # pixels per step
        return move
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        # Auto-paddle: predict where ball will hit bottom and move there
        if self.auto_paddle:
            # Predict ball x position when it reaches paddle height
            paddle_top = self.screen_height - self.paddle_height
            if self.ball_vy > 0:  # Ball moving down
                time_to_paddle = (paddle_top - self.ball_y) / self.ball_vy
                predicted_x = self.ball_x + self.ball_vx * time_to_paddle
                
                # Account for wall bounces
                while predicted_x < 0 or predicted_x > self.screen_width:
                    if predicted_x < 0:
                        predicted_x = -predicted_x
                    elif predicted_x > self.screen_width:
                        predicted_x = 2 * self.screen_width - predicted_x
                
                # Move paddle towards predicted position
                target_x = np.clip(predicted_x, 
                                   self.paddle_width // 2, 
                                   self.screen_width - self.paddle_width // 2)
                move = (target_x - self.paddle_x) * 1.0  # Instant movement
            else:
                move = 0
        else:
            # Decode action from network
            move = self._decode_action(action)
        
        self.paddle_x = np.clip(self.paddle_x + move, 
                                self.paddle_width // 2, 
                                self.screen_width - self.paddle_width // 2)
        
        # Update ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Wall collisions (left/right) - add slight angle variation
        if self.ball_x - self.ball_radius <= 0:
            self.ball_vx *= -1
            self.ball_x = self.ball_radius
            # Add small random angle variation on wall hit
            angle_var = np.random.uniform(-0.15, 0.15)  # ~±8.5 degrees
            speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
            angle = np.arctan2(self.ball_vy, self.ball_vx) + angle_var
            self.ball_vx = speed * np.cos(angle)
            self.ball_vy = speed * np.sin(angle)
        elif self.ball_x + self.ball_radius >= self.screen_width:
            self.ball_vx *= -1
            self.ball_x = self.screen_width - self.ball_radius
            # Add small random angle variation on wall hit
            angle_var = np.random.uniform(-0.15, 0.15)
            speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
            angle = np.arctan2(self.ball_vy, self.ball_vx) + angle_var
            self.ball_vx = speed * np.cos(angle)
            self.ball_vy = speed * np.sin(angle)
        
        # Top wall - add slight angle variation
        if self.ball_y - self.ball_radius <= 0:
            self.ball_vy *= -1
            self.ball_y = self.ball_radius
            angle_var = np.random.uniform(-0.15, 0.15)
            speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
            angle = np.arctan2(self.ball_vy, self.ball_vx) + angle_var
            self.ball_vx = speed * np.cos(angle)
            self.ball_vy = speed * np.sin(angle)
        
        # Paddle collision (bottom)
        paddle_left = self.paddle_x - self.paddle_width // 2
        paddle_right = self.paddle_x + self.paddle_width // 2
        paddle_top = self.screen_height - self.paddle_height
        
        if (self.ball_y + self.ball_radius >= paddle_top and
            paddle_left <= self.ball_x <= paddle_right):
            self.ball_vy *= -1  # Reverse direction, keep speed constant
            self.ball_y = paddle_top - self.ball_radius
            
            # Angle based on where ball hits paddle
            hit_pos = (self.ball_x - self.paddle_x) / (self.paddle_width / 2)
            # Set velocity magnitude to ball_speed, direction based on hit position
            # hit_pos in [-1, 1], map to angle in [-80°, 80°] for more horizontal movement
            angle = hit_pos * np.radians(80)
            self.ball_vx = self.ball_speed * np.sin(angle)
            self.ball_vy = -self.ball_speed * np.cos(angle)
            
            self.score += 1
            self.hits += 1
            reward = 1.0
        else:
            reward = 0.01  # Survival reward
        
        # Ball fell off bottom (game over)
        terminated = False
        if self.ball_y - self.ball_radius > self.screen_height:
            terminated = True
            reward = -1.0
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        obs = self._encode_observation()
        info = {
            "score": self.score, 
            "hits": self.hits,
            "ball_pos": (self.ball_x, self.ball_y),
            "paddle_x": self.paddle_x,
            "ball_vel": (self.ball_vx, self.ball_vy)
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.paddle_x = self.screen_width // 2
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        self.ball_vx = self.ball_speed * np.random.choice([-1, 1])
        self.ball_vy = -self.ball_speed * np.random.uniform(0.5, 1.5)
        self.step_count = 0
        self.score = 0
        self.hits = 0
        self.prev_ball_pos = None
        return self._encode_observation(), {}
    
    def render(self):
        """Render game frame as RGB array."""
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Draw ball
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), 
                   self.ball_radius, (255, 255, 255), -1)
        
        # Draw paddle
        paddle_left = int(self.paddle_x - self.paddle_width // 2)
        paddle_right = int(self.paddle_x + self.paddle_width // 2)
        paddle_top = self.screen_height - self.paddle_height
        cv2.rectangle(frame, (paddle_left, paddle_top), 
                      (paddle_right, self.screen_height), (0, 255, 0), -1)
        
        # Draw score
        cv2.putText(frame, f"Score: {self.score}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Hits: {self.hits}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.render_mode == "human":
            # Scale up for display
            display = cv2.resize(frame, (self.screen_width * 3, self.screen_height * 3),
                               interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Pinball", display)
            cv2.waitKey(1)
        
        return frame


