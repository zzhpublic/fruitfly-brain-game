"""Maze navigation environment."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List

class MazeEnv(gym.Env):
    """
    Maze navigation for central complex / mushroom body.
    
    Observations: Visual landmarks (optic lobes) + goal direction (central complex)
    Actions: Motor commands (descending neurons)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self,
                 maze_size: int = 15,
                 n_landmarks: int = 4,
                 max_steps: int = 2000,
                 dt: float = 0.1):
        super().__init__()
        
        self.maze_size = maze_size
        self.n_landmarks = n_landmarks
        self.max_steps = max_steps
        self.dt = dt
        
        # Maze: 0 = free, 1 = wall
        self.maze = self._generate_maze()
        
        # Landmarks at fixed positions
        self.landmarks = self._place_landmarks()
        
        # Agent state
        self.agent_pos = np.array([1.0, 1.0])
        self.agent_dir = 0.0  # radians
        self.goal_pos = np.array([maze_size - 2, maze_size - 2])
        
        # Observation: landmark bearings + goal direction + wall distances
        self.n_visual_neurons = 36  # 10-degree bins
        self.n_compass_neurons = 16  # Head direction cells
        self.observation_space = spaces.Box(
            low=0, high=100, 
            shape=(self.n_visual_neurons + self.n_compass_neurons + 8,), 
            dtype=np.float32
        )
        
        # Action: turn left/right, move forward
        self.n_action_neurons = 12  # 4 turn left, 4 turn right, 4 forward
        self.action_space = spaces.Box(
            low=0, high=50, shape=(self.n_action_neurons,), dtype=np.float32
        )
        
        self.step_count = 0
    
    def _generate_maze(self) -> np.ndarray:
        """Generate random maze using recursive backtracking."""
        maze = np.ones((self.maze_size, self.maze_size), dtype=int)
        
        def carve(x, y):
            maze[y, x] = 0
            dirs = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            np.random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.maze_size - 1 and 0 < ny < self.maze_size - 1:
                    if maze[ny, nx] == 1:
                        maze[y + dy//2, x + dx//2] = 0
                        carve(nx, ny)
        
        carve(1, 1)
        # Ensure goal is reachable
        maze[self.maze_size-2, self.maze_size-2] = 0
        maze[self.maze_size-3, self.maze_size-2] = 0
        maze[self.maze_size-2, self.maze_size-3] = 0
        return maze
    
    def _place_landmarks(self) -> List[np.ndarray]:
        """Place landmarks at corners and center."""
        landmarks = []
        margin = 2
        positions = [
            [margin, margin],
            [self.maze_size - margin - 1, margin],
            [margin, self.maze_size - margin - 1],
            [self.maze_size // 2, self.maze_size // 2],
        ]
        for pos in positions[:self.n_landmarks]:
            if self.maze[pos[1], pos[0]] == 0:
                landmarks.append(np.array(pos, dtype=float))
        return landmarks
    
    def _get_wall_distances(self) -> np.ndarray:
        """Raycast distances to walls in 8 directions."""
        distances = []
        for i in range(8):
            angle = self.agent_dir + i * np.pi / 4
            dx, dy = np.cos(angle), np.sin(angle)
            dist = 0
            x, y = self.agent_pos
            while dist < 10:
                x += dx * 0.1
                y += dy * 0.1
                ix, iy = int(x), int(y)
                if ix < 0 or ix >= self.maze_size or iy < 0 or iy >= self.maze_size:
                    break
                if self.maze[iy, ix] == 1:
                    break
                dist += 0.1
            distances.append(dist)
        return np.array(distances)
    
    def _encode_observation(self) -> np.ndarray:
        """Encode as visual + compass + proximity."""
        # Visual: landmark bearings
        visual = np.zeros(self.n_visual_neurons)
        for lm in self.landmarks:
            vec = lm - self.agent_pos
            bearing = np.arctan2(vec[1], vec[0]) - self.agent_dir
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            idx = int((bearing + np.pi) / (2 * np.pi) * self.n_visual_neurons)
            if 0 <= idx < self.n_visual_neurons:
                dist = np.linalg.norm(vec)
                visual[idx] = np.exp(-dist / 5.0) * 100
        
        # Compass: head direction cells
        compass = np.zeros(self.n_compass_neurons)
        idx = int((self.agent_dir + np.pi) / (2 * np.pi) * self.n_compass_neurons)
        if 0 <= idx < self.n_compass_neurons:
            compass[idx] = 100
        
        # Proximity: wall distances
        proximity = self._get_wall_distances() * 10
        
        # Goal direction
        goal_vec = self.goal_pos - self.agent_pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - self.agent_dir
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi
        goal_encoding = np.array([np.cos(goal_angle), np.sin(goal_angle)]) * 50
        
        obs = np.concatenate([visual, compass, proximity, goal_encoding])
        
        # Poisson spikes
        spike_probs = 1 - np.exp(-obs * self.dt / 1000.0)
        spikes = (np.random.random(len(obs)) < spike_probs).astype(np.float32)
        
        return spikes
    
    def _decode_action(self, action: np.ndarray) -> Tuple[float, float]:
        """Decode to turn rate and forward speed."""
        turn_left = action[:4].sum() / 4.0
        turn_right = action[4:8].sum() / 4.0
        forward = action[8:].sum() / 4.0
        
        turn_rate = (turn_right - turn_left) * 0.5  # rad/step
        speed = forward * 0.5  # units/step
        
        return turn_rate, speed
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        turn_rate, speed = self._decode_action(action)
        
        # Update heading
        self.agent_dir += turn_rate
        self.agent_dir = (self.agent_dir + np.pi) % (2 * np.pi) - np.pi
        
        # Move forward
        new_pos = self.agent_pos + np.array([np.cos(self.agent_dir), np.sin(self.agent_dir)]) * speed
        
        # Check collision
        ix, iy = int(new_pos[0]), int(new_pos[1])
        if (0 <= ix < self.maze_size and 0 <= iy < self.maze_size and 
            self.maze[iy, ix] == 0):
            self.agent_pos = new_pos
        else:
            # Hit wall - small negative reward
            pass
        
        # Check goal
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        terminated = dist_to_goal < 1.0
        
        if terminated:
            reward = 10.0
        else:
            # Reward for getting closer to goal
            reward = -dist_to_goal * 0.01
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        obs = self._encode_observation()
        info = {"pos": self.agent_pos.copy(), "dist_to_goal": dist_to_goal}
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.agent_pos = np.array([1.0, 1.0])
        self.agent_dir = 0.0
        self.step_count = 0
        return self._encode_observation(), {}
    
    def render(self):
        if self.render_mode == "human":
            grid = self.maze.copy()
            ix, iy = int(self.agent_pos[0]), int(self.agent_pos[1])
            if 0 <= ix < self.maze_size and 0 <= iy < self.maze_size:
                grid[iy, ix] = 2
            gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
            grid[gy, gx] = 3
            chars = {0: " ", 1: "█", 2: "●", 3: "★"}
            for row in grid:
                print("".join(chars.get(c, "?") for c in row))
            print(f"Pos: {self.agent_pos}, Dir: {self.agent_dir:.2f}")
