"""Serotonin system for mood, feeding, and circadian rhythms."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class SerotoninParams:
    """Serotonin system parameters."""
    baseline: float = 0.08
    feeding_scale: float = 0.4
    circadian_amplitude: float = 0.3
    stress_scale: float = -0.2
    decay_tau: float = 2000.0  # ms
    n_neurons: int = 80  # CSD, LP1, LP2, SE0, SE1, SE2, SE3 clusters

class SerotoninSystem:
    """
    Drosophila serotonin system.
    
    Functions:
    - Feeding behavior modulation
    - Circadian rhythm regulation
    - Stress response (inhibitory)
    - Sleep/wake cycles
    - Aggression suppression
    """
    
    def __init__(self, params: Optional[SerotoninParams] = None, dt: float = 0.1):
        self.params = params or SerotoninParams()
        self.dt = dt
        
        self.activity = np.zeros(self.params.n_neurons, dtype=np.float32)
        self.baseline = np.full(self.params.n_neurons, self.params.baseline, dtype=np.float32)
        
        # Circadian phase (0-2π)
        self.circadian_phase = 0.0
        
        # Target regions
        self.target_regions = [
            "mushroom_body", "lateral_horn", "central_complex",
            "optic_lobes", "descending_neurons", "neuroendocrine",
            "enteric_nervous_system"
        ]
        self.region_concentration = {r: 0.0 for r in self.target_regions}
        
        # Projection weights
        self.projection_weights = np.random.uniform(0.1, 1.0,
            (self.params.n_neurons, len(self.target_regions))).astype(np.float32)
        self.projection_weights /= self.projection_weights.sum(axis=0, keepdims=True)
    
    def step(self, feeding: float = 0.0, stress: float = 0.0, 
             circadian_time: float = None):
        """Update serotonin based on state."""
        # Decay
        self.activity *= np.exp(-self.dt / self.params.decay_tau)
        
        # Circadian rhythm
        if circadian_time is not None:
            self.circadian_phase = 2 * np.pi * (circadian_time % 24) / 24
        else:
            self.circadian_phase += 2 * np.pi * self.dt / (24 * 3600 * 1000)  # 24h in ms
        
        circadian_mod = self.params.circadian_amplitude * np.sin(self.circadian_phase)
        
        # Modulatory inputs
        self.activity[:20] += feeding * self.params.feeding_scale  # CSD
        self.activity[20:40] += circadian_mod  # LP1/LP2
        self.activity[40:] += stress * self.params.stress_scale  # SE clusters
        
        # Baseline
        self.activity += self.baseline * self.dt
        
        # Volume transmission
        for i, region in enumerate(self.target_regions):
            conc = np.sum(self.activity * self.projection_weights[:, i])
            self.region_concentration[region] = (
                self.region_concentration[region] * 0.995
                + conc * 0.005
            )
    
    def get_region_concentration(self, region: str) -> float:
        return self.region_concentration.get(region, 0.0)
    
    def reset(self):
        self.activity.fill(0.0)
        for r in self.target_regions:
            self.region_concentration[r] = 0.0
