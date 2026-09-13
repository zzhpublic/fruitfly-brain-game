"""Octopamine system for arousal, locomotion, and metabolic state."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class OctopamineParams:
    """Octopamine system parameters."""
    baseline: float = 0.05
    locomotion_scale: float = 0.5
    starvation_scale: float = 0.3
    stress_scale: float = 0.4
    decay_tau: float = 1000.0  # ms
    n_neurons: int = 100  # VUM, VPM, OA-VL, OA-VPM clusters

class OctopamineSystem:
    """
    Drosophila octopamine system.
    
    Functions:
    - Arousal and wakefulness
    - Locomotion modulation
    - Metabolic state (starvation)
    - Stress response
    - Aggression
    """
    
    def __init__(self, params: Optional[OctopamineParams] = None, dt: float = 0.1):
        self.params = params or OctopamineParams()
        self.dt = dt
        
        self.activity = np.zeros(self.params.n_neurons, dtype=np.float32)
        self.baseline = np.full(self.params.n_neurons, self.params.baseline, dtype=np.float32)
        
        # Target regions
        self.target_regions = [
            "mushroom_body", "central_complex", "optic_lobes",
            "lateral_horn", "descending_neurons", "neuroendocrine"
        ]
        self.region_concentration = {r: 0.0 for r in self.target_regions}
        
        # Projection weights
        self.projection_weights = np.random.uniform(0.1, 1.0,
            (self.params.n_neurons, len(self.target_regions))).astype(np.float32)
        self.projection_weights /= self.projection_weights.sum(axis=0, keepdims=True)
    
    def step(self, locomotion: float = 0.0, starvation: float = 0.0, 
             stress: float = 0.0, arousal: float = 0.0):
        """Update octopamine based on behavioral state."""
        # Decay
        self.activity *= np.exp(-self.dt / self.params.decay_tau)
        
        # Modulatory inputs
        self.activity[:20] += locomotion * self.params.locomotion_scale  # VUM
        self.activity[20:40] += starvation * self.params.starvation_scale  # VPM
        self.activity[40:60] += stress * self.params.stress_scale  # OA-VL
        self.activity[60:] += arousal * 0.3  # OA-VPM
        
        # Baseline
        self.activity += self.baseline * self.dt
        
        # Volume transmission
        for i, region in enumerate(self.target_regions):
            conc = np.sum(self.activity * self.projection_weights[:, i])
            self.region_concentration[region] = (
                self.region_concentration[region] * 0.99
                + conc * 0.01
            )
    
    def get_region_concentration(self, region: str) -> float:
        return self.region_concentration.get(region, 0.0)
    
    def reset(self):
        self.activity.fill(0.0)
        for r in self.target_regions:
            self.region_concentration[r] = 0.0
