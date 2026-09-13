"""Dopamine system for reward-based learning."""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class DopamineParams:
    """Dopamine system parameters."""
    baseline: float = 0.1
    reward_scale: float = 1.0
    punishment_scale: float = -0.5
    decay_tau: float = 500.0  # ms
    diffusion_tau: float = 100.0  # ms, volume transmission
    n_clusters: int = 4  # PAM, PPL1, PPL2, PPL3
    neurons_per_cluster: int = 50

class DopamineSystem:
    """
    Drosophila dopamine system with multiple clusters.
    
    Clusters:
    - PAM: Reward (sugar, water, mating)
    - PPL1: Punishment (shock, bitter, heat)
    - PPL2: Novelty, arousal
    - PPL3: Sleep/wake regulation
    """
    
    def __init__(self, params: Optional[DopamineParams] = None, dt: float = 0.1):
        self.params = params or DopamineParams()
        self.dt = dt
        
        # Cluster activities
        self.cluster_activity = np.zeros(self.params.n_clusters, dtype=np.float32)
        self.cluster_baseline = np.full(self.params.n_clusters, self.params.baseline, dtype=np.float32)
        
        # Volume transmission (diffusion to target regions)
        self.target_regions = [
            "mushroom_body", "lateral_horn", "central_complex", 
            "optic_lobes", "descending_neurons"
        ]
        self.region_concentration = {r: 0.0 for r in self.target_regions}
        
        # Projection weights from clusters to regions
        self.projection_weights = np.random.uniform(0.1, 1.0, 
            (self.params.n_clusters, len(self.target_regions))).astype(np.float32)
        # Normalize
        self.projection_weights /= self.projection_weights.sum(axis=0, keepdims=True)
        
    def step(self, rewards: Dict[str, float], punishments: Dict[str, float] = None):
        """
        Update dopamine based on rewards/punishments.
        
        Args:
            rewards: Dict of reward signals (e.g., {"sugar": 1.0, "water": 0.5})
            punishments: Dict of punishment signals
        """
        # Decay cluster activity
        self.cluster_activity *= np.exp(-self.dt / self.params.decay_tau)
        
        # Process rewards (PAM cluster = index 0)
        total_reward = sum(rewards.values()) if rewards else 0.0
        if total_reward > 0:
            self.cluster_activity[0] += total_reward * self.params.reward_scale
        
        # Process punishments (PPL1 cluster = index 1)
        total_punishment = sum(punishments.values()) if punishments else 0.0
        if total_punishment > 0:
            self.cluster_activity[1] += total_punishment * self.params.punishment_scale
        
        # Add baseline
        self.cluster_activity += self.cluster_baseline * self.dt
        
        # Volume transmission to target regions
        for i, region in enumerate(self.target_regions):
            # Weighted sum of cluster activities
            conc = np.sum(self.cluster_activity * self.projection_weights[:, i])
            # Diffusion dynamics
            self.region_concentration[region] = (
                self.region_concentration[region] * np.exp(-self.dt / self.params.diffusion_tau)
                + conc * (1 - np.exp(-self.dt / self.params.diffusion_tau))
            )
    
    def get_region_concentration(self, region: str) -> float:
        """Get dopamine concentration in a target region."""
        return self.region_concentration.get(region, 0.0)
    
    def get_all_concentrations(self) -> np.ndarray:
        """Get concentrations for all target regions."""
        return np.array([self.region_concentration[r] for r in self.target_regions])
    
    def reset(self):
        """Reset dopamine system."""
        self.cluster_activity.fill(0.0)
        for r in self.target_regions:
            self.region_concentration[r] = 0.0
