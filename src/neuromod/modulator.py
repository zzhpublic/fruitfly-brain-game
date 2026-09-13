"""Base neuromodulator class and unified interface."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

@dataclass
class ModulatorParams:
    """Base parameters for neuromodulators."""
    name: str
    baseline: float = 0.1
    decay_tau: float = 1000.0
    target_regions: List[str] = None

class Neuromodulator(ABC):
    """Abstract base class for neuromodulatory systems."""
    
    def __init__(self, params: ModulatorParams, dt: float = 0.1):
        self.params = params
        self.dt = dt
        self.target_regions = params.target_regions or []
        self.region_concentration = {r: 0.0 for r in self.target_regions}
    
    @abstractmethod
    def step(self, **kwargs):
        """Update modulator state."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset modulator state."""
        pass
    
    def get_region_concentration(self, region: str) -> float:
        return self.region_concentration.get(region, 0.0)
    
    def get_all_concentrations(self) -> Dict[str, float]:
        return self.region_concentration.copy()

class UnifiedNeuromodulation:
    """Unified interface for all neuromodulatory systems."""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        from .dopamine import DopamineSystem, DopamineParams
        from .octopamine import OctopamineSystem, OctopamineParams
        from .serotonin import SerotoninSystem, SerotoninParams
        
        self.dopamine = DopamineSystem(DopamineParams(), dt)
        self.octopamine = OctopamineSystem(OctopamineParams(), dt)
        self.serotonin = SerotoninSystem(SerotoninParams(), dt)
        
        # All target regions
        self.all_regions = list(set(
            list(self.dopamine.target_regions) +
            list(self.octopamine.target_regions) +
            list(self.serotonin.target_regions)
        ))
    
    def step(self, rewards: Dict = None, punishments: Dict = None,
             locomotion: float = 0.0, starvation: float = 0.0,
             stress: float = 0.0, arousal: float = 0.0,
             feeding: float = 0.0, circadian_time: float = None):
        """Update all neuromodulatory systems."""
        self.dopamine.step(rewards or {}, punishments or {})
        self.octopamine.step(locomotion, starvation, stress, arousal)
        self.serotonin.step(feeding, stress, circadian_time)
    
    def get_concentrations(self, region: str) -> Dict[str, float]:
        """Get all modulator concentrations for a region."""
        return {
            "dopamine": self.dopamine.get_region_concentration(region),
            "octopamine": self.octopamine.get_region_concentration(region),
            "serotonin": self.serotonin.get_region_concentration(region),
        }
    
    def get_all_concentrations(self) -> Dict[str, Dict[str, float]]:
        """Get concentrations for all regions."""
        return {r: self.get_concentrations(r) for r in self.all_regions}
    
    def reset(self):
        self.dopamine.reset()
        self.octopamine.reset()
        self.serotonin.reset()
