"""Leaky Integrate-and-Fire (LIF) neuron model with conductance-based synapses."""
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class LIFParams:
    """LIF neuron parameters (biologically plausible for Drosophila)."""
    C_m: float = 100.0       # pF, membrane capacitance
    g_L: float = 10.0        # nS, leak conductance
    E_L: float = -60.0       # mV, leak reversal potential
    V_th: float = -40.0      # mV, spike threshold
    V_reset: float = -60.0   # mV, reset potential
    t_ref: float = 2.0       # ms, refractory period
    # Synaptic parameters
    E_rev: float = -70.0     # mV, synaptic reversal (E_exc=0, E_inh=-80)
    g_syn_max: float = 1.0   # nS, max synaptic conductance
    tau_syn: float = 5.0     # ms, synaptic time constant
    # Neuromodulation
    da_gain: float = 0.5     # Dopamine modulation of threshold
    
    @property
    def tau_m(self) -> float:
        return self.C_m / self.g_L  # ms
    
    @property
    def R_m(self) -> float:
        return 1.0 / self.g_L  # GΩ

class LIFNeuron:
    """Conductance-based LIF neuron with Dale's law compliance."""
    
    def __init__(self, params: LIFParams, dt: float = 0.1):
        self.params = params
        self.dt = dt
        
        # State variables
        self.V = params.E_L
        self.refractory = 0.0
        self.g_syn = 0.0
        self.spike_history = []
        
    def step(self, I_ext: float, da_conc: float = 0.0) -> bool:
        """
        Single simulation step.
        
        Args:
            I_ext: External current (pA)
            da_conc: Dopamine concentration (modulates threshold)
            
        Returns:
            True if spike occurred
        """
        # Refractory period
        if self.refractory > 0:
            self.refractory -= self.dt
            self.V = self.params.V_reset
            self.g_syn *= np.exp(-self.dt / self.params.tau_syn)
            return False
        
        # Synaptic conductance decay
        self.g_syn *= np.exp(-self.dt / self.params.tau_syn)
        
        # Dopamine modulation of threshold
        V_th_mod = self.params.V_th - self.params.da_gain * da_conc * 5.0
        
        # Membrane potential update (conductance-based)
        # C dV/dt = -g_L(V - E_L) - g_syn(V - E_rev) + I_ext
        I_syn = self.g_syn * (self.V - self.params.E_rev)
        dV = (-self.params.g_L * (self.V - self.params.E_L) - I_syn + I_ext) / self.params.C_m
        self.V += dV * self.dt
        
        # Spike detection
        if self.V >= V_th_mod:
            self.V = self.params.V_reset
            self.refractory = self.params.t_ref
            self.spike_history.append(1)
            return True
        
        self.spike_history.append(0)
        return False
    
    def inject_synaptic(self, g_syn: float):
        """Inject synaptic conductance."""
        self.g_syn += g_syn
    
    def reset(self):
        """Reset neuron state."""
        self.V = self.params.E_L
        self.refractory = 0.0
        self.g_syn = 0.0
        self.spike_history = []


class LIFPopulation:
    """Vectorized population of LIF neurons with shared parameters."""
    
    def __init__(self, n_neurons: int, params: LIFParams, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.params = params
        self.dt = dt
        
        # State variables (arrays)
        self.V = np.full(n_neurons, params.E_L, dtype=np.float32)
        self.refractory = np.zeros(n_neurons, dtype=np.float32)
        self.g_syn = np.zeros(n_neurons, dtype=np.float32)
        
    def step(self, I_ext: np.ndarray, da_conc: float = 0.0) -> np.ndarray:
        """
        Vectorized step for all neurons.
        
        Args:
            I_ext: Array of external currents (pA), shape (n_neurons,)
            da_conc: Dopamine concentration (modulates threshold)
            
        Returns:
            Boolean array of spikes, shape (n_neurons,)
        """
        # Refractory period
        in_refractory = self.refractory > 0
        self.refractory[in_refractory] -= self.dt
        self.V[in_refractory] = self.params.V_reset
        self.g_syn[in_refractory] *= np.exp(-self.dt / self.params.tau_syn)
        
        # Synaptic conductance decay (for non-refractory)
        not_refractory = ~in_refractory
        self.g_syn[not_refractory] *= np.exp(-self.dt / self.params.tau_syn)
        
        # Dopamine modulation of threshold
        V_th_mod = self.params.V_th - self.params.da_gain * da_conc * 5.0
        
        # Membrane potential update (conductance-based)
        # C dV/dt = -g_L(V - E_L) - g_syn(V - E_rev) + I_ext
        I_syn = self.g_syn * (self.V - self.params.E_rev)
        dV = (-self.params.g_L * (self.V - self.params.E_L) - I_syn + I_ext) / self.params.C_m
        self.V += dV * self.dt
        
        # Spike detection
        spiked = (self.V >= V_th_mod) & not_refractory
        
        # Reset spiked neurons
        self.V[spiked] = self.params.V_reset
        self.refractory[spiked] = self.params.t_ref
        
        return spiked
    
    def inject_synaptic(self, g_syn: np.ndarray):
        """Inject synaptic conductance."""
        self.g_syn += g_syn
    
    def reset(self):
        """Reset population state."""
        self.V.fill(self.params.E_L)
        self.refractory.fill(0.0)
        self.g_syn.fill(0.0)
