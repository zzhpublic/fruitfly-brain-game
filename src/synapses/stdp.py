"""Spike-Timing-Dependent Plasticity (STDP) implementation."""
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class STDPParams:
    """STDP parameters for Drosophila synapses."""
    tau_pre: float = 20.0      # ms, pre-synaptic trace decay
    tau_post: float = 20.0     # ms, post-synaptic trace decay
    A_pre: float = 0.01        # LTP amplitude
    A_post: float = -0.012     # LTD amplitude (negative)
    w_min: float = 0.0         # Minimum weight
    w_max: float = 1.0         # Maximum weight
    w_init_mean: float = 0.5   # Initial weight mean
    w_init_std: float = 0.1    # Initial weight std

class STDPSynapse:
    """STDP synapse with eligibility traces."""
    
    def __init__(self, n_pre: int, n_post: int, params: Optional[STDPParams] = None, 
                 dt: float = 0.1, connectivity: float = 0.1,
                 pre_indices: Optional[np.ndarray] = None,
                 post_indices: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None):
        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or STDPParams()
        self.dt = dt
        
        # Eligibility traces
        self.pre_trace = np.zeros(n_pre, dtype=np.float32)
        self.post_trace = np.zeros(n_post, dtype=np.float32)
        
        # If connectome data provided, use it; otherwise generate random sparse connections
        if pre_indices is not None and post_indices is not None and weights is not None:
            self.pre_indices = pre_indices
            self.post_indices = post_indices
            self.weights = weights
        else:
            # Weight matrix (sparse)
            self.connectivity = connectivity
            n_connections = int(n_pre * n_post * connectivity)
            
            # Sparse representation
            self.pre_indices = np.random.randint(0, n_pre, n_connections)
            self.post_indices = np.random.randint(0, n_post, n_connections)
            self.weights = np.random.normal(
                self.params.w_init_mean, 
                self.params.w_init_std, 
                n_connections
            ).astype(np.float32)
            self.weights = np.clip(self.weights, self.params.w_min, self.params.w_max)
        
    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
             dopamine: Optional[np.ndarray] = None):
        """
        Update STDP traces and weights.
        
        Args:
            pre_spikes: Boolean array of presynaptic spikes
            post_spikes: Boolean array of postsynaptic spikes
            dopamine: Optional dopamine signal for gated STDP
        """
        # Update traces
        self.pre_trace *= np.exp(-self.dt / self.params.tau_pre)
        self.post_trace *= np.exp(-self.dt / self.params.tau_post)
        
        self.pre_trace[pre_spikes] += 1.0
        self.post_trace[post_spikes] += 1.0
        
        # Weight updates (vectorized for sparse connections)
        # LTP: pre before post
        pre_active = self.pre_trace[self.pre_indices] > 0
        post_active = self.post_trace[self.post_indices] > 0
        
        # Standard STDP
        dw = np.zeros_like(self.weights)
        
        # Pre before post -> LTP
        ltp_mask = pre_active & post_spikes[self.post_indices]
        dw[ltp_mask] += self.params.A_pre * self.post_trace[self.post_indices[ltp_mask]]
        
        # Post before pre -> LTD
        ltd_mask = post_active & pre_spikes[self.pre_indices]
        dw[ltd_mask] += self.params.A_post * self.pre_trace[self.pre_indices[ltd_mask]]
        
        # Dopamine gating (three-factor rule)
        if dopamine is not None:
            # Modulate by postsynaptic dopamine
            if np.isscalar(dopamine):
                da = dopamine
            else:
                da = dopamine[self.post_indices]
            dw *= (1.0 + da)  # Dopamine enhances LTP, suppresses LTD
        
        # Apply updates
        self.weights += dw
        self.weights = np.clip(self.weights, self.params.w_min, self.params.w_max)
    
    def get_dense_weights(self) -> np.ndarray:
        """Convert to dense weight matrix (for visualization)."""
        W = np.zeros((self.n_post, self.n_pre), dtype=np.float32)
        W[self.post_indices, self.pre_indices] = self.weights
        return W
    
    def compute_current(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Compute postsynaptic current from presynaptic spikes."""
        I_post = np.zeros(self.n_post, dtype=np.float32)
        if np.any(pre_spikes):
            active_pre = np.where(pre_spikes)[0]
            for pre_idx in active_pre:
                mask = self.pre_indices == pre_idx
                I_post[self.post_indices[mask]] += self.weights[mask]
        return I_post
