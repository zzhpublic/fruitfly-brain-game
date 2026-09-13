"""Build spiking network from connectome data."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from neurons.lif import LIFNeuron, LIFParams
from synapses.stdp import STDPSynapse, STDPParams
from neuromod.modulator import UnifiedNeuromodulation
from connectome.loader import ConnectomeData, Neuron, Synapse

@dataclass
class NetworkConfig:
    """Configuration for network assembly."""
    dt: float = 0.1  # ms
    regions: List[str] = field(default_factory=lambda: [
        "optic_lobes", "mushroom_body", "central_complex", 
        "lateral_horn", "neuromodulatory"
    ])
    # Neuron model params per region
    neuron_params: Dict[str, LIFParams] = field(default_factory=dict)
    # STDP params per connection type
    stdp_params: Dict[str, STDPParams] = field(default_factory=dict)
    # Connection probability scaling
    conn_scale: float = 1.0
    # Dale's law enforcement
    enforce_dale: bool = True
    # Neuromodulation
    use_neuromodulation: bool = True

class NetworkBuilder:
    """Assemble spiking network from connectome."""
    
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self.neurons: Dict[int, LIFNeuron] = {}
        self.synapses: Dict[Tuple[str, str], STDPSynapse] = {}  # (pre_region, post_region)
        self.neuron_to_region: Dict[int, str] = {}
        self.region_neuron_ids: Dict[str, List[int]] = {r: [] for r in self.config.regions}
        self.neuromod = UnifiedNeuromodulation(self.config.dt) if self.config.use_neuromodulation else None
    
    def build_from_connectome(self, connectome: ConnectomeData):
        """Build network from connectome data."""
        # Create neurons
        for neuron in connectome.neurons.values():
            if neuron.region not in self.config.regions:
                continue
            
            # Get region-specific params or use defaults
            params = self.config.neuron_params.get(neuron.region, LIFParams())
            
            # Adjust for neurotransmitter (Dale's law)
            if self.config.enforce_dale:
                if neuron.neurotransmitter == "GABA":
                    params.E_rev = -80.0  # Inhibitory
                elif neuron.neurotransmitter == "glutamate":
                    params.E_rev = 0.0    # Excitatory
                # Acetylcholine: mixed, default -70
            
            lif = LIFNeuron(params, self.config.dt)
            self.neurons[neuron.id] = lif
            self.neuron_to_region[neuron.id] = neuron.region
            self.region_neuron_ids[neuron.region].append(neuron.id)
        
        # Create synapses between regions
        for pre_region in self.config.regions:
            for post_region in self.config.regions:
                pre_ids = self.region_neuron_ids[pre_region]
                post_ids = self.region_neuron_ids[post_region]
                
                if not pre_ids or not post_ids:
                    continue
                
                # Filter synapses from connectome
                region_synapses = [
                    s for s in connectome.synapses
                    if s.pre_id in pre_ids and s.post_id in post_ids
                ]
                
                if not region_synapses:
                    continue
                
                # Create STDP synapse group
                n_pre = len(pre_ids)
                n_post = len(post_ids)
                
                # Map global IDs to local indices
                pre_id_to_idx = {nid: i for i, nid in enumerate(pre_ids)}
                post_id_to_idx = {nid: i for i, nid in enumerate(post_ids)}
                
                stdp_params = self.config.stdp_params.get(
                    f"{pre_region}->{post_region}", 
                    STDPParams()
                )
                
                synapse = STDPSynapse(n_pre, n_post, stdp_params, self.config.dt)
                
                # Set weights from connectome
                for s in region_synapses:
                    if s.pre_id in pre_id_to_idx and s.post_id in post_id_to_idx:
                        pre_idx = pre_id_to_idx[s.pre_id]
                        post_idx = post_id_to_idx[s.post_id]
                        # Find connection in sparse matrix
                        mask = (synapse.pre_indices == pre_idx) & (synapse.post_indices == post_idx)
                        if np.any(mask):
                            synapse.weights[mask] = s.size
                
                self.synapses[(pre_region, post_region)] = synapse
        
        return self
    
    def get_neuron_ids(self, region: str) -> List[int]:
        return self.region_neuron_ids.get(region, [])
    
    def get_synapse_group(self, pre_region: str, post_region: str) -> Optional[STDPSynapse]:
        return self.synapses.get((pre_region, post_region))
    
    def step(self, external_input: Dict[str, np.ndarray] = None,
             rewards: Dict = None, punishments: Dict = None,
             locomotion: float = 0.0, starvation: float = 0.0,
             stress: float = 0.0, arousal: float = 0.0,
             feeding: float = 0.0, circadian_time: float = None):
        """
        Single simulation step.
        
        Args:
            external_input: Dict of region -> input current array
            rewards/punishments: For dopamine
            locomotion/starvation/stress/arousal: For octopamine
            feeding/circadian_time: For serotonin
        """
        # Update neuromodulation
        if self.neuromod:
            self.neuromod.step(
                rewards=rewards, punishments=punishments,
                locomotion=locomotion, starvation=starvation,
                stress=stress, arousal=arousal,
                feeding=feeding, circadian_time=circadian_time
            )
        
        # Collect spikes from all neurons
        all_spikes = {}
        for region in self.config.regions:
            n_ids = self.region_neuron_ids[region]
            if not n_ids:
                all_spikes[region] = np.array([], dtype=bool)
                continue
            
            # Get external input for this region
            I_ext = np.zeros(len(n_ids))
            if external_input and region in external_input:
                I_ext = external_input[region]
            
            # Get neuromodulator concentrations
            da_conc = 0.0
            if self.neuromod:
                da_conc = self.neuromod.dopamine.get_region_concentration(region)
            
            # Step each neuron
            spikes = np.zeros(len(n_ids), dtype=bool)
            for i, nid in enumerate(n_ids):
                neuron = self.neurons[nid]
                # Add synaptic input
                I_syn = 0.0
                for (pre_r, post_r), syn in self.synapses.items():
                    if post_r == region:
                        pre_spikes = all_spikes.get(pre_r, np.array([], dtype=bool))
                        if len(pre_spikes) > 0:
                            I_syn += syn.compute_current(pre_spikes)[i]
                
                total_I = I_ext[i] + I_syn
                spiked = neuron.step(total_I, da_conc)
                spikes[i] = spiked
            
            all_spikes[region] = spikes
        
        # Update STDP
        for (pre_r, post_r), syn in self.synapses.items():
            pre_spikes = all_spikes.get(pre_r, np.array([], dtype=bool))
            post_spikes = all_spikes.get(post_r, np.array([], dtype=bool))
            
            da = None
            if self.neuromod:
                da = self.neuromod.dopamine.get_region_concentration(post_r)
            
            syn.step(pre_spikes, post_spikes, da)
        
        return all_spikes
    
    def get_state(self) -> Dict:
        """Get network state for monitoring."""
        state = {
            "neurons": {},
            "synapses": {},
            "neuromodulation": {}
        }
        
        for region in self.config.regions:
            n_ids = self.region_neuron_ids[region]
            if n_ids:
                state["neurons"][region] = {
                    "v": np.array([self.neurons[nid].V for nid in n_ids]),
                    "refractory": np.array([self.neurons[nid].refractory for nid in n_ids]),
                }
        
        for (pre_r, post_r), syn in self.synapses.items():
            state["synapses"][f"{pre_r}->{post_r}"] = {
                "weights": syn.weights.copy(),
                "mean_weight": float(np.mean(syn.weights)),
            }
        
        if self.neuromod:
            state["neuromodulation"] = self.neuromod.get_all_concentrations()
        
        return state
