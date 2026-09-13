"""Connectome data structures and loader."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml

@dataclass
class Neuron:
    """Single neuron from connectome."""
    id: int
    region: str
    cell_type: str
    hemisphere: str  # "left", "right", "midline"
    x: float
    y: float
    z: float
    n_pre: int = 0
    n_post: int = 0
    neurotransmitter: str = "unknown"  # "acetylcholine", "GABA", "glutamate", "dopamine", etc.

@dataclass
class Synapse:
    """Single synapse from connectome."""
    pre_id: int
    post_id: int
    x: float
    y: float
    z: float
    size: float = 1.0  # T-bar/PSD size proxy for strength
    synapse_type: str = "chemical"  # "chemical", "electrical"

@dataclass
class ConnectomeData:
    """Full connectome dataset."""
    neurons: Dict[int, Neuron] = field(default_factory=dict)
    synapses: List[Synapse] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def get_neurons_by_region(self, region: str) -> List[Neuron]:
        return [n for n in self.neurons.values() if n.region == region]
    
    def get_neurons_by_type(self, cell_type: str) -> List[Neuron]:
        return [n for n in self.neurons.values() if n.cell_type == cell_type]
    
    def get_synapses_for_neuron(self, neuron_id: int, direction: str = "out") -> List[Synapse]:
        if direction == "out":
            return [s for s in self.synapses if s.pre_id == neuron_id]
        else:
            return [s for s in self.synapses if s.post_id == neuron_id]
    
    def get_adjacency_matrix(self, region: str = None, sparse: bool = True):
        """Get connectivity matrix for a region or full brain."""
        if region:
            neuron_ids = [n.id for n in self.get_neurons_by_region(region)]
        else:
            neuron_ids = list(self.neurons.keys())
        
        id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        n = len(neuron_ids)
        
        if sparse:
            from scipy import sparse
            rows, cols, data = [], [], []
            for s in self.synapses:
                if s.pre_id in id_to_idx and s.post_id in id_to_idx:
                    rows.append(id_to_idx[s.post_id])
                    cols.append(id_to_idx[s.pre_id])
                    data.append(s.size)
            return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            W = np.zeros((n, n), dtype=np.float32)
            for s in self.synapses:
                if s.pre_id in id_to_idx and s.post_id in id_to_idx:
                    W[id_to_idx[s.post_id], id_to_idx[s.pre_id]] += s.size
            return W

class ConnectomeLoader:
    """Load connectome from various formats."""
    
    def __init__(self, config_path: str = "config/connectome.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def load_flywire_hdf5(self, filepath: str) -> ConnectomeData:
        """Load FlyWire connectome from HDF5."""
        import h5py
        data = ConnectomeData()
        
        with h5py.File(filepath, 'r') as f:
            # Neurons
            if 'neurons' in f:
                for key in f['neurons'].keys():
                    grp = f['neurons'][key]
                    neuron = Neuron(
                        id=int(key),
                        region=grp.attrs.get('region', 'unknown'),
                        cell_type=grp.attrs.get('cell_type', 'unknown'),
                        hemisphere=grp.attrs.get('hemisphere', 'unknown'),
                        x=grp.attrs.get('x', 0.0),
                        y=grp.attrs.get('y', 0.0),
                        z=grp.attrs.get('z', 0.0),
                        n_pre=grp.attrs.get('n_pre', 0),
                        n_post=grp.attrs.get('n_post', 0),
                        neurotransmitter=grp.attrs.get('neurotransmitter', 'unknown')
                    )
                    data.neurons[neuron.id] = neuron
            
            # Synapses
            if 'synapses' in f:
                for key in f['synapses'].keys():
                    grp = f['synapses'][key]
                    syn = Synapse(
                        pre_id=grp.attrs['pre_id'],
                        post_id=grp.attrs['post_id'],
                        x=grp.attrs.get('x', 0.0),
                        y=grp.attrs.get('y', 0.0),
                        z=grp.attrs.get('z', 0.0),
                        size=grp.attrs.get('size', 1.0),
                        synapse_type=grp.attrs.get('type', 'chemical')
                    )
                    data.synapses.append(syn)
            
            # Metadata
            data.metadata = dict(f.attrs)
            data.regions = list(set(n.region for n in data.neurons.values()))
            data.cell_types = list(set(n.cell_type for n in data.neurons.values()))
        
        return data
    
    def load_fafb_zarr(self, filepath: str) -> ConnectomeData:
        """Load FAFB connectome from Zarr."""
        import zarr
        data = ConnectomeData()
        
        root = zarr.open(filepath, mode='r')
        
        # Neurons table
        if 'neurons' in root:
            neurons_arr = root['neurons']
            for i in range(len(neurons_arr)):
                row = neurons_arr[i]
                neuron = Neuron(
                    id=int(row['id']),
                    region=str(row['region']),
                    cell_type=str(row['cell_type']),
                    hemisphere=str(row['hemisphere']),
                    x=float(row['x']),
                    y=float(row['y']),
                    z=float(row['z']),
                    n_pre=int(row.get('n_pre', 0)),
                    n_post=int(row.get('n_post', 0)),
                    neurotransmitter=str(row.get('neurotransmitter', 'unknown'))
                )
                data.neurons[neuron.id] = neuron
        
        # Synapses table
        if 'synapses' in root:
            synapses_arr = root['synapses']
            for i in range(len(synapses_arr)):
                row = synapses_arr[i]
                syn = Synapse(
                    pre_id=int(row['pre_id']),
                    post_id=int(row['post_id']),
                    x=float(row['x']),
                    y=float(row['y']),
                    z=float(row['z']),
                    size=float(row.get('size', 1.0)),
                    synapse_type=str(row.get('type', 'chemical'))
                )
                data.synapses.append(syn)
        
        data.regions = list(set(n.region for n in data.neurons.values()))
        data.cell_types = list(set(n.cell_type for n in data.neurons.values()))
        
        return data
    
    def load_neuprint_json(self, filepath: str) -> ConnectomeData:
        """Load connectome from NeuPrint JSON export."""
        import json
        data = ConnectomeData()
        
        with open(filepath) as f:
            export = json.load(f)
        
        # Neurons
        for n in export.get('neurons', []):
            neuron = Neuron(
                id=n['bodyId'],
                region=n.get('region', 'unknown'),
                cell_type=n.get('type', 'unknown'),
                hemisphere=n.get('hemisphere', 'unknown'),
                x=n.get('x', 0.0),
                y=n.get('y', 0.0),
                z=n.get('z', 0.0),
                n_pre=n.get('pre', 0),
                n_post=n.get('post', 0),
                neurotransmitter=n.get('nt', 'unknown')
            )
            data.neurons[neuron.id] = neuron
        
        # Synapses (ROI-based)
        for s in export.get('synapses', []):
            syn = Synapse(
                pre_id=s['pre'],
                post_id=s['post'],
                x=s.get('x', 0.0),
                y=s.get('y', 0.0),
                z=s.get('z', 0.0),
                size=s.get('size', 1.0),
                synapse_type=s.get('type', 'chemical')
            )
            data.synapses.append(syn)
        
        data.regions = list(set(n.region for n in data.neurons.values()))
        data.cell_types = list(set(n.cell_type for n in data.neurons.values()))
        
        return data
    
    def create_synthetic(self, config: Dict = None) -> ConnectomeData:
        """Create synthetic connectome based on config.yaml specifications."""
        config = config or self.config
        data = ConnectomeData()
        
        # Create neurons per region
        neuron_id = 0
        for region_name, region_cfg in config.get('regions', {}).items():
            n_neurons = region_cfg.get('neurons', 1000)
            cell_types = region_cfg.get('cell_types', ['unknown'])
            neurotransmitters = region_cfg.get('neurotransmitters', ['acetylcholine'])
            
            for i in range(n_neurons):
                cell_type = np.random.choice(cell_types)
                nt = np.random.choice(neurotransmitters)
                
                neuron = Neuron(
                    id=neuron_id,
                    region=region_name,
                    cell_type=cell_type,
                    hemisphere=np.random.choice(['left', 'right']),
                    x=np.random.uniform(0, 1000),
                    y=np.random.uniform(0, 1000),
                    z=np.random.uniform(0, 500),
                    neurotransmitter=nt
                )
                data.neurons[neuron_id] = neuron
                neuron_id += 1
        
        # Create synapses based on connection probabilities
        regions = list(config.get('regions', {}).keys())
        conn_probs = config.get('connection_probabilities', {})
        
        for pre_region in regions:
            pre_neurons = data.get_neurons_by_region(pre_region)
            for post_region in regions:
                post_neurons = data.get_neurons_by_region(post_region)
                
                prob_key = f"{pre_region}->{post_region}"
                prob = conn_probs.get(prob_key, 0.01)
                
                for pre_n in pre_neurons:
                    for post_n in post_neurons:
                        if np.random.random() < prob:
                            syn = Synapse(
                                pre_id=pre_n.id,
                                post_id=post_n.id,
                                x=(pre_n.x + post_n.x) / 2,
                                y=(pre_n.y + post_n.y) / 2,
                                z=(pre_n.z + post_n.z) / 2,
                                size=np.random.exponential(1.0)
                            )
                            data.synapses.append(syn)
                            pre_n.n_post += 1
                            post_n.n_pre += 1
        
        data.regions = regions
        data.cell_types = list(set(n.cell_type for n in data.neurons.values()))
        
        return data
