"""
Google FlyBrain / NeuPrint Integration Module

Provides access to Google's FlyBrain connectome data via:
1. NeuPrint API (neuprint.janelia.org) - primary public access
2. FlyWire API (flywire.ai) - for proofread neurons
3. Direct data downloads from Google Cloud Storage

References:
- NeuPrint: https://neuprint.janelia.org/
- FlyWire: https://flywire.ai/
- FlyBrain: https://github.com/google/flybrain
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import neuprint-python client
try:
    import neuprint
    NEUPRINT_AVAILABLE = True
except ImportError:
    NEUPRINT_AVAILABLE = False
    logger.warning("neuprint-python not installed. Install with: pip install neuprint-python")


@dataclass
class FlyBrainConfig:
    """Configuration for FlyBrain access."""
    # NeuPrint API
    neuprint_server: str = "https://neuprint.janelia.org"
    neuprint_dataset: str = "hemibrain:v1.2.1"  # or "flywire_fafb:v1.0"
    neuprint_token: Optional[str] = None  # Optional auth token
    
    # FlyWire API
    flywire_server: str = "https://flywire.ai"
    flywire_token: Optional[str] = None
    
    # Cache directory
    cache_dir: str = "./data/flybrain_cache"
    
    # Download settings
    chunk_size: int = 10000
    timeout: int = 300


class NeuPrintClient:
    """Client for NeuPrint API (hemibrain, flywire_fafb datasets) using neuprint-python."""
    
    def __init__(self, config: FlyBrainConfig):
        self.config = config
        if not NEUPRINT_AVAILABLE:
            raise ImportError("neuprint-python not installed. Run: pip install neuprint-python")
        
        self.client = neuprint.Client(
            config.neuprint_server,
            dataset=config.neuprint_dataset,
            token=config.neuprint_token
        )
    
    def _query(self, cypher: str) -> Dict:
        """Execute Cypher query against NeuPrint using neuprint-python."""
        try:
            result = self.client.fetch_custom(cypher)
            if hasattr(result, 'values'):
                # Convert numpy array to list of lists
                return {"data": result.values.tolist()}
            elif hasattr(result, 'to_dict'):
                return {"data": result.to_dict('records')}
            else:
                return {"data": result}
        except Exception as e:
            logger.error(f"NeuPrint query failed: {e}")
            raise
    
    def get_neurons_by_region(self, region: str, limit: int = 10000) -> List[Dict]:
        """Get neurons in a brain region."""
        cypher = f"""
        MATCH (n:Neuron)
        WHERE n.region = '{region}'
        RETURN n.bodyId as bodyId, n.type as type, n.instance as instance,
               n.region as region, n.hemisphere as hemisphere,
               n.x as x, n.y as y, n.z as z,
               n.pre as pre, n.post as post
        LIMIT {limit}
        """
        result = self._query(cypher)
        return result.get('data', [])
    
    def get_neurons_by_type(self, cell_type: str, limit: int = 10000) -> List[Dict]:
        """Get neurons by cell type (uses CONTAINS for partial matching)."""
        cypher = f"""
        MATCH (n:Neuron)
        WHERE n.type CONTAINS '{cell_type}'
        RETURN n.bodyId as bodyId, n.type as type, n.instance as instance,
               n.region as region, n.hemisphere as hemisphere,
               n.x as x, n.y as y, n.z as z,
               n.pre as pre, n.post as post
        LIMIT {limit}
        """
        result = self._query(cypher)
        return result.get('data', [])
    
    def get_synapses_between(self, pre_ids: List[int], post_ids: List[int]) -> List[Dict]:
        """Get synapses between specific neuron sets."""
        pre_str = ",".join(map(str, pre_ids))
        post_str = ",".join(map(str, post_ids))
        cypher = f"""
        MATCH (pre:Neuron)-[s:Synapse]->(post:Neuron)
        WHERE pre.bodyId IN [{pre_str}] AND post.bodyId IN [{post_str}]
        RETURN pre.bodyId as pre, post.bodyId as post,
               s.x as x, s.y as y, s.z as z,
               s.size as size, s.type as type
        """
        result = self._query(cypher)
        return result.get('data', [])
    
    def get_roi_info(self) -> List[Dict]:
        """Get available ROIs (brain regions)."""
        cypher = """
        MATCH (r:ROIInfo)
        RETURN r.name as name, r.volume as volume
        """
        result = self._query(cypher)
        return result.get('data', [])
    
    def get_neuron_roi_counts(self, body_ids: List[int]) -> List[Dict]:
        """Get ROI counts for neurons."""
        ids_str = ",".join(map(str, body_ids))
        cypher = f"""
        MATCH (n:Neuron)
        WHERE n.bodyId IN [{ids_str}]
        RETURN n.bodyId as bodyId, n.roiInfo as roiInfo
        """
        result = self._query(cypher)
        return result.get('data', [])


class FlyWireClient:
    """Client for FlyWire API (proofread connectome)."""
    
    def __init__(self, config: FlyBrainConfig):
        self.config = config
        self.base_url = f"{config.flywire_server}/api/v1"
        self.headers = {"Content-Type": "application/json"}
        if config.flywire_token:
            self.headers["Authorization"] = f"Bearer {config.flywire_token}"
    
    def get_neuron(self, root_id: int) -> Dict:
        """Get neuron details by root ID."""
        url = f"{self.base_url}/neuron/{root_id}"
        response = requests.get(url, headers=self.headers, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_synapses(self, root_id: int, direction: str = "both") -> List[Dict]:
        """Get synapses for a neuron."""
        url = f"{self.base_url}/neuron/{root_id}/synapses"
        params = {"direction": direction}
        response = requests.get(url, headers=self.headers, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_mesh(self, root_id: int) -> Dict:
        """Get 3D mesh for a neuron."""
        url = f"{self.base_url}/neuron/{root_id}/mesh"
        response = requests.get(url, headers=self.headers, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()
    
    def search_neurons(self, query: str, limit: int = 100) -> List[Dict]:
        """Search neurons by name/type."""
        url = f"{self.base_url}/search"
        params = {"q": query, "limit": limit}
        response = requests.get(url, headers=self.headers, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()


class FlyBrainIntegrator:
    """
    High-level integrator for Google FlyBrain data.
    
    Combines NeuPrint (hemibrain) and FlyWire (FAFB) data sources
    to build connectome datasets compatible with our SNN simulation.
    """
    
    def __init__(self, config: FlyBrainConfig = None):
        self.config = config or FlyBrainConfig()
        self.neuprint = NeuPrintClient(self.config)
        self.flywire = FlyWireClient(self.config)
        
        # Ensure cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def fetch_hemibrain_regions(self) -> Dict[str, List[int]]:
        """
        Fetch all hemibrain regions and their neuron counts.
        
        Returns:
            Dict mapping region name -> list of bodyIds
        """
        roi_info = self.neuprint.get_roi_info()
        regions = {}
        
        for roi in roi_info:
            region_name = roi['name']
            # Get neuron count for this ROI
            cypher = f"""
            MATCH (n:Neuron)
            WHERE n.roiInfo CONTAINS '{region_name}'
            RETURN count(n) as count
            """
            result = self.neuprint._query(cypher)
            count = result.get('data', [[0]])[0][0]
            if count > 0:
                regions[region_name] = count
                logger.info(f"Region {region_name}: {count} neurons")
        
        return regions
    
    def fetch_region_neurons(self, region: str, max_neurons: int = 5000) -> List[Dict]:
        """Fetch neurons for a specific region (uses cell type matching)."""
        return self.neuprint.get_neurons_by_type(region, limit=max_neurons)
    
    def fetch_cell_type_neurons(self, cell_type: str, max_neurons: int = 5000) -> List[Dict]:
        """Fetch neurons for a specific cell type."""
        raw = self.neuprint.get_neurons_by_type(cell_type, limit=max_neurons)
        # Convert list of lists to list of dicts
        keys = ['bodyId', 'type', 'instance', 'region', 'hemisphere', 'x', 'y', 'z', 'pre', 'post']
        return [dict(zip(keys, row)) for row in raw]
    
    def fetch_subcircuit(self, 
                         pre_regions: List[str], 
                         post_regions: List[str],
                         max_neurons_per_region: int = 1000) -> Tuple[List[Dict], List[Dict]]:
        """
        Fetch a subcircuit between pre and post cell types.
        
        Returns:
            (neurons, synapses) lists
        """
        all_neurons = {}
        all_synapses = []
        
        # Fetch neurons from all cell types
        for cell_type in pre_regions + post_regions:
            neurons = self.fetch_cell_type_neurons(cell_type, max_neurons_per_region)
            for n in neurons:
                all_neurons[n['bodyId']] = n
        
        pre_ids = [n['bodyId'] for n in all_neurons.values() 
                   if n.get('type', '') in pre_regions]
        post_ids = [n['bodyId'] for n in all_neurons.values() 
                    if n.get('type', '') in post_regions]
        
        # Fetch synapses between them
        synapses = self.neuprint.get_synapses_between(pre_ids, post_ids)
        
        return list(all_neurons.values()), synapses
    
    def export_to_connectome_data(self, 
                                   neurons: List[Dict], 
                                   synapses: List[Dict]) -> 'ConnectomeData':
        """Convert NeuPrint data to our ConnectomeData format."""
        from .loader import ConnectomeData, Neuron, Synapse
        
        data = ConnectomeData()
        
        # Map hemibrain cell types to our brain regions
        cell_type_to_region = {
            # Optic lobes
            'LC': 'optic_lobes', 'LPLC': 'optic_lobes', 'LPTC': 'optic_lobes',
            'Tm': 'optic_lobes', 'TmY': 'optic_lobes', 'mALC': 'optic_lobes', 'AVLP': 'optic_lobes',
            # Central complex
            'PEN': 'central_complex', 'EPG': 'central_complex', 'PEG': 'central_complex',
            'PFN': 'central_complex', 'PFL': 'central_complex', 'FR': 'central_complex',
            'FC': 'central_complex', 'FS': 'central_complex', 'FB': 'central_complex',
            'hDelta': 'central_complex', 'vDelta': 'central_complex',
            # Mushroom body
            'KC': 'mushroom_body', 'APL': 'mushroom_body', 'DAN': 'mushroom_body',
            'MBON': 'mushroom_body', 'PPL': 'mushroom_body', 'PAM': 'mushroom_body',
            # Lateral horn
            'LH': 'lateral_horn', 'LHPV': 'lateral_horn',
            # Descending neurons
            'DNa': 'descending_neurons', 'DNb': 'descending_neurons', 'DNc': 'descending_neurons',
            'DNd': 'descending_neurons', 'DNg': 'descending_neurons', 'DNp': 'descending_neurons',
            'DN': 'descending_neurons', 'MDN': 'descending_neurons',
        }
        
        def get_region(cell_type: str) -> str:
            for prefix, region in cell_type_to_region.items():
                if cell_type.startswith(prefix):
                    return region
            return 'unknown'
        
        # Convert neurons
        for n in neurons:
            cell_type = n.get('type', 'unknown')
            region = get_region(cell_type)
            neuron = Neuron(
                id=n['bodyId'],
                region=region,
                cell_type=cell_type,
                hemisphere=n.get('hemisphere', 'unknown'),
                x=n.get('x', 0.0),
                y=n.get('y', 0.0),
                z=n.get('z', 0.0),
                n_pre=n.get('pre', 0),
                n_post=n.get('post', 0),
                neurotransmitter=n.get('nt', 'unknown')
            )
            data.neurons[neuron.id] = neuron
            data.neurons[neuron.id] = neuron
        
        # Convert synapses
        for s in synapses:
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
            if s['pre'] in data.neurons:
                data.neurons[s['pre']].n_post += 1
            if s['post'] in data.neurons:
                data.neurons[s['post']].n_pre += 1
        
        data.regions = list(set(n.region for n in data.neurons.values()))
        data.cell_types = list(set(n.cell_type for n in data.neurons.values()))
        
        return data
    
    def build_game_ready_connectome(self, 
                                     target_regions: List[str] = None,
                                     max_neurons_per_region: int = 1000) -> 'ConnectomeData':
        """
        Build a connectome optimized for game playing.
        
        Target cell types for visual-motor pathway:
        - Optic lobe cell types (LC, LPLC, LPTC, Tm, TmY, mALC, AVLP)
        - Central complex (PEN, EPG, PEG, PFN, PFL, FR, FC, FS, FB, hDelta, vDelta)
        - Mushroom body (KC, APL, DAN, MBON, PPL, PAM)
        - Lateral horn (LH, LHPV)
        - Descending neurons (DNa, DNb, DNc, DNd, DNg, DNp, DN, MDN)
        """
        if target_regions is None:
            target_regions = [
                'LC', 'LPLC', 'LPTC', 'Tm', 'TmY', 'mALC', 'AVLP',  # Optic lobes
                'PEN', 'EPG', 'PEG', 'PFN', 'PFL', 'FR', 'FC', 'FS', 'FB', 'hDelta', 'vDelta',  # Central complex
                'KC', 'APL', 'DAN', 'MBON', 'PPL', 'PAM',  # Mushroom body
                'LH', 'LHPV',  # Lateral horn
                'DNa', 'DNb', 'DNc', 'DNd', 'DNg', 'DNp', 'DN', 'MDN'  # Descending neurons
            ]
        
        logger.info(f"Fetching subcircuit for cell types: {target_regions}")
        neurons, synapses = self.fetch_subcircuit(
            pre_regions=target_regions[:7],  # Optic lobes as input
            post_regions=target_regions[7:],  # Central complex, MB, LH, DN as output
            max_neurons_per_region=max_neurons_per_region
        )
        
        return self.export_to_connectome_data(neurons, synapses)
    
    def cache_connectome(self, data: 'ConnectomeData', name: str):
        """Cache connectome to disk."""
        import pickle
        cache_path = Path(self.config.cache_dir) / f"{name}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Cached connectome to {cache_path}")
    
    def load_cached_connectome(self, name: str) -> 'ConnectomeData':
        """Load connectome from cache."""
        import pickle
        cache_path = Path(self.config.cache_dir) / f"{name}.pkl"
        with open(cache_path, 'rb') as f:
            return pickle.load(f)


def create_flybrain_config_from_env() -> FlyBrainConfig:
    """Create config from environment variables."""
    return FlyBrainConfig(
        neuprint_token=os.getenv('NEUPRINT_TOKEN'),
        flywire_token=os.getenv('FLYWIRE_TOKEN'),
        cache_dir=os.getenv('FLYBRAIN_CACHE_DIR', './data/flybrain_cache')
    )


# Convenience function for quick access
def fetch_hemibrain_visual_motor(max_neurons: int = 1000) -> 'ConnectomeData':
    """
    Quick function to fetch visual-motor pathway from hemibrain.
    
    Usage:
        from src.connectome.google_flybrain import fetch_hemibrain_visual_motor
        connectome = fetch_hemibrain_visual_motor(max_neurons=2000)
    """
    config = create_flybrain_config_from_env()
    integrator = FlyBrainIntegrator(config)
    return integrator.build_game_ready_connectome(max_neurons_per_region=max_neurons)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    config = create_flybrain_config_from_env()
    integrator = FlyBrainIntegrator(config)
    
    # Get available regions
    print("Fetching hemibrain regions...")
    regions = integrator.fetch_hemibrain_regions()
    print(f"Found {len(regions)} regions")
    for name, count in sorted(regions.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name}: {count} neurons")