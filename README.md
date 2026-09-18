# Fruit Fly Brain Game Playing System

A biologically-inspired neural simulation system that connects a fruit fly (Drosophila melanogaster) connectome to game environments, enabling the fly brain to learn and play games through dopamine-gated STDP plasticity.

## Overview

This project implements a closed-loop system where:
1. **Game environments** (Pinball, Pong, Maze, Odor Navigation, Looming Escape) provide visual/motor input
2. **Neural network** (5 brain regions, ~52K neurons) processes sensory input through biologically-constrained connectivity
3. **Plasticity mechanisms** (STDP + dopamine gating) enable learning from reward signals
4. **Visualization tools** show real-time brain activation during gameplay

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Game Env      │────▶│  Neural Network  │────▶│   Action Out    │
│  (5 games)      │     │  (5 regions)     │     │  (Left/Right/   │
│                 │     │                  │     │   Jump/None)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Plasticity      │
                     │  (STDP + DA)     │
                     └──────────────────┘
```

### Brain Regions (5 total)
| Region | Neurons | Function |
|--------|---------|----------|
| Optic Lobes | 20,000 | Visual processing (screen → spikes) |
| Central Complex | 10,000 | Navigation, motor planning |
| Mushroom Body | 15,000 | Learning, memory, valence |
| Lateral Horn | 5,000 | Innate behavioral responses |
| Descending Neurons | 2,000 | Motor output (actions) |

### Signal Flow
```
Screen pixels (84×84) → Optic Lobes (receptive fields) → Central Complex → 
Mushroom Body (Kenyon cells) → Lateral Horn → Descending Neurons → Action
```

## Installation

```bash
# Clone repository
git clone https://github.com/zzhpublic/fruitfly-brain-game.git
cd fruitfly-brain-game

# Install dependencies
pip install -r requirements.txt

# Optional: Set NeuPrint API token for real connectome data
export NEUPRINT_TOKEN="your_token_here"
```

### Requirements
- Python 3.8+
- PyTorch, NumPy, SciPy
- Pygame (for games)
- Matplotlib (for visualization)
- NetworkX (for connectome graphs)
- Optional: `neuprint-python` for NeuPrint API access

## Quick Start

### 1. Test Signal Propagation (Core Verification)
```bash
python demo_closed_loop.py --game pinball --steps 1000
```
Expected: Spikes propagate through all 5 regions (optic_lobes → central_complex → mushroom_body → lateral_horn → descending_neurons)

### 2. Run Brain Activation Visualization
```bash
python demo_brain_activation.py --game pinball --steps 500 --save-frames
```
Shows: 3D brain view, spike raster, firing rates, voltage traces, game overlay

### 3. Test Google FlyBrain Integration
```bash
python demo_flybrain.py --regions optic_lobes central_complex mushroom_body
```
Downloads real connectome data from NeuPrint/FlyWire (synthetic fallback if no token)

### 4. Train with Reinforcement Learning
```bash
python train.py --game pinball --episodes 100 --checkpoint-dir checkpoints/
```
Saves checkpoints with synaptic weights, neuron states, neuromodulator states

## Game Environments

| Game | Description | Observation | Actions |
|------|-------------|-------------|---------|
| **Pinball** | Ball bounces, paddle catches | 84×84 screen | Left, Right, None |
| **Pong** | Classic paddle vs ball | 84×84 screen | Up, Down, None |
| **Maze** | Navigate to target | 84×84 top-down | Forward, Left, Right |
| **Odor** | Follow odor gradient | 84×84 concentration | Up, Down, Left, Right |
| **Looming** | Escape expanding object | 84×84 looming disk | Jump, Left, Right |

### Pinball Specifics (Latest Improvements)
- Ball radius: 2px (smaller for precision)
- Paddle width: 20px (shorter, more challenging)
- Auto-paddle: Predicts ball landing position accounting for wall bounces
- Constant speed: 4.0 (no acceleration on paddle hits)
- Angle variation: ±8.5° random on wall collisions

## Configuration

### Network Config (`config/connectome_test.yaml`)
```yaml
regions:
  optic_lobes: 20000
  central_complex: 10000
  mushroom_body: 15000
  lateral_horn: 5000
  descending_neurons: 2000

connection_probabilities:
  optic_lobes->central_complex: 0.02
  central_complex->mushroom_body: 0.015
  mushroom_body->lateral_horn: 0.01
  lateral_horn->descending_neurons: 0.03
  # ... recurrent connections

conn_scale: 100  # Weight multiplier for signal propagation
```

### FlyBrain Config (`config/google_flybrain.yaml`)
```yaml
neuprint:
  server: "https://neuprint.janelia.org"
  dataset: "hemibrain:v1.2.1"
  
target_regions:
  - "ME(R)"  # Medulla - visual processing
  - "LO(R)"  # Lobula - visual processing
  - "CX"     # Central complex
  - "MB"     # Mushroom body
  - "LH"     # Lateral horn
  - "DN"     # Descending neurons
```

## Key Components

### Neural Simulation (`src/`)
```
src/
├── neurons/
│   └── lif.py           # Vectorized LIF population (100x speedup)
├── synapses/
│   └── stdp.py          # STDP with dopamine gating
├── networks/
│   └── assembly.py      # NetworkBuilder with prev_spikes fix
├── connectome/
│   ├── loader.py        # Synthetic connectome (vectorized Poisson)
│   └── google_flybrain.py  # NeuPrint/FlyWire API clients
├── games/
│   ├── pinball.py       # Pinball with auto-paddle
│   ├── pong.py
│   ├── maze.py
│   ├── odor_navigation.py
│   └── looming_escape.py
├── neuromodulation/
│   └── dopamine.py      # Dopamine system (VTA/SNc)
└── utils/
    └── config.py        # Configuration loading
```

### Critical Fixes (Signal Propagation)
1. **NetworkBuilder.step()**: Uses `prev_spikes` for synaptic transmission (spikes from step t affect step t+1)
2. **LIFPopulation.step()**: Added `I_syn_ext` parameter for external synaptic current injection
3. **STDPSynapse.compute_current()**: Fixed sign convention `g * (E_rev - V)` for inward current
4. **Dale's Law**: Fixed E_rev assignment (per synapse, not per region)
5. **Connection probabilities**: Added to test config for synthetic generation

## Data Flow

### Input Pipeline (Screen → Spikes)
```python
# In demo_closed_loop.py
screen = game.get_screen()           # (84, 84) uint8
spikes = screen_to_spikes(screen)    # Poisson encoding → optic_lobes spikes
network.step(spikes)                 # Propagate through 5 regions
action = decode_action(spikes)       # descending_neurons → action
game.step(action)                    # Execute in environment
```

### Output Pipeline (Spikes → Action)
```python
# Descending neuron populations mapped to actions
dn_spikes = network.populations['descending_neurons'].spikes
left_rate = dn_spikes[:500].mean()   # First 500 neurons = LEFT
right_rate = dn_spikes[500:1000].mean()  # Next 500 = RIGHT
# ... threshold comparison → discrete action
```

## Visualization

### Brain Activation Demo (`demo_brain_activation.py`)
Real-time 5-panel display:
1. **3D Brain View** - Regions colored by activity (red=high)
2. **Spike Raster** - Last 100ms of spikes per region
3. **Firing Rates** - Rolling 500ms rate per region
4. **Voltage Traces** - Sample neuron membrane potentials
5. **Game Overlay** - Current game frame with receptive fields

```bash
# Save frames for video
python demo_brain_activation.py --game pinball --steps 1000 --save-frames --output-dir frames/
ffmpeg -framerate 30 -i frames/frame_%04d.png brain_activation.mp4
```

## Google FlyBrain / NeuPrint Integration

### Features
- **NeuPrintClient**: Cypher queries against hemibrain:v1.2.1
- **FlyWireClient**: FlyWire FAFB dataset access
- **FlyBrainIntegrator**: Maps FlyBrain regions → our 5-region model
- **Caching**: Local pickle cache avoids re-downloads
- **Synthetic Fallback**: Works without API token

### Usage
```python
from src.connectome.google_flybrain import FlyBrainIntegrator

integrator = FlyBrainIntegrator(config_path="config/google_flybrain.yaml")
connectome = integrator.build_connectome()
# Returns: adjacency matrix, neuron metadata, region mapping
```

### Target Cell Types
| Region | Cell Types |
|--------|------------|
| Optic Lobes | T4, T5, Mi1, Mi4, Tm3, Tm9, CT1, LPi |
| Central Complex | E-PG, P-EN, P-EG, Delta7, PFN |
| Mushroom Body | KC, APL, DAN, MBON |
| Lateral Horn | LHON, LHIN, PN |
| Descending Neurons | DNa01, DNa02, DNb01, DNb02, DNp01 |

## Training

### Reinforcement Learning Loop
```python
# In train.py
for episode in range(episodes):
    obs = env.reset()
    for step in range(max_steps):
        spikes = screen_to_spikes(obs)
        network.step(spikes)
        action = decode_action(network)
        obs, reward, done, _ = env.step(action)
        
        # Dopamine signal from reward
        dopamine = compute_dopamine(reward, expected_reward)
        network.apply_plasticity(dopamine)
```

### Checkpoint System
Saves to `checkpoints/`:
- `synaptic_weights.npz` - All STDP weights
- `neuron_states.npz` - V, refractory, g_syn_exc, g_syn_inh
- `neuromodulator_states.npz` - DA, eligibility traces
- `metrics.json` - Episode rewards, spike counts, plasticity stats

```bash
# Resume training
python train.py --resume checkpoints/episode_50
```

## Performance

| Metric | Value |
|--------|-------|
| Neurons (test config) | ~52,000 |
| Synapses (test config) | ~1.2M |
| Step time (vectorized) | ~2ms |
| Signal propagation | 5 regions verified |
| Games implemented | 5 |

## Troubleshooting

### No Spikes Propagating
- Check `conn_scale` in config (try 100-200)
- Verify `connection_probabilities` in config
- Ensure `prev_spikes` is used in NetworkBuilder.step()

### Import Errors
```bash
# Run from project root
cd /root/fruitfly-brain-game
PYTHONPATH=. python demo_closed_loop.py
```

### Headless Display (for CI/servers)
```bash
# Install Xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run -a python demo_closed_loop.py --game pinball
```

## Project Structure

```
fruitfly-brain-game/
├── config/
│   ├── connectome_test.yaml      # Test network config
│   ├── connectome_full.yaml      # Full ~70k neuron config
│   └── google_flybrain.yaml      # FlyBrain integration config
├── src/
│   ├── neurons/lif.py
│   ├── synapses/stdp.py
│   ├── networks/assembly.py
│   ├── connectome/loader.py
│   ├── connectome/google_flybrain.py
│   ├── games/*.py
│   ├── neuromodulation/dopamine.py
│   └── utils/config.py
├── demo_closed_loop.py           # Main closed-loop demo
├── demo_brain_activation.py      # Brain visualization
├── demo_flybrain.py              # FlyBrain integration demo
├── train.py                      # RL training loop
├── visualize.py                  # Static analysis plots
├── ARCHITECTURE.md               # Detailed architecture docs
├── requirements.txt
└── README.md
```

## References

- **NeuPrint**: https://neuprint.janelia.org (hemibrain:v1.2.1)
- **FlyWire**: https://flywire.ai (FAFB dataset)
- **Drosophila Connectome**: Scheffer et al. 2020, Xu et al. 2020
- **STDP with Dopamine**: Izhikevich 2007, Frémaux & Gerstner 2016

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Citation

If you use this code, please cite:
```
@software{fruitfly-brain-game,
  title = {Fruit Fly Brain Game Playing System},
  author = {zzhpublic},
  url = {https://github.com/zzhpublic/fruitfly-brain-game},
  year = {2024}
}
```