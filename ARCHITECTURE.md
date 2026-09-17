# Fruit Fly Brain Game - Architecture Documentation

## Overview

This project implements a spiking neural network (SNN) simulation of the Drosophila (fruit fly) brain, connected to game environments through a closed-loop system. The network processes visual input from games and generates motor actions.

---

## Code Structure

```
fruitfly-brain-game/
├── config/
│   ├── connectome.yaml          # Full brain config (~70k neurons, 5 regions)
│   ├── connectome_test.yaml     # Test config (1,199 neurons)
│   └── training.yaml            # Training hyperparameters
├── src/
│   ├── neurons/
│   │   ├── __init__.py
│   │   └── lif.py               # LIF neuron model + vectorized LIFPopulation
│   ├── synapses/
│   │   ├── __init__.py
│   │   └── stdp.py              # STDP with dopamine-gated plasticity
│   ├── neuromod/
│   │   ├── __init__.py
│   │   ├── modulator.py         # Base neuromodulator class
│   │   ├── dopamine.py          # Dopamine system (reward prediction error)
│   │   ├── octopamine.py        # Octopamine system (arousal/vigor)
│   │   └── serotonin.py         # Serotonin system (patience/impulse control)
│   ├── connectome/
│   │   ├── __init__.py
│   │   └── loader.py            # Connectome loading (4 formats + synthetic)
│   ├── networks/
│   │   ├── __init__.py
│   │   └── assembly.py          # NetworkBuilder - assembles brain from connectome
│   ├── games/
│   │   ├── __init__.py
│   │   ├── pong.py              # Pong environment
│   │   ├── maze.py              # Maze navigation
│   │   ├── odor.py              # Odor navigation
│   │   ├── looming.py           # Looming escape
│   │   └── pinball.py           # Pinball/Breakout
│   └── __init__.py
├── train.py                     # Training entry point with checkpoints
├── visualize.py                 # Real-time matplotlib dashboard
├── demo_camera.py               # Camera input demo
├── demo_closed_loop.py          # Closed-loop: game screen → network → action
├── play.py                      # Inference/evaluation script
├── requirements.txt
└── README.md
```

---

## Core Components

### 1. Neuron Model (`src/neurons/lif.py`)

**LIFParams** - Neuron parameters:
```python
@dataclass
class LIFParams:
    C_m: float = 200.0        # Membrane capacitance (pF)
    g_L: float = 10.0         # Leak conductance (nS)
    E_L: float = -60.0        # Leak reversal potential (mV)
    V_th: float = -40.0       # Spike threshold (mV)
    V_reset: float = -60.0    # Reset potential (mV)
    t_ref: float = 2.0        # Refractory period (ms)
    E_exc: float = 0.0        # Excitatory reversal (mV)
    E_inh: float = -75.0      # Inhibitory reversal (mV)
```

**LIFPopulation** - Vectorized population simulation:
- Processes all neurons in a region simultaneously using NumPy
- 100x speedup over per-neuron loops
- State variables: `V` (voltage), `refractory` (refractory timer), `g_syn_exc`, `g_syn_inh` (synaptic conductances)
- `step(dt, I_ext, I_syn_ext=None, E_syn_ext=None)` - advances simulation by dt ms

### 2. Synapse Model (`src/synapses/stdp.py`)

**STDPSynapse** - Spike-timing-dependent plasticity:
- **Eligibility trace**: `e_ij` tracks pre-post spike timing
- **Dopamine gating**: Weight change = `η * e_ij * dopamine_concentration`
- **Dale's law**: Each neuron is either excitatory or inhibitory
- `compute_current(pre_spikes, V_post)` - returns synaptic current: `g * (E_rev - V_post)`

### 3. Neuromodulatory Systems (`src/neuromod/`)

Three volume-transmission systems:
- **Dopamine**: Reward prediction error (RPE), gates STDP plasticity
- **Octopamine**: Arousal/vigor, modulates neuron gain
- **Serotonin**: Patience/impulse control, modulates thresholds

All use diffusion-based volume transmission with spatial decay.

### 4. Connectome Loader (`src/connectome/loader.py`)

Supports 4 formats:
1. **FlyWire HDF5** - Full connectome from FlyWire
2. **FAFB Zarr** - FAFB dataset
3. **NeuPrint JSON** - NeuPrint connectome
4. **Synthetic** - Generated from region sizes and connection probabilities

**Optimization**: Vectorized Poisson sampling with `max_synapses` cap (default 1M) for large networks.

### 5. Network Assembly (`src/networks/assembly.py`)

**NetworkBuilder** - Builds network from connectome config:
- Creates LIFPopulation for each brain region
- Creates STDPSynapse for each connection
- Enforces Dale's law (disabled by default - was causing issues)
- **Critical fix**: Stores `prev_spikes` from previous step for synaptic transmission

**Network step flow**:
```
1. Encode game observation → optic_lobes spike input
2. For each region:
   a. Compute synaptic currents from prev_spikes
   b. Step LIFPopulation with I_ext + I_syn
   c. Collect current step spikes
3. Store current spikes as prev_spikes for next step
4. Decode descending_neurons spikes → action
```

---

## Game Environments (`src/games/`)

All games inherit from `gymnasium.Env` with spike-based I/O:

### Observation Encoding (Game → Network)
```
Game State (ball pos, paddle pos, velocity)
    ↓
Receptive Fields (8x8 grid for motion, position bins)
    ↓
Poisson Spike Generation (rate → spike probability)
    ↓
Spike Vector (96 dims: 64 motion + 32 position) → optic_lobes
```

### Action Decoding (Network → Game)
```
descending_neurons spikes (10 neurons: 5 left, 5 right)
    ↓
Sum spikes per side
    ↓
move = (right_rate - left_rate) * 15.0 pixels/step
    ↓
Paddle movement
```

### Games Available

| Game | Observation | Action | Description |
|------|-------------|--------|-------------|
| Pong | 96-d spike vector | 10-d spike vector | Classic Pong |
| Maze | 96-d spike vector | 10-d spike vector | Maze navigation |
| Odor | 96-d spike vector | 10-d spike vector | Odor gradient following |
| Looming | 96-d spike vector | 10-d spike vector | Escape expanding object |
| Pinball | 96-d spike vector | 10-d spike vector | Breakout with auto-paddle |

---

## Data Flow: Game → Network → Game

### Closed-Loop Pipeline (`demo_closed_loop.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAME LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. GAME STEP                                                   │
│     env.step(action) → obs, reward, terminated, info           │
│                                                                 │
│  2. OBSERVATION ENCODING (env._encode_observation)             │
│     Game state → Receptive fields → Poisson spikes             │
│     Output: spike_vector[96] (float32, 0 or 1)                 │
│                                                                 │
│  3. NETWORK INPUT INJECTION                                     │
│     network.step(optic_lobe_input=spike_vector)                │
│                                                                 │
│  4. NETWORK PROPAGATION (NetworkBuilder.step)                  │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ For each region:                                    │    │
│     │   I_syn = sum_j W_ij * g_syn * (E_rev - V) * prev_spikes_j │
│     │   V, spikes = LIFPopulation.step(I_ext + I_syn)    │    │
│     │   Store spikes as prev_spikes for next step        │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  5. ACTION DECODING                                             │
│     descending_neurons spikes → left/right rates               │
│     action = argmax([left_rate, stay_rate, right_rate])        │
│                                                                 │
│  6. LOOP BACK TO STEP 1                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Signal Flow

#### Step 1: Game Observation → Optic Lobes Input
```python
# In demo_closed_loop.py
obs, reward, terminated, truncated, info = env.step(action)
spike_input = obs  # Already encoded as spikes by env._encode_observation()

# Inject into network
network.step(optic_lobe_input=spike_input)
```

#### Step 2: Network Step (`NetworkBuilder.step`)
```python
def step(self, dt=1.0, optic_lobe_input=None):
    all_spikes = {}
    
    for region_name, pop in self.populations.items():
        # 1. Compute synaptic current from PREVIOUS step's spikes
        I_syn = np.zeros(pop.n)
        for conn_name, synapse in self.synapses.items():
            if synapse.post_region == region_name:
                pre_spikes = self.prev_spikes.get(synapse.pre_region, np.zeros(synapse.n_pre))
                I_syn += synapse.compute_current(pre_spikes, pop.V)
        
        # 2. Add external input for optic_lobes
        I_ext = np.zeros(pop.n)
        if region_name == "optic_lobes" and optic_lobe_input is not None:
            I_ext[:len(optic_lobe_input)] = optic_lobe_input * 50000  # pA
        
        # 3. Step population
        spikes = pop.step(dt, I_ext, I_syn_ext=I_syn, E_syn_ext=0.0)
        all_spikes[region_name] = spikes
    
    # 4. Store for next step
    self.prev_spikes = all_spikes.copy()
    return all_spikes
```

#### Step 3: Synaptic Current Computation (`STDPSynapse.compute_current`)
```python
def compute_current(self, pre_spikes, V_post):
    # pre_spikes: binary array [n_pre]
    # V_post: voltage array [n_post]
    # Returns: current array [n_post] in pA
    
    # Conductance from pre-synaptic spikes
    g_syn = self.weights.T @ pre_spikes  # [n_post]
    
    # Current = g * (E_rev - V)  (INWARD = DEPOLARIZING for E_rev > V)
    E_rev = self.E_exc if self.pre_type == "excitatory" else self.E_inh
    I_syn = g_syn * (E_rev - V_post)
    
    return I_syn
```

#### Step 4: Population Step (`LIFPopulation.step`)
```python
def step(self, dt, I_ext, I_syn_ext=None, E_syn_ext=None):
    # Update conductances (exponential decay)
    self.g_syn_exc *= np.exp(-dt / self.tau_syn_exc)
    self.g_syn_inh *= np.exp(-dt / self.tau_syn_inh)
    
    # Add external synaptic current
    if I_syn_ext is not None:
        if E_syn_ext is not None:
            # Convert current to conductance: g = I / (E_rev - V)
            g_ext = I_syn_ext / (E_syn_ext - self.V + 1e-10)
            self.g_syn_exc += np.maximum(g_ext, 0)
            self.g_syn_inh += np.minimum(g_ext, 0)
    
    # LIF dynamics: C * dV/dt = -g_L*(V-E_L) - g_exc*(V-E_exc) - g_inh*(V-E_inh) + I_ext
    dV = (-self.g_L * (self.V - self.E_L) 
          - self.g_syn_exc * (self.V - self.E_exc) 
          - self.g_syn_inh * (self.V - self.E_inh) 
          + I_ext) / self.C_m * dt
    
    self.V += dV
    
    # Spike detection
    spikes = (self.V >= self.V_th) & (self.refractory <= 0)
    self.V[spikes] = self.V_reset
    self.refractory[spikes] = self.t_ref
    
    return spikes.astype(np.float32)
```

#### Step 5: Action Decoding
```python
# In demo_closed_loop.py
dn_spikes = all_spikes.get("descending_neurons", np.zeros(50))
left_spikes = dn_spikes[:5].sum()
right_spikes = dn_spikes[5:].sum()
stay_spikes = dn_spikes[10:15].sum() if len(dn_spikes) > 10 else 0

rates = [left_spikes/5, stay_spikes/5, right_spikes/5]
action = np.argmax(rates)  # 0=left, 1=stay, 2=right
```

---

## Training Pipeline (`train.py`)

### Checkpoint System
```python
# Save
checkpoint = {
    'synaptic_weights': {name: syn.weights for name, syn in network.synapses.items()},
    'neuron_states': {name: {'V': pop.V, 'refractory': pop.refractory, 
                              'g_syn_exc': pop.g_syn_exc, 'g_syn_inh': pop.g_syn_inh} 
                      for name, pop in network.populations.items()},
    'neuromodulator_states': {name: mod.get_state() for name, mod in neuromodulators.items()},
    'metrics': metrics_history,
    'config': config
}
torch.save(checkpoint, path)

# Load
checkpoint = torch.load(path)
for name, weights in checkpoint['synaptic_weights'].items():
    network.synapses[name].weights = weights
for name, state in checkpoint['neuron_states'].items():
    network.populations[name].V = state['V']
    network.populations[name].refractory = state['refractory']
    ...
```

### Training Loop
```python
for episode in range(n_episodes):
    obs, _ = env.reset()
    for step in range(max_steps):
        # Encode observation
        spike_input = env._encode_observation()
        
        # Network forward
        spikes = network.step(optic_lobe_input=spike_input)
        
        # Decode action
        action = decode_action(spikes['descending_neurons'])
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Neuromodulation (dopamine = reward prediction error)
        dopamine.modulate(reward - expected_reward)
        
        # STDP weight updates happen in synapse.step()
        
        if terminated or truncated:
            break
```

---

## Visualization (`visualize.py`)

Real-time dashboard showing:
- **Raster plots**: Spike trains for each region
- **Voltage traces**: Membrane potential of sample neurons
- **Population rates**: Firing rate over time
- **Synaptic weights**: Weight matrices heatmap
- **Neuromodulator levels**: DA, OA, 5-HT concentrations
- **Game state**: Current game frame

---

## Key Design Decisions

### 1. Vectorized Simulation
- `LIFPopulation` processes entire regions at once
- Pre-computed synaptic weight matrices
- 100x speedup: 0.17s → 0.002s per step

### 2. Synaptic Transmission Timing
- **Critical**: Spikes from step t affect post-synaptic neurons at step t+1
- Implemented via `prev_spikes` storage in `NetworkBuilder`
- Without this, signals couldn't propagate through layers

### 3. Current Sign Convention
- `I_syn = g * (E_rev - V)` 
- Excitatory (E_rev=0mV): inward current when V < 0 → depolarizing
- Inhibitory (E_rev=-75mV): outward current when V > -75 → hyperpolarizing

### 4. Dale's Law
- Disabled by default (was incorrectly setting E_rev per region)
- Can be enabled in config if needed

### 5. Synthetic Connectome Generation
- Vectorized Poisson sampling per connection type
- `max_synapses` cap prevents memory explosion
- Connection probabilities defined in config

---

## Configuration

### `connectome_test.yaml` (Test Config)
```yaml
regions:
  optic_lobes: {n_neurons: 448, neuron_type: "LIF", ...}
  mushroom_body: {n_neurons: 216, ...}
  central_complex: {n_neurons: 280, ...}
  lateral_horn: {n_neurons: 150, ...}
  neuromodulatory: {n_neurons: 50, ...}
  descending_neurons: {n_neurons: 50, ...}

connections:
  - {pre: "optic_lobes", post: "central_complex", prob: 0.15, ...}
  - {pre: "optic_lobes", post: "mushroom_body", prob: 0.1, ...}
  ...

connection_probabilities:
  optic_lobes->central_complex: 0.15
  optic_lobes->mushroom_body: 0.1
  ...

neuron_params:
  C_m: 200.0
  g_L: 10.0
  E_L: -60.0
  V_th: -40.0
  V_reset: -60.0
  t_ref: 2.0
```

### `training.yaml`
```yaml
simulation:
  dt: 1.0          # ms per step
  duration: 1000   # ms per episode

plasticity:
  stdp:
    tau_plus: 20.0
    tau_minus: 20.0
    A_plus: 0.01
    A_minus: 0.012
    dopamine_gated: true

neuromodulation:
  dopamine:
    baseline: 0.1
    tau: 200.0
  octopamine:
    baseline: 0.05
    tau: 500.0
  serotonin:
    baseline: 0.05
    tau: 1000.0

rl:
  gamma: 0.99
  lr_actor: 1e-4
  lr_critic: 1e-3
```

---

## Running the System

### Closed-Loop Demo
```bash
# Pinball with auto-paddle (no network)
python demo_closed_loop.py --game pinball --auto-paddle

# Pinball with network control
python demo_closed_loop.py --game pinball --steps 500

# Other games
python demo_closed_loop.py --game pong --steps 500
python demo_closed_loop.py --game maze --steps 500
```

### Training
```bash
# Train on pinball
python train.py --game pinball --episodes 100 --checkpoint-dir checkpoints/

# Resume from checkpoint
python train.py --game pinball --episodes 100 --resume checkpoints/latest.pt
```

### Visualization
```bash
python visualize.py --game pinball --steps 500
```

---

## Signal Propagation Verification

The network now correctly propagates signals:
```
Optic Lobes (448 neurons) 
    → 20 spikes from game input
    ↓
Central Complex (280 neurons) 
    → ~264 spikes (strong recurrent)
    ↓
Mushroom Body (216 neurons) 
    → ~214 spikes
    ↓
Lateral Horn (150 neurons) 
    → ~147 spikes
    ↓
Descending Neurons (50 neurons) 
    → ~46 spikes
    ↓
Action: LEFT/STAY/RIGHT
```

**Key insight**: The high threshold (-40mV) and leak (-60mV) require ~500,000 pA to spike. Synaptic weights (mean ~100 nS after conn_scale=100) with driving force ~60mV produce ~6000 pA/synapse. With ~40 synapses from optic_lobes→central_complex, total I_syn ~240,000 pA - sufficient to reach threshold.

---

## Known Limitations

1. **Activity dies out** without sustained input or recurrent excitation
2. **No long-term memory** - weights don't persist without checkpointing
3. **Neuromodulation not fully integrated** into closed-loop
4. **Camera demo** requires physical camera (not available headless)
5. **Full 70k neuron config** not yet tested end-to-end

---

## Future Work

1. Strengthen recurrent connections for sustained activity
2. Integrate dopamine-gated STDP in closed-loop
3. Add eligibility traces for delayed reward
4. Scale to full connectome
5. Implement multi-game curriculum learning