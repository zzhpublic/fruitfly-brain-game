# 🧠 Fruit Fly Brain Game - Drosophila Neural Simulation

A computational neuroscience project simulating the *Drosophila melanogaster* (fruit fly) brain to play simple games through biologically-inspired neural networks.

## 🎯 Project Overview

This project creates a spiking neural network (SNN) model of the fruit fly brain (~100,000 neurons) and trains it to play simple games using:
- **Connectome data** from FlyWire/FAFB datasets
- **Neuromodulator systems** (dopamine, octopamine, serotonin)
- **Reinforcement learning** via synaptic plasticity (STDP + dopamine modulation)
- **Real-time visualization** of neural activity

## 🧬 Biological Basis

| Brain Region | Neurons | Function | Game Role |
|-------------|---------|----------|-----------|
| **Optic Lobes** | ~60,000 | Visual processing | Game screen input |
| **Mushroom Body** | ~2,500 | Learning & memory | Policy/value network |
| **Central Complex** | ~3,000 | Navigation, motor control | Action selection |
| **Lateral Horn** | ~1,500 | Innate behaviors | Reward prediction |
| **Neuromodulatory** | ~500 | Dopamine/OA/5HT | RL signals |

## 🎮 Supported Games

1. **Pong** - Visual tracking + paddle control
2. **Maze Navigation** - Spatial memory + pathfinding
3. **Odor Tracking** - Chemotaxis simulation
4. **Looming Escape** - Threat detection + evasion
5. **Custom RL Environments** - Gymnasium compatible

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download connectome data (optional, uses synthetic if unavailable)
python scripts/download_connectome.py

# Train on Pong
python train.py --game pong --episodes 10000

# Visualize neural activity
python visualize.py --checkpoint checkpoints/pong_best.pt

# Play against the fly
python play.py --game pong --human
```

## 📁 Project Structure

```
fruitfly-brain-game/
├── config/                 # Configuration files
│   ├── connectome.yaml     # Brain region definitions
│   ├── training.yaml       # Hyperparameters
│   └── games/              # Game-specific configs
├── data/                   # Connectome & dataset storage
├── src/
│   ├── connectome/         # Connectome loading & processing
│   ├── neurons/            # Neuron models (LIF, ALIF, etc.)
│   ├── synapses/           # Synapse models (STDP, plasticity)
│   ├── neuromod/           # Dopamine, octopamine, serotonin
│   ├── networks/           # Brain region networks
│   ├── games/              # Game environments
│   ├── training/           # RL algorithms
│   └── visualization/      # Real-time brain activity viz
├── notebooks/              # Analysis notebooks
├── scripts/                # Utility scripts
├── checkpoints/            # Model checkpoints
└── logs/                   # Training logs
```

## 🔬 Key Features

- **Biologically-accurate**: Uses real fly connectome topology
- **Neuromodulated RL**: Dopamine-gated STDP for credit assignment
- **Multi-timescale**: Fast spiking (ms) + slow learning (min/hr)
- **Interpretability**: Every neuron maps to real fly neuron type
- **GPU accelerated**: Brian2 / JAX / PyTorch backends

## 📚 References

- FlyWire Connectome: `Nature 2023` / `eLife 2024`
- FAFB EM Volume: `Zheng et al. 2018`
- Drosophila RL: `Aso et al. 2014`, `Cohn et al. 2015`
- SNN RL: `Bellec et al. 2020`, `Yin et al. 2021`

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines.

## 📄 License

MIT License - See `LICENSE` for details.
