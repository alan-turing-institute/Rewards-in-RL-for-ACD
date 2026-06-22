# Beyond Rewards in RL for Cyber Defence

Code for the ICML 2026 paper *"Beyond Rewards in Reinforcement Learning for Cyber Defence"*.

This repository contains the training and evaluation code used to run all experiments in the paper, covering two cyber-defence environments:

- **Yawning Titan (YT)** - an abstract, graph-based cyber-security simulation developed by Dstl.
- **miniCAGE** - a lightweight wrapper around the CybORG++ environment developed at the Alan Turing Institute.

---

## Repository layout

```
Rewards-in-RL-for-ACD/
├── yawning_titan_training/          # YT experiment code
│   ├── Training/
│   │   ├── parallel_training.py     ← main entry point: edit config, then run this
│   │   ├── experiment_runner.py
│   │   ├── yawning_titan_run_wandb.py
│   │   ├── hyperparams.py           ← tuned hyperparameters per reward function
│   │   └── Minimal_network_gamemode.json
│   ├── Evaluation/
│   │   ├── Evaluation_score.py      ← per-model evaluation (GT score, CVaR)
│   │   ├── Confidence_interval.py   ← 95% CIs across agents
│   │   └── Reliability_Evaluation.py ← IQR reliability metrics (requires W&B)
│   └── Networks/
│       ├── N_node_generator.py      ← linear chain networks (paper main experiments)
│       └── Diamond_network.py       ← diamond topology (optional)
│
├── mini_cage_training/              # miniCAGE experiment code
│   ├── Training/
│   │   ├── SB3_training.py          ← main entry point for miniCAGE
│   │   └── single_agent_gym_wrapper.py
│   ├── mini_CAGE/                   ← the miniCAGE environment (self-contained)
│   └── setup.py                     ← installs mini_cage_training as 'CybORG_plus_plus'
│
├── YAWNING-TITAN/                   # modified copy of the YT library (see CHANGES.md)
├── environments/                    # conda environment files
│   ├── yt_macos.yml
│   ├── yt_linux.yml
│   ├── minicage_macos.yml
│   └── minicage_linux.yml
└── docker/                          # Docker (recommended for full reproducibility)
    ├── Dockerfile.yt
    ├── Dockerfile.minicage
    └── docker-compose.yml
```

---

## Quick start - Docker (recommended)

Docker removes all environment compatibility issues across macOS, Linux, and Windows.

```bash
# Clone the repo
git clone https://github.com/alan-turing-institute/Rewards-in-RL-for-ACD.git
cd Rewards-in-RL-for-ACD

# --- Yawning Titan experiments ---
docker build -f docker/Dockerfile.yt -t yt-rewards .
# Interactive shell:
docker run --rm -it -v $(pwd)/results:/repo/results yt-rewards bash
# Inside the container, edit the config in parallel_training.py then:
python parallel_training.py

# --- miniCAGE experiments ---
docker build -f docker/Dockerfile.minicage -t minicage-rewards .
docker run --rm -it -v $(pwd)/results:/repo/results minicage-rewards bash
# Inside the container:
python SB3_training.py
```

Results are written to `./results/` on your host machine (mounted into the container).

---

## Quick start - Conda

**Yawning Titan (macOS)**
```bash
conda env create -f environments/yt_macos.yml
conda activate yt_rewards_macos
pip install --no-deps -e ./YAWNING-TITAN
pip install --no-deps -e .
# Install rl-reliability-metrics (needed for Reliability_Evaluation.py only):
pip install tensorflow gin-config absl-py
pip install --no-deps git+https://github.com/google-research/rl-reliability-metrics.git
```

**Yawning Titan (Linux)**
```bash
conda env create -f environments/yt_linux.yml
conda activate yt_rewards_intel_linux
pip install --no-deps -e ./YAWNING-TITAN
pip install --no-deps -e .
pip install tensorflow gin-config absl-py
pip install --no-deps git+https://github.com/google-research/rl-reliability-metrics.git
```

**miniCAGE (macOS)**
```bash
conda env create -f environments/minicage_macos.yml
conda activate miniCAGE_rewards_macos
# SB3 2.3.2 is required (newer than what the conda file pins):
pip install stable-baselines3==2.3.2
# Install the miniCAGE package so imports resolve correctly:
pip install -e ./mini_cage_training
```

**miniCAGE (Linux)**
```bash
conda env create -f environments/minicage_linux.yml
conda activate miniCAGE_rewards_linux
pip install stable-baselines3==2.3.2
pip install -e ./mini_cage_training
```

---

## Running Yawning Titan experiments

1. **Edit the config** at the top of `yawning_titan_training/Training/parallel_training.py`:

   ```python
   WANDB_PROJECT = "your-project-name"   # your W&B project
   WANDB_ENTITY  = "your-wandb-entity"   # your W&B username or team

   NET_SHAPE         = ['linear']         # 'linear' or 'diamond'
   NODE_COMBINATIONS = [5, 10]            # network sizes
   REWARD_FUNCTIONS  = ['scaffolded', 'complex_dense', 'simple_pos_neg',
                         'simple_positive', 'simple_negative']
   ORDER             = ["Red_Blue", "Blue_Red", "Balanced"]
   ACTION_SPACE_SET  = ["simple_action_space", "decoy_action_space"]
   ALGO              = ["PPO"]            # "PPO" or "DQN"
   NO_RUNS           = 10                 # independent seeds per config
   ```

2. **Run training** from the `Training/` directory:

   ```bash
   cd yawning_titan_training/Training
   python parallel_training.py

   # Long runs - use nohup:
   mkdir -p ../../nohup
   nohup python -u parallel_training.py > ../../nohup/run1.log 2>&1 &
   ```

3. **Evaluate** trained models:

   ```bash
   # Score-based evaluation (GT score, CVaR)
   cd yawning_titan_training/Evaluation
   python Evaluation_score.py

   # Reliability metrics (IQR - requires W&B login)
   python Reliability_Evaluation.py

   # Confidence intervals across agents
   python Confidence_interval.py
   ```

---

## Running miniCAGE experiments

1. **Edit the config** at the top of `mini_cage_training/Training/SB3_training.py`:

   ```python
   WANDB_PROJECT = "your-project-name"
   WANDB_ENTITY  = "your-wandb-entity"
   GROUP_NAME    = "SB3_PPO_simple_default_2500000"

   NUM_RUNS        = 25
   TOTAL_TIMESTEPS = 2_500_000
   ```

2. Choose PPO or DQN by commenting/uncommenting the relevant model block in the `train_worker` function.

3. **Run training**:

   ```bash
   cd mini_cage_training/Training
   python SB3_training.py
   ```

---

## Notes on reward functions

| Name in code | Description in paper |
|---|---|
| `simple_positive` | Positive Rewards |
| `simple_negative` | Negative Rewards |
| `simple_pos_neg` | Simple Positive and Negative Rewards |
| `scaffolded` | Dense Negative Rewards |
| `complex_dense` | Complex Dense Rewards |

The reward function implementations are in `YAWNING-TITAN/src/yawning_titan/envs/generic/core/reward_functions.py`. See `YAWNING-TITAN/CHANGES.md` for the full list of modifications to the upstream YT codebase.

---

## Acknowledgements

- **Yawning Titan** - Dstl, MIT Licence. [github.com/dstl/YAWNING-TITAN](https://github.com/dstl/YAWNING-TITAN)
  Our copy includes targeted modifications; see `YAWNING-TITAN/CHANGES.md`.
- **CybORG++** / **miniCAGE** - Alan Turing Institute. [github.com/alan-turing-institute/CybORG_plus_plus](https://github.com/alan-turing-institute/CybORG_plus_plus)
- **rl-reliability-metrics** - Google Research, Apache 2.0. [github.com/google-research/rl-reliability-metrics](https://github.com/google-research/rl-reliability-metrics)
- **Stable-Baselines3** - [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
