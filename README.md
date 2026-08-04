# Beyond Rewards in RL for Cyber Defence Codebase

This is the codebase that accompanies the ICML 2026 paper *[“Beyond Rewards in Reinforcement Learning for Cyber 
Defence.”](https://arxiv.org/pdf/2602.04809)*
The work explores how different reward‐function designs affect **autonomous cyber‑defence (ACD)** agents trained in 
the **[Yawning Titan](https://github.com/dstl/YAWNING-TITAN)** cyber‑gym and the **[MiniCAGE](https://github.com/alan-turing-institute/CybORG_plus_plus/tree/main/mini_CAGE)** environment from the 
CybORG++ toolkit.

This repository contains the training and evaluation code used to run all experiments in the paper, covering two 
aforementioned cyber-defence environments:

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
│   │   ├── Reliability_Evaluation.py ← IQR reliability metrics (requires W&B)
│   │   └── utils.py                 ← shared evaluation helpers
│   └── Networks/
│       ├── N_node_generator.py      ← linear chain networks (paper main experiments)
│       └── Diamond_network.py       ← circular topology (optional)
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
cd /repo/yawning_titan_training/Training
python parallel_training.py

# --- miniCAGE experiments ---
docker build -f docker/Dockerfile.minicage -t minicage-rewards .
docker run --rm -it -v $(pwd)/results:/repo/results minicage-rewards bash
# Inside the container:
cd /repo/mini_cage_training/Training
python SB3_training.py
```

Results are written to `./results/` on your host machine (mounted into the container):
Yawning Titan writes to `results/train_log/`, and miniCAGE writes checkpoints to
`results/PPO_models/` and TensorBoard logs to `results/dqn_mini_cage_tensorboard/`.

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
# Install the miniCAGE package so imports resolve correctly:
pip install -e ./mini_cage_training
```

**miniCAGE (Linux)**
```bash
conda env create -f environments/minicage_linux.yml
conda activate miniCAGE_rewards_linux
pip install -e ./mini_cage_training
```

---

## Running Yawning Titan experiments

1. **Edit the config** at the top of `yawning_titan_training/Training/parallel_training.py`:

   ```python
   USE_WANDB     = True                  # Weights & Biases logging on/off
   WANDB_PROJECT = "your-project-name"   # your W&B project
   WANDB_ENTITY  = "your-wandb-entity"   # your W&B username or team

   NET_SHAPE         = ['linear']         # 'linear' or 'diamond'
   NODE_COMBINATIONS = [5, 10]            # network sizes
   REWARD_FUNCTIONS  = ['dense_negative', 'complex_dense_negative', 'simple_pos_neg',
                         'simple_positive', 'simple_negative']
   ORDER             = ["Red_Blue", "Blue_Red", "Balanced"]
   ACTION_SPACE_SET  = ["simple_action_space", "decoy_action_space"]
   ALGO              = ["PPO"]            # "PPO" or "DQN"
   NO_RUNS           = 10                 # independent seeds per config
   ```

   **Weights & Biases** logging is controlled by `USE_WANDB`. We recommend
   leaving it on to track your runs (metrics, configs, and saved models). You can run
   `wandb login` (or set `WANDB_API_KEY`) once beforehand. Set `USE_WANDB = False`
   to train without a W&B account or login, e.g. for a quick local trial.

2. **Run training** from the `Training/` directory:

   ```bash
   cd yawning_titan_training/Training
   python parallel_training.py

   # Long runs - use nohup:
   mkdir -p ../../nohup
   nohup python -u parallel_training.py > ../../nohup/run1.log 2>&1 &
   ```

3. **Evaluate** trained models. Run these from the `Evaluation/` directory, in order:

   ```bash
   cd yawning_titan_training/Evaluation

   # (a) Score-based evaluation (ground-truth score + CVaR) for one set of agents.
   #     Point --agent_dir at a trained PPO folder under results/train_log/ and pass
   #     the matching --reward_type. Per-agent scores are written to results/eval_log/.
   python Evaluation_score.py \
       --agent_dir ../../results/train_log/simple_action_space/simple_positive/5_nodes/PPO \
       --reward_type "Positive Rewards"

   # (b) Confidence intervals across agents. Reads the scores written by (a) from
   #     results/eval_log/ (override the location with the EVAL_ROOT env var).
   python Confidence_interval.py

   # (c) Reliability metrics (IQR). Pulls run histories from Weights & Biases, so it
   #     only works for runs trained with W&B enabled (USE_WANDB = True). Run
   #     `wandb login` first, then set `entity` and `project` at the top of the script.
   python Reliability_Evaluation.py
   ```

   `--reward_type` maps the paper's reward name to the trained reward function:
   1. simple_positive
   2. simple_negative
   3. simple_pos_neg
   4. dense_negative
   5. complex_dense_negative

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

   As with Yawning Titan, W&B logging is on by default (`USE_WANDB = True` in the script) so run `wandb login` first, 
   or set `USE_WANDB = False` to train without a W&B account.

2. Choose PPO or DQN by commenting/uncommenting the relevant model block in the `train_worker` function.

3. **Run training**:

   ```bash
   cd mini_cage_training/Training
   python SB3_training.py
   ```

## Citation

If you use this code or build on our work, please cite the paper:

```bibtex
@inproceedings{beyond_rewards_acd_2026,
  title     = {Beyond Rewards in Reinforcement Learning for Cyber Defence},
  author    = {Bates, E. and Hicks, C. and Mavroudis, V.},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {To appear. Preprint: arXiv:2602.04809}
}
```

---

## Acknowledgements

- **Yawning Titan** - Dstl, MIT Licence. [github.com/dstl/YAWNING-TITAN](https://github.com/dstl/YAWNING-TITAN)
  Our copy includes targeted modifications; see `YAWNING-TITAN/CHANGES.md`.
- **CybORG++** / **miniCAGE** - Alan Turing Institute. [github.com/alan-turing-institute/CybORG_plus_plus](https://github.com/alan-turing-institute/CybORG_plus_plus)
- **rl-reliability-metrics** - Google Research, Apache 2.0. [github.com/google-research/rl-reliability-metrics](https://github.com/google-research/rl-reliability-metrics)
- **Stable-Baselines3** - [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
