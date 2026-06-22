"""
parallel_training.py — train multiple Yawning Titan reward-function variants in parallel.

Edit the CONFIGURATION block below, then run:

    cd yawning_titan_training/Training
    python parallel_training.py

    # or with nohup for long runs:
    nohup python -u parallel_training.py > ../../nohup/training_1.log 2>&1 &
"""

import multiprocessing
import os
import logging
import tempfile
import time
import datetime

from experiment_runner import run_experiment_with_logging
from hyperparams import REWARD_TO_HYPERPARAMS

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_SILENT"] = "True"
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to control what gets trained
# ═══════════════════════════════════════════════════════════════════

# W&B settings — set your project and entity (team) name here.
WANDB_PROJECT = "YT-Rewards-in-RL-for-ACD"
WANDB_ENTITY = ""   # your wandb username or team name; leave blank to use your default

# Where to save trained models (relative to the repo root, or absolute)
OUTPUT_LOCATION = "../../results"

# Network topology: 'linear' (chain) or 'diamond'
NET_SHAPE = ['linear']

# Number of nodes in the network
NODE_COMBINATIONS = [5, 10]

# Reward functions to train.
# Options: 'scaffolded', 'complex_dense', 'simple_pos_neg', 'simple_positive', 'simple_negative'
# Note: 'scaffolded' is the Dense Negative reward described in the paper.
REWARD_FUNCTIONS = ['scaffolded', 'complex_dense', 'simple_pos_neg', 'simple_positive', 'simple_negative']

# Steps per episode
N_STEPS = 100

NODE_VULNERABILITY = 1
RED_AGENT_SKILL = 1

# Agent turn order per episode. Options: 'Red_Blue', 'Blue_Red', 'Balanced'
# 'Balanced' alternates each episode.
ORDER = ["Red_Blue", "Blue_Red", "Balanced"]

EVAL_TYPE = "initial_eval"

# Action spaces to include.
# 'simple_action_space' — Blue cannot place decoys.
# 'decoy_action_space'  — Blue can place decoys.
ACTION_SPACE_SET = ["simple_action_space", "decoy_action_space"]

# Algorithms to use. Options: "PPO", "DQN"
ALGO = ["PPO"]

# Number of independent training runs per configuration
NO_RUNS = 10

# Max parallel processes. Reduce if memory is tight.
MAX_PROCESSES = 10

# GPU count (0 = CPU only)
GPU = 0

# ═══════════════════════════════════════════════════════════════════


if GPU > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(GPU))


def run_experiment_wrapper(*args, **kwargs):
    run_experiment_with_logging(*args, **kwargs)


def run_parallel_experiments(n_nodes, reward_function, order, action_set, net_shape, algorithm, timesteps):
    group_name = (
        f'{n_nodes}_Nodes_{algorithm}_{net_shape}_YT_{order}_'
        f'{reward_function}_{action_set}_Set_{N_STEPS}_Step_Episodes'
    )
    hyperparams = REWARD_TO_HYPERPARAMS.get(reward_function, REWARD_TO_HYPERPARAMS.get('missing_key', {}))

    with tempfile.TemporaryDirectory() as temp_dir:
        processes = []

        for i in range(1, NO_RUNS + 1):
            trial_name = f'sb3_EpLen{N_STEPS}_{order}_Skill1_Vul{NODE_VULNERABILITY}_run_{i}'
            logging.info(f"Starting process for trial {i}: {trial_name}")

            if GPU > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str((i - 1) % GPU)

            p = multiprocessing.Process(
                target=run_experiment_wrapper,
                args=(
                    trial_name, group_name, EVAL_TYPE, n_nodes, reward_function, N_STEPS,
                    NODE_VULNERABILITY, RED_AGENT_SKILL, timesteps, temp_dir, action_set,
                    hyperparams, order, net_shape, algorithm,
                    WANDB_PROJECT, WANDB_ENTITY, OUTPUT_LOCATION,
                ),
            )
            processes.append(p)
            p.start()

            while sum(1 for proc in processes if proc.is_alive()) >= MAX_PROCESSES:
                time.sleep(0.1)

        for p in processes:
            p.join()

        logging.info(f"Completed all experiments for {n_nodes} nodes with {reward_function}.")


def _timesteps_for_nodes(n_nodes: int) -> int:
    """Scale training length with network size, matching paper settings."""
    return {2: 500_000, 5: 1_000_000, 10: 2_000_000, 20: 2_000_000, 50: 2_500_000}.get(
        n_nodes, 1_000_000
    )


if __name__ == '__main__':
    for algorithm in ALGO:
        for net_shape in NET_SHAPE:
            for action_set in ACTION_SPACE_SET:
                for n_nodes in NODE_COMBINATIONS:
                    timesteps = _timesteps_for_nodes(n_nodes)
                    for reward_function in REWARD_FUNCTIONS:
                        for order in ORDER:
                            print(
                                f"[{datetime.datetime.now():%H:%M:%S}] Starting {algorithm} | "
                                f"{n_nodes}-node {net_shape} | {reward_function} | {order} | "
                                f"{timesteps:,} steps"
                            )
                            run_parallel_experiments(
                                n_nodes, reward_function, order, action_set,
                                net_shape, algorithm, timesteps,
                            )
                            print(f"[{datetime.datetime.now():%H:%M:%S}] Finished {reward_function} / {order}")
