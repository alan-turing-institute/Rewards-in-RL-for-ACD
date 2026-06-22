"""
SB3_training.py — launch N PPO runs in parallel (one process each)

"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from multiprocessing import Process, set_start_method
from pathlib import Path
from typing import Optional
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from single_agent_gym_wrapper import MiniCageBlue
from stable_baselines3.common.callbacks import BaseCallback

# ═══════════════════════════════════════════════════════════════════════
# Training Config
# ═══════════════════════════════════════════════════════════════════════
NUM_RUNS: int = 25
TOTAL_TIMESTEPS: int = 2_500_000

USE_WANDB: bool = True           # flip to False to disable W&B logging
USE_TENSORBOARD: bool = True     # if True, each run gets its own TB dir

WANDB_PROJECT: str = ""           # Add your W&B project name here
WANDB_ENTITY: str | None = ""    # Add your W&B entity/team name here (or None for default)
GROUP_NAME: str = f"SB3_PPO_simple_default_{TOTAL_TIMESTEPS}"  # descriptive group label

# These hyper-parameters are taken from the cardiff solution
LEARNING_RATE: float = 0.002
GAMMA: float = 0.99
CLIP_RANGE: float = 0.2
N_EPOCHS: int = 6

SAVE_DIR: Path = Path(f"./PPO_models/{GROUP_NAME}")
SAVE_DIR.mkdir(exist_ok=True)

def make_env(seed: Optional[int] = None):
    """Factory that returns a **monitored** MiniCageBlue env."""
    env = MiniCageBlue(red_policy="bline", max_steps=100, remove_bugs=True)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)

    env = Monitor(env)
    return env

class HostsCompromisedCallback(BaseCallback):
    """
    Logs two metrics every rollout (≈ at the same timesteps SB3 logs rewards):
      • custom/GT_score      – max(after_red, after_blue)
      • custom/Step_score    – after_blue only
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._gt_buf: list[float] = []
        self._blue_buf: list[float] = []

    # called every environment step
    def _on_step(self) -> bool:
        # infos is a list (one dict per vec-env slot)
        for info in self.locals.get("infos", []):
            red_val = info.get("Hosts_compromised_after_red")
            blue_val = info.get("Hosts_compromised_after_blue")

            # ── GT_score buffer ──────────────────────────────────────────
            if (red_val is not None) or (blue_val is not None):
                # choose the larger *of the values that exist*
                gt = max(v for v in (red_val, blue_val) if v is not None)
                self._gt_buf.append(gt)

            # ── Step_score buffer ───────────────────────────────────────
            if blue_val is not None:
                self._blue_buf.append(blue_val)

        return True  # keep training

    # called automatically at the end of each rollout
    def _on_rollout_end(self) -> None:
        if self._gt_buf:
            self.logger.record(
                "custom/GT_score",
                float(np.mean(self._gt_buf))
            )
            self._gt_buf.clear()

        if self._blue_buf:
            self.logger.record(
                "custom/Step_score",
                float(np.mean(self._blue_buf))
            )
            self._blue_buf.clear()

def train_worker(idx: int):
    """Launch a single PPO run (executed inside its own process)."""

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dqn_mini_cage_bline_{time_tag}_{idx}"

    # Make environment
    env = DummyVecEnv([lambda: make_env(seed=idx)])
    env = VecMonitor(env)

    # TensorBoard dir
    tb_dir: Optional[str]
    if USE_TENSORBOARD:
        tb_dir = f"./dqn_mini_cage_tensorboard/run_{idx}"
        os.makedirs(tb_dir, exist_ok=True)
    else:
        # create a temporary directory so SB3 still instantiates the writer
        tb_dir = tempfile.mkdtemp() if USE_WANDB else None

    # Initialise model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tb_dir,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        n_epochs=N_EPOCHS,
        seed=idx,  # unique seed per run
        device="auto",
    )

    # model = DQN(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=1,
    #     tensorboard_log=tb_dir,
    #     # learning_rate=LEARNING_RATE,
    #     # gamma=GAMMA,
    #     exploration_final_eps=0.005,
    #     buffer_size=200_000,
    #     # n_epochs=N_EPOCHS,
    #     seed=idx,  # unique seed per run
    #     device="auto",
    # )

    # Build callbacks
    callback_list = []

    callback_list.append(HostsCompromisedCallback())

    if USE_WANDB:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=GROUP_NAME,
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,  # TB writer exists, so sync it
            config=dict(
                algorithm="PPO",
                total_timesteps=TOTAL_TIMESTEPS,
                env="MiniCageBlue",
                seed=idx,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                clip_range=CLIP_RANGE,
                n_epochs=N_EPOCHS,
                exploration_final_eps=0.005,
                buffer_size=200_000,
            ),
        )

        callback_list.append(
            WandbCallback(
                gradient_save_freq=1_000,
                model_save_path=str(SAVE_DIR),
                verbose=0,
            )

        )

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list or None,
        log_interval=10,
    )

    # Save checkpoint
    ckpt_path = SAVE_DIR / f"{run_name}.zip"
    model.save(ckpt_path)

    if USE_WANDB:
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(str(ckpt_path))
        run.log_artifact(artifact)
        run.finish()

    print(f"Run {idx}: finished. Model saved to {ckpt_path}")


if __name__ == "__main__":
    # sleep
    # time.sleep(5400)
    time.sleep(12600)

    try:
        set_start_method("spawn")  # does nothing if already set
    except RuntimeError:
        pass


    START_IDX = 1
    processes: list[Process] = []
    for idx in range(START_IDX, START_IDX + NUM_RUNS):
        p = Process(target=train_worker, args=(idx,), daemon=False)
        p.start()
        processes.append(p)

    # Wait for all workers to complete
    for p in processes:
        p.join()

    print("\n All runs finished!")