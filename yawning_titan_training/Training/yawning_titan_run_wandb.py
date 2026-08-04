from __future__ import annotations

import json
import os.path
import pickle
from typing import List, Dict, Any
import pathlib
import shutil
from datetime import datetime
from logging import Logger, getLogger
from typing import Dict, Final, List, Optional, Union
from uuid import uuid4
from pprint import pprint
from collections import defaultdict

import yaml
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy as PPOMlp
from stable_baselines3.dqn import MlpPolicy as DQNMlp
from wandb.integration.sb3 import WandbCallback

from yawning_titan import AGENTS_DIR, PPO_TENSORBOARD_LOGS_DIR
from yawning_titan.agents.fixed_red import FixedRedAgent
from yawning_titan.agents.nsa_red import NSARed
from yawning_titan.agents.simple_blue import SimpleBlue
from yawning_titan.agents.sinewave_red import SineWaveRedAgent
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.exceptions import YawningTitanRunError
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.game_modes.game_mode_db import default_game_mode
from yawning_titan.networks.network import Network
from yawning_titan.networks.network_db import default_18_node_network
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ppo.policies import ActorCriticPolicy
import numpy as np
import wandb
import torch as th

_LOGGER = getLogger(__name__)
_LOGGER.setLevel(40)

class CriticBiasedPolicy_NegativeBias(PPOMlp):
    """Customise critic bias to -ve value for ablation studies"""
    def _build(self, lr_schedule):
        # Let SB3 build everything first
        super()._build(lr_schedule)

        # Now value_net exists, so we can change its bias
        with th.no_grad():
            self.value_net.bias.fill_(-10)

class EntropyDecayCallback(BaseCallback):
    """Used to anneal PPO ent_coeff over time"""
    """Decays from 0.02 to 0 over first 50% of training"""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        p = self.num_timesteps / self.total_timesteps

        if p <= 0.7:
            self.model.ent_coef = 0.012 * (1 - p / 0.7)
        else:
            self.model.ent_coef = 0.0

        return True

class CustomEvalCallback(BaseCallback):
    """
    A single callback that does two things per episode:
      1) Logs a single "std_reward" (negative #compromised nodes) using final step's end_state
      2) Logs "avg_reward_delta" for the episode by comparing consecutive step rewards
    """

    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env

        self.episode_std_rewards = []
        self.reward_deltas: List[float] = []
        self._last_reward: Optional[float] = None

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]

        if isinstance(rewards, (list, np.ndarray)):
            step_reward = float(np.mean(rewards))
        else:
            step_reward = float(rewards)

        if self._last_reward is not None:
            self.reward_deltas.append(abs(step_reward - self._last_reward))
        self._last_reward = step_reward

        if np.any(dones):
            final_info = infos[0]
            end_state = final_info.get("end_state", {})
            no_compromised_nodes = sum(1 for v in end_state.values() if v == 1)
            std_reward = -no_compromised_nodes

            self.episode_std_rewards.append(std_reward)

            avg_reward_delta = float(np.mean(self.reward_deltas)) if self.reward_deltas else 0.0

            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/avg_reward_delta", avg_reward_delta)

            self.reward_deltas.clear()
            self._last_reward = None

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout. This can happen before the episode
        finishes if SB3 decides to collect a new rollout. Not strictly needed
        here, so we leave it blank.
        """
        pass

    def _on_training_end(self) -> None:
        """
        Called after the entire training is done. We can compute an overall average
        std_reward if we wish. We'll do last 20% as per your existing logic:
        """
        if self.episode_std_rewards:
            # Compute average of last 20% of episodes
            last_20pct = max(1, len(self.episode_std_rewards) // 5)  # if n < 5, just use all episodes
            last20_subset = self.episode_std_rewards[-last_20pct:]
            mean_std_rew_score = sum(last20_subset) / len(last20_subset)
            self.logger.record("eval/last20pct_std_reward_mean", mean_std_rew_score)
            self.logger.dump(self.num_timesteps)
            # wandb.log({"eval/last20pct_std_reward_mean": mean_recent_gt_score})

class GTScoreCallback(BaseCallback):
    """
    Logs a single "eval/GTScore" each time an episode ends (done=True),
    where GTScore is the sum over all steps in the episode of the maximum compromised count.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_gt_scores = []  # Store GTScore per episode
        self.current_episode_gtscore = 0  # Accumulator for the current episode

    # def _on_rollout_start(self) -> None:
    #     # Reset the accumulator at the beginning of each rollout/episode.
    #     self.current_episode_gtscore = 0

    def _on_step(self) -> bool:
        # Get the info for the current step; assuming single env, so index 0.
        info = self.locals["infos"][0]
        
        # Extract per-step compromised counts from both mid and end states.
        mid_compromised_states = info.get("mid_step_info", {}).get("mid_step", {}).get("mid_state_compromised", {})
        mid_compromised_count = sum(mid_compromised_states.values())
        # print(f"Mid compromised states: {mid_compromised_states}")
        
        compromised_states = info.get("end_state", {})
        compromised_count = sum(compromised_states.values())
        # print(f"End compromised states: {compromised_states}")
        
        # For this step, use the maximum compromised count.
        step_max = max(mid_compromised_count, compromised_count)
        # Accumulate the value.
        self.current_episode_gtscore += step_max
        # print(f"Step max compromised count: {step_max}")
        
        dones = self.locals["dones"]
        if np.any(dones):
            # At the end of the episode, log the accumulated GTScore.
            gt_score = -self.current_episode_gtscore  # Negative if you want higher scores for fewer compromises.
            # print(f"GTScore for this episode: {gt_score}")
            self.episode_gt_scores.append(gt_score)
            self.logger.record("eval/GTScore", gt_score)
            
            # Reset for the next episode.
            self.current_episode_gtscore = 0

        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """
        After training, compute and log the average GTScore for the last 20% of episodes.
        """
        if self.episode_gt_scores:
            # Compute average over the last 20% of episodes (or at least one).
            last_20pct = max(1, len(self.episode_gt_scores) // 5)
            last20_subset = self.episode_gt_scores[-last_20pct:]
            mean_gt_score = sum(last20_subset) / len(last20_subset)
            self.logger.record("eval/last20pct_GTScore_mean", mean_gt_score)
            self.logger.dump(self.num_timesteps)


class DiagnosticPPO(PPO):
    """PPO with per-training-step gradient diagnostics and optional separate clipping.

    Logs gradient norms (before clipping) broken down by network component:
    policy head, value head, and shared layers.

    When separate_grad_clip=True, clips policy and value parameter gradients
    independently so that large value gradients cannot steal the policy's
    gradient budget through the joint norm.
    """

    def __init__(self, *args, separate_grad_clip=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.separate_grad_clip = separate_grad_clip

    def _build_param_component_map(self):
        """Map each parameter to its network component."""
        self._param_component = {}
        for p in self.policy.mlp_extractor.shared_net.parameters():
            self._param_component[id(p)] = 'shared'
        for p in self.policy.mlp_extractor.policy_net.parameters():
            self._param_component[id(p)] = 'policy'
        for p in self.policy.mlp_extractor.value_net.parameters():
            self._param_component[id(p)] = 'value'
        for p in self.policy.action_net.parameters():
            self._param_component[id(p)] = 'policy'
        for p in self.policy.value_net.parameters():
            self._param_component[id(p)] = 'value'

    def train(self) -> None:
        if not hasattr(self, '_param_component'):
            self._build_param_component_map()

        original_clip = th.nn.utils.clip_grad_norm_
        grad_records = []
        param_component = self._param_component
        separate = self.separate_grad_clip

        def _instrumented_clip(parameters, max_norm, norm_type=2.0, **kwargs):
            if isinstance(parameters, th.Tensor):
                params = [parameters]
            else:
                params = list(parameters)

            # Classify params and compute pre-clip norms
            policy_params = []
            value_params = []
            norms = {'total': [], 'policy': [], 'value': [], 'shared': []}
            for p in params:
                comp = param_component.get(id(p), 'shared')
                if comp == 'value':
                    value_params.append(p)
                else:  # 'policy', 'shared', or unclassified
                    policy_params.append(p)
                if p.grad is not None:
                    g_norm = th.norm(p.grad.detach(), float(norm_type))
                    norms['total'].append(g_norm)
                    norms[comp].append(g_norm)

            record = {}
            for key, vals in norms.items():
                record[key] = (
                    th.norm(th.stack(vals), float(norm_type)).item() if vals else 0.0
                )
            grad_records.append(record)

            if separate:
                # Clip each group independently — value can't steal policy budget
                if policy_params:
                    original_clip(policy_params, max_norm, norm_type=norm_type, **kwargs)
                if value_params:
                    original_clip(value_params, max_norm, norm_type=norm_type, **kwargs)
                return th.tensor(record['total'])
            else:
                return original_clip(params, max_norm, norm_type=norm_type, **kwargs)

        th.nn.utils.clip_grad_norm_ = _instrumented_clip
        try:
            super().train()
        finally:
            th.nn.utils.clip_grad_norm_ = original_clip

        if grad_records:
            for key in ('total', 'policy', 'value', 'shared'):
                vals = [r[key] for r in grad_records]
                self.logger.record(
                    f"diagnostics/grad_norm_{key}_mean", float(np.mean(vals))
                )
                self.logger.record(
                    f"diagnostics/grad_norm_{key}_max", float(np.max(vals))
                )
            ratios = [
                r['value'] / r['total'] if r['total'] > 0 else 0.0
                for r in grad_records
            ]
            self.logger.record(
                "diagnostics/value_grad_budget_ratio", float(np.mean(ratios))
            )
            # Joint clip fraction (how often total norm exceeds threshold)
            clipped = sum(
                1 for r in grad_records if r['total'] > self.max_grad_norm
            )
            self.logger.record(
                "diagnostics/grad_clip_fraction", clipped / len(grad_records)
            )
            # Per-group clip fractions
            policy_clipped = sum(
                1 for r in grad_records if r['policy'] > self.max_grad_norm
            )
            value_clipped = sum(
                1 for r in grad_records if r['value'] > self.max_grad_norm
            )
            self.logger.record(
                "diagnostics/policy_grad_clip_fraction",
                policy_clipped / len(grad_records)
            )
            self.logger.record(
                "diagnostics/value_grad_clip_fraction",
                value_clipped / len(grad_records)
            )


class TabularQAgent:
    """
    Classical tabular Q-learning with epsilon-greedy exploration.

    State values are discretised from the continuous observation vector into a
    finite key, allowing sparse table-based learning without neural networks.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_fraction: float = 0.8,
        state_bins: int = 10,
        seed: int = 42,
        verbose: int = 0,
    ):
        self.env = env
        self.alpha = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_fraction = float(max(1e-6, epsilon_decay_fraction))
        self.state_bins = int(max(1, state_bins))
        self.verbose = verbose

        self.num_actions = int(self.env.action_space.n)
        self.rng = np.random.default_rng(seed)
        self.num_timesteps = 0
        self.epsilon = self.epsilon_start
        self.q_table = defaultdict(self._new_action_values)

    def _new_action_values(self) -> np.ndarray:
        # Use for SP not Ablated-SP
        #return np.zeros(self.num_actions, dtype=np.float32)
    
        # Use for Ablated-SP
        return np.full(self.num_actions, -50)

    def _state_to_key(self, observation: np.ndarray) -> tuple:
        obs = np.asarray(observation, dtype=np.float32).flatten()
        obs = np.clip(obs, 0.0, 1.0)
        discretised = np.rint(obs * self.state_bins).astype(np.int16)
        return tuple(discretised.tolist())

    def _epsilon_for_step(self, current_step: int, total_steps: int) -> float:
        decay_steps = max(1, int(total_steps * self.epsilon_decay_fraction))
        progress = min(1.0, current_step / decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _select_action(self, state_key: tuple, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.env.action_space.sample())

        q_values = self.q_table[state_key]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(self.rng.choice(best_actions))

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        n_eval_episodes: int = 1,
        progress_bar: bool = False,
        **kwargs,
    ):
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        prev_step_reward: Optional[float] = None
        episode_reward_deltas: List[float] = []
        current_episode_gtscore = 0
        episode_std_rewards: List[float] = []
        episode_gt_scores: List[float] = []

        for step in range(int(total_timesteps)):
            self.num_timesteps += 1
            self.epsilon = self._epsilon_for_step(step, int(total_timesteps))

            state_key = self._state_to_key(obs)
            action = self._select_action(state_key, self.epsilon)

            next_obs, reward, done, info = self.env.step(action)
            next_state_key = self._state_to_key(next_obs)
            reward = float(reward)

            td_target = reward
            if not done:
                td_target += self.gamma * float(np.max(self.q_table[next_state_key]))

            td_error = td_target - float(self.q_table[state_key][action])
            self.q_table[state_key][action] += self.alpha * td_error

            if prev_step_reward is None:
                reward_delta = 0.0
            else:
                reward_delta = abs(reward - prev_step_reward)
            episode_reward_deltas.append(reward_delta)
            prev_step_reward = reward

            mid_compromised_states = info.get("mid_step_info", {}).get("mid_step", {}).get("mid_state_compromised", {})
            mid_compromised_count = int(sum(mid_compromised_states.values()))
            end_state = info.get("end_state", {})
            end_compromised_count = int(sum(end_state.values()))
            current_episode_gtscore += max(mid_compromised_count, end_compromised_count)

            episode_reward += reward
            episode_length += 1

            if done:
                std_reward = -end_compromised_count
                avg_reward_delta = float(np.mean(episode_reward_deltas)) if episode_reward_deltas else 0.0
                gt_score = -current_episode_gtscore

                episode_std_rewards.append(std_reward)
                episode_gt_scores.append(gt_score)

                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/episode_reward": episode_reward,
                            "train/episode_length": episode_length,
                            "train/epsilon": self.epsilon,
                            "train/q_table_states": len(self.q_table),
                            "eval/std_reward": std_reward,
                            "eval/avg_reward_delta": avg_reward_delta,
                            "eval/GTScore": gt_score,
                        },
                        step=self.num_timesteps,
                    )

                obs = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                prev_step_reward = None
                episode_reward_deltas = []
                current_episode_gtscore = 0
            else:
                obs = next_obs

        if wandb.run is not None:
            if episode_std_rewards:
                last_20pct_std = max(1, len(episode_std_rewards) // 5)
                wandb.log(
                    {
                        "eval/last20pct_std_reward_mean": float(np.mean(episode_std_rewards[-last_20pct_std:])),
                    },
                    step=self.num_timesteps,
                )
            if episode_gt_scores:
                last_20pct_gt = max(1, len(episode_gt_scores) // 5)
                wandb.log(
                    {
                        "eval/last20pct_GTScore_mean": float(np.mean(episode_gt_scores[-last_20pct_gt:])),
                    },
                    step=self.num_timesteps,
                )

        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        state_key = self._state_to_key(observation)
        epsilon = 0.0 if deterministic else self.epsilon
        action = self._select_action(state_key, epsilon)
        return action, state

    def get_env(self):
        return self.env

    def save(self, path: str):
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_fraction": self.epsilon_decay_fraction,
            "state_bins": self.state_bins,
            "epsilon": self.epsilon,
            "num_timesteps": self.num_timesteps,
            "q_table": dict(self.q_table),
        }
        with open(path, "wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str, env, verbose: int = 0, seed: int = 42):
        with open(path, "rb") as file:
            payload = pickle.load(file)

        agent = cls(
            env=env,
            learning_rate=payload.get("alpha", 0.1),
            gamma=payload.get("gamma", 0.99),
            epsilon_start=payload.get("epsilon_start", 1.0),
            epsilon_end=payload.get("epsilon_end", 0.05),
            epsilon_decay_fraction=payload.get("epsilon_decay_fraction", 0.8),
            state_bins=payload.get("state_bins", 10),
            seed=seed,
            verbose=verbose,
        )
        agent.epsilon = payload.get("epsilon", agent.epsilon_end)
        agent.num_timesteps = payload.get("num_timesteps", 0)
        agent.q_table = defaultdict(agent._new_action_values, payload.get("q_table", {}))
        return agent

class YawningTitanRun:
    """
    The ``YawningTitanRun`` class is the run class for training YT agents from a given set of parameters.

    The ``YawningTitanRun`` class can be used 'straight out of the box', as all params have default values.

    .. code:: python

        yt_run = YawningTitanRun()

    The ``YawningTitanRun`` class can also be used manually by setting auto=False.

    .. code:: python

        yt_run = YawningTitanRun(auto=False)
        yt_run.setup()
        yt_run.train()
        yt_run.evaluate()

    Trained agents can be saved by calling ``.save()``. If no path is provided, a path is generated using the
    AGENTS_DIR, today's date, and the uuid of the instance of ``YawningTitanRun``.

    .. code:: python

        yt_run = YawningTitanRun()
        yt_run.save()

    .. todo::

        - Build a reporting functionality that captures all logs and eval and generates a PDF report.
        - Add multiple training runs functionality for the same agent.
        - Add the ability to load a saved agent and continue training it.
    """

    def __init__(
        self,
        network: Optional[Network] = None,
        game_mode: Optional[GameMode] = None,
        red_agent_class=RedInterface,
        blue_agent_class=BlueInterface,
        print_metrics: bool = False,
        show_metrics_every: int = 1,
        collect_additional_per_ts_data: bool = True,
        eval_freq: int = 10000,
        total_timesteps: int = 200000,
        training_runs: int = 1,
        n_eval_episodes: int = 1,
        deterministic: bool = False,
        warn: bool = True,
        render: bool = False,
        verbose: int = 1,
        logger: Optional[Logger] = None,
        output_dir: Optional[str] = None,
        auto: bool = True,
        seed: int = 42,
        agent_order: str = 'Blue_Red',
        algorithm: str = 'None',
        # === New hyperparameters ===
        learning_rate: float = 3e-4,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        gamma: float = 0.99,
        vf_coef: float = 0.5,
        n_epochs: int = 10,
        n_hidden_layers: int = 2,
        hidden_layer_size: int = 64,
        batch_size: int=64,
        n_steps: int=2048,
        ent_coef: float=0.0,
        separate_networks: bool = False,
        separate_grad_clip: bool = False,
        q_epsilon_start: float = 1.0,
        q_epsilon_end: float = 0.05,
        q_epsilon_decay_fraction: float = 0.8,
        q_state_bins: int = 10,
        **kwargs,
    ):
        """
        The YawningTitanRun constructor.

        # TODO: Add proper Sphinx mapping for classes/methods.

        :param network: An instance of ``Network``.
        :param game_mode: An instance of ``GameMode``.
        :param red_agent_class: The agent/action set class used for the red agent.
        :param blue_agent_class: The agent/action set class used for the blue agent.
        :param print_metrics: Print the metrics if True. Default value = True.
        :param show_metrics_every: Prints the metrics every ``show_metrics_every`` time steps. Default value = 10.
        :param collect_additional_per_ts_data: Collects additional per-timestep data if True.Default value = False.
        :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback. Default value = 10,000.
        :param total_timesteps: The number of samples (env steps) to train on. Default value = 200000.
        :param training_runs: The number of times the agent is trained.
        :param n_eval_episodes: The number of episodes to evaluate the agent. Default value = 1.
        :param deterministic: Whether the evaluation should use stochastic or deterministic actions. Default value =
            False.
        :param warn: Output additional warnings mainly related to the interaction with stable_baselines if True.
            Default value = True.
        :param render: Renders the environment during evaluation if True. Default value = False.
        :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used),
            2 for debug messages. Default value = 1.
        :param logger: An optional custom logger to override the use of the default module logger.
        :param output_dir: An optional output path for eval output and saved agent zip file. If none is provided,
            a path is generated using the ``yawning_titan.AGENTS_DIR``, today's date, and the uuid of the instance
            of ``YawningTitanRun``.
        :param auto: If True, ``setup()``, ``train()``, and ``evaluate()`` are called automatically.
        TODO: detail the extra parameters we have added
        """
        # -- store hyperparameters as instance attributes --
        self.learning_rate = learning_rate
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.separate_networks = separate_networks
        self.separate_grad_clip = separate_grad_clip
        self.q_epsilon_start = q_epsilon_start
        self.q_epsilon_end = q_epsilon_end
        self.q_epsilon_decay_fraction = q_epsilon_decay_fraction
        self.q_state_bins = q_state_bins

        # Give the run an uuid
        self.uuid: Final[str] = str(uuid4())

        # Initialise required instance variables as None
        self.network_interface: Optional[NetworkInterface] = None
        self.red: Optional[RedInterface] = None
        self.blue: Optional[BlueInterface] = None
        self.env: Optional[GenericNetworkEnv] = None
        self.agent: Optional[PPO] = None
        self.eval_callback: Optional[EvalCallback] = None

        # Set the network using the network arg if one was passed,
        # otherwise use the default 18 node network.
        if network:
            self.network: Network = network
        else:
            self.network = default_18_node_network()

        # Set the game_mode using the game_mode arg if one was passed,
        # otherwise use the game mode
        if game_mode:
            self.game_mode: GameMode = game_mode
        else:
            self.game_mode = default_game_mode()

        self._red_agent_class = red_agent_class
        self._blue_agent_class = blue_agent_class

        self.print_metrics = print_metrics
        self.show_metrics_every = show_metrics_every
        self.collect_additional_per_ts_data = collect_additional_per_ts_data
        self.eval_freq = eval_freq
        self.total_timesteps = total_timesteps
        self.training_runs = training_runs
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.warn = warn
        self.render = render
        self.verbose = verbose
        self.auto = auto
        self.seed = seed
        self.agent_order = agent_order
        self.algorithm = algorithm

        self.logger = _LOGGER if logger is None else logger
        self.logger.debug(f'YT run  {self.uuid}: Run initialised')

        self.output_dir = output_dir


        # Automatically setup, train, and evaluate the agent if auto is True.
        if self.auto:
            self.setup()
            self.train()
            self.evaluate()
            self.save()

    def _args_dict(self):
        return {
            'uuid': self.uuid,
            'network': self.network.to_dict(json_serializable=True),
            'game_mode': self.game_mode.to_dict(json_serializable=True),
            'red_agent_class': self._red_agent_class.__name__,
            'blue_agent_class': self._blue_agent_class.__name__,
            'print_metrics': self.print_metrics,
            'show_metrics_every': self.show_metrics_every,
            'collect_additional_per_ts_data': self.collect_additional_per_ts_data,
            'eval_freq': self.eval_freq,
            'total_timesteps': self.total_timesteps,
            'training_runs': self.training_runs,
            'n_eval_episodes': self.n_eval_episodes,
            'deterministic': self.deterministic,
            'warn': self.warn,
            'render': self.render,
            'verbose': self.verbose,
            'auto': self.auto,
        }
    
    def _get_new_algorithm(self) -> BaseAlgorithm:
        algorithm_key = str(self.algorithm).upper()

        if algorithm_key == "PPO":
            if self.separate_networks:
                net_arch = [dict(
                    pi=[self.hidden_layer_size] * self.n_hidden_layers,
                    vf=[self.hidden_layer_size] * self.n_hidden_layers,
                )]
            else:
                net_arch = [self.hidden_layer_size] * self.n_hidden_layers
            policy_kwargs = dict(net_arch=net_arch)

            return DiagnosticPPO(
                #CriticBiasedPolicy_NegativeBias, -- init value net with negative bias
                ActorCriticPolicy,
                self.env,
                verbose=self.verbose,
                tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
                seed=self.seed,
                # === pass the hyperparameters ===
                learning_rate=self.learning_rate,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                gamma=self.gamma,
                vf_coef=self.vf_coef,
                n_epochs=self.n_epochs,
                n_steps=self.n_steps,
                ent_coef=self.ent_coef,
                batch_size=self.batch_size,
                policy_kwargs=policy_kwargs,
                separate_grad_clip=self.separate_grad_clip,
            )
        elif algorithm_key == "DQN":
            # TODO: Set DQN hyperparams
            return DQN(
                DQNMlp,
                self.env,
                verbose=self.verbose,
                tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
                seed=self.seed,
                # === pass the hyperparameters ===
                exploration_final_eps=0.005,
                buffer_size=200_000,
            )
        elif algorithm_key in {"TABULAR_Q", "Q_LEARNING", "QLEARNING"}:
            return TabularQAgent(
                env=self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                epsilon_start=self.q_epsilon_start,
                epsilon_end=self.q_epsilon_end,
                epsilon_decay_fraction=self.q_epsilon_decay_fraction,
                state_bins=self.q_state_bins,
                seed=self.seed,
                verbose=self.verbose,
            )
        else:
            print(f"Algorithm {self.algorithm} not implemented.")
            exit()

    # def _get_new_ppo(self) -> PPO:
    #     """Get a new instance of ``stable_baselines.ppo.ppo.PPO``."""

    #     net_arch = [self.hidden_layer_size] * self.n_hidden_layers
    #     policy_kwargs = dict(net_arch=net_arch)

    #     return PPO(
    #         PPOMlp,
    #         self.env,
    #         verbose=self.verbose,
    #         tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
    #         seed=self.seed,
    #         # === pass the hyperparameters ===
    #         learning_rate=self.learning_rate,
    #         gae_lambda=self.gae_lambda,
    #         clip_range=self.clip_range,
    #         gamma=self.gamma,
    #         vf_coef=self.vf_coef,
    #         n_epochs=self.n_epochs,
    #         batch_size=self.batch_size,
    #         policy_kwargs=policy_kwargs
    #     )

    def _load_existing_model(self, model_zip_path: str) -> BaseAlgorithm:
         """Load an existing model file into PPO/DQN/etc"""
        # TODO: Should hyperparams be set when loading?
         algorithm_key = str(self.algorithm).upper()
         if algorithm_key == "PPO":
             return PPO.load(
                model_zip_path,
                self.env,
                verbose=self.verbose,
                tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
                seed=self.seed,
            )
         elif algorithm_key == "DQN":
             return DQN.load(
                model_zip_path,
                self.env,
                verbose=self.verbose,
                tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
                seed=self.seed,
            )
         elif algorithm_key in {"TABULAR_Q", "Q_LEARNING", "QLEARNING"}:
             return TabularQAgent.load(
                model_zip_path,
                self.env,
                verbose=self.verbose,
                seed=self.seed,
            )
         else:
            print(f"Algorithm {self.algorithm} not implemented.")
            exit()

    # def _load_existing_ppo(self, ppo_zip_path: str) -> PPO:
    #     """Load an existing ppo.zip file into ``stable_baselines.ppo.ppo.PPO``."""
    #     return PPO.load(
    #         ppo_zip_path,
    #         self.env,
    #         verbose=self.verbose,
    #         tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
    #         seed=self.seed,
    #     )
    
    def setup(self, new: bool = True, model_zip_path: Optional[str] = None):
        """
        Performs a setup of the ``NetworkInterface``, ``GenericNetworkEnv``, ``PPO`` algorithm.

        The setup needs to be performed before training can occur.

        :param new: If True, a new instance of PPO is generated. If False, a ppo_zip_path must be passed tooo.
        :param ppo_zip_path: Optional path to a saved ppo.zip file. Required if new = False.

        :raise AttributeError: When new=False and ppo_zip_path hasn't been provided.
        """
        if not new and not model_zip_path:
            msg = 'Performing setup when new=False requires ppo_zip_path as the path of a saved ppo.zip file.'
            try:
                raise AttributeError(msg)
            except AttributeError as e:
                _LOGGER.critical(e)
                raise e

        if self.output_dir:
            if isinstance(self.output_dir, str):
                self.output_dir = pathlib.Path(self.output_dir)
        else:
            self.output_dir = pathlib.Path(
                os.path.join(
                    AGENTS_DIR, 'trained', str(datetime.now().date()), f'{self.uuid}'
                )
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.network_interface = NetworkInterface(
            game_mode=self.game_mode, network=self.network
        )
        self.logger.debug(f'YT run  {self.uuid}: Network interface created')

        self.red = self._red_agent_class(self.network_interface)
        self.logger.debug(f'YT run  {self.uuid}: Red agent created')

        self.blue = self._blue_agent_class(self.network_interface)
        self.logger.debug(f'YT run  {self.uuid}: Blue agent created')

        self.env = GenericNetworkEnv(
            red_agent=self.red,
            blue_agent=self.blue,
            network_interface=self.network_interface,
            print_metrics=self.print_metrics,
            show_metrics_every=self.show_metrics_every,
            collect_additional_per_ts_data=self.collect_additional_per_ts_data,
            agent_order=self.agent_order
        )
        self.logger.debug(f'YT run  {self.uuid}: GenericNetworkEnv created')

        self.logger.debug(f'YT run  {self.uuid}: Performing env check')
        check_env(self.env, warn=self.warn)
        self.logger.debug(f'YT run  {self.uuid}: Env checking complete')

        self.env.reset()
        self.logger.debug(f'YT run  {self.uuid}: GenericNetworkEnv reset')

        self.logger.debug(f'YT run  {self.uuid}: Instantiating agent')
        if new:
            self.agent = self._get_new_algorithm()
        else:
            self.agent = self._load_existing_model(model_zip_path)
        self.logger.debug(f'YT run  {self.uuid}: Agent instantiated')


        # Create the evaluation environment
        eval_env = GenericNetworkEnv(
            red_agent=self.red,
            blue_agent=self.blue,
            network_interface=self.network_interface,
            print_metrics=self.print_metrics,
            show_metrics_every=self.show_metrics_every,
            collect_additional_per_ts_data=self.collect_additional_per_ts_data,
            agent_order=self.agent_order
        )

        if str(self.algorithm).upper() in {"TABULAR_Q", "Q_LEARNING", "QLEARNING"}:
            self.eval_callback = None
            self.callbacks = None
            self.logger.debug(
                f'YT run {self.uuid}: Using tabular Q-learning (SB3 callbacks disabled).'
            )
        else:
            # Instantiate the custom callback
            custom_eval_callback = CustomEvalCallback(eval_env)
            gtScore_callback = GTScoreCallback(eval_env)
            ent_coeff_callback = EntropyDecayCallback(
                total_timesteps=self.total_timesteps
            )

            self.logger.debug(f'YT run  {self.uuid}: Eval callback set')

            # GTScore + std_reward logged from training episodes; no separate
            # EvalCallback needed (saves ~50k redundant env steps per run).
            self.callbacks = [custom_eval_callback, gtScore_callback,
                              ent_coeff_callback]

            # Only attach the W&B callback when a run is active (i.e. the caller
            # called wandb.init). WandbCallback asserts wandb.run is not None, so
            # this keeps training working when W&B logging is turned off.
            if wandb.run is not None:
                self.callbacks.insert(0, WandbCallback())

    def train(self) -> Union[PPO, None]:
        """
        Trains the agent.

        :return: The trained instance of ``stable_baselines3.ppo.ppo.PPO``.
        """
        if self.env and self.agent:
            self.logger.debug(f'YT run  {self.uuid}: Performing agent training')
            for i in range(self.training_runs):
                self.agent.learn(
                    total_timesteps=self.total_timesteps,
                    n_eval_episodes=self.n_eval_episodes,
                    callback=self.callbacks,
                    progress_bar=False,
                )
                self.logger.debug(f'YT run  {self.uuid}: Training run {i + 1} complete')

                self.env.reset()
                self.logger.debug(f'YT run  {self.uuid}: GenericNetworkEnv reset')

            self.logger.debug(f'YT run  {self.uuid}: Agent training complete')
            return self.agent
        else:
            self.logger.error(
                f'Cannot train the agent for YT run  {self.uuid} as the run has not been setup. '
                f'Call .setup() on the instance of {self.__class__.__name__} to setup the run.'
            )

    def evaluate(self) -> Union[tuple[float, float], tuple[List[float], List[int]]]:
        """
        Evaluates the trained agent.

        :return: Mean reward per episode, std of reward per episode.
        """
        if self.agent:
            return evaluate_policy(
                self.agent, self.env, n_eval_episodes=self.n_eval_episodes
            )
        else:
            self.logger.error(
                f'Cannot evaluate YT run  {self.uuid} as the agent has not been trained. '
                f'Call .train() on the instance of {self.__class__.__name__} to train the agent.'
            )

    def save(self) -> Union[str, None]:
        """
        Saves the trained agent using the stable_baselines3 save as zip functionality.

        The instance of the active algorithm is saved to <algo>_{index}.zip (or .pkl for tabular Q-learning),
        where index starts at 0.
        The YawningTitanRun args are saved to args_{index}.json.
        The YawningTitanRun.uuid is saved to UUID_{index}.

        If ppo_0.zip already exists, the method tries ppo_1.zip, then ppo_2.zip, and so on.

        :return: The path to which the agent has been saved, or None if the agent is not trained.
        """
        if self.agent:
            algorithm_key = str(self.algorithm).upper()
            if algorithm_key == "PPO":
                model_prefix, model_ext = "ppo", ".zip"
            elif algorithm_key == "DQN":
                model_prefix, model_ext = "dqn", ".zip"
            elif algorithm_key in {"TABULAR_Q", "Q_LEARNING", "QLEARNING"}:
                model_prefix, model_ext = "tabular_q", ".pkl"
            else:
                model_prefix, model_ext = "agent", ".zip"

            # Find the next available index
            i = 0
            while True:
                agent_filename = f"{model_prefix}_{i}{model_ext}"
                args_filename = f"args_{i}.json"
                uuid_filename = f"UUID_{i}"

                agent_path = os.path.join(self.output_dir, agent_filename)
                args_path = os.path.join(self.output_dir, args_filename)
                uuid_path = os.path.join(self.output_dir, uuid_filename)

                # If none of the files with index i exists, we can use that index
                if not (os.path.exists(agent_path) or 
                        os.path.exists(args_path) or
                        os.path.exists(uuid_path)):
                    break
                i += 1

            # Save the agent
            self.agent.save(path=agent_path)

            # Dump the args to a JSON file
            with open(args_path, 'x') as file:
                json.dump(self._args_dict(), file, indent=4)

            # Write the UUID file
            with open(uuid_path, 'x') as file:
                file.write(self.uuid)

            self.logger.debug(
                f'YT run {self.uuid}: Saved trained agent (Stable Baselines3 PPO) to: {agent_path}'
            )

            return str(agent_path)
        else:
            self.logger.error(
                f'Cannot save the trained agent from YT run {self.uuid} as the agent has not been '
                f'trained. Call .train() on the instance of {self.__class__.__name__} to train the agent.'
            )
            return None

    def _build_inventory_file(self):
        # Walk the output_dir to build an inventory file
        inventory_path = os.path.join(self.output_dir, 'INVENTORY')
        if os.path.isfile(inventory_path):
            os.remove(inventory_path)
        self.logger.debug(
            f'YT run  {self.uuid}: Building INVENTORY file {inventory_path}.'
        )

        with open(inventory_path, 'w') as inventory:
            inventory.write('file, ST_SIZE')
            inventory.write('\n')
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if file != 'INVENTORY':
                        file_path = os.path.join(root, file)
                        dir_path = file_path.replace(str(self.output_dir), '')[1:]
                        file_stat = os.stat(file_path)
                        inventory.write(f'{dir_path}, {file_stat.st_size}')
                        inventory.write('\n')
                        self.logger.debug(
                            f'YT run  {self.uuid}: File added to inventory: {dir_path}.'
                        )
        self.logger.debug(f'YT run  {self.uuid}: Finished building INVENTORY file.')

    def export(self) -> str:
        """
        Export the YawningTitanRun as a zip.

        The contents of output_dir is archived to the agents_dir exported dir.

        Included is an INVENTORY file that contains all files and their sizes. This is used for file verification when
        an exported YawningTitanRun is imported.

        :return: The exported filepath as a str.
        """
        self.logger.debug(f'YT run  {self.uuid}: Performing export.')
        self.save()

        self._build_inventory_file()

        # Make a zip archive of the output dir
        exported_root = pathlib.Path(os.path.join(AGENTS_DIR, 'exported'))
        exported_root.mkdir(parents=True, exist_ok=True)
        export_path = os.path.join(exported_root, f'EXPORTED_YT_RUN_{self.uuid}')
        self.logger.debug(
            f'YT run  {self.uuid}: Making a zip archive of {self.output_dir} and writing to {export_path}.zip.'
        )
        shutil.make_archive(export_path, 'zip', self.output_dir)
        self.logger.debug(f'YT run  {self.uuid}: Export completed.')
        return f'{export_path}.zip'

    # TODO: Remove once proper AgentClass sub-classes have been created and mapped as a function in the main module.
    @classmethod
    def _get_agent_class_from_str(cls, agent_class_str):
        """Maps AgentClass string names to their actual class."""
        mapping = {
            'RedInterface': RedInterface,
            'SineWaveRedAgent': SineWaveRedAgent,
            'FixedRedAgent': FixedRedAgent,
            'NSARed': NSARed,
            'BlueInterface': BlueInterface,
            'SimpleBlue': SimpleBlue,
        }
        return mapping[agent_class_str]

    @classmethod
    def _load_args_file(cls, path: str) -> Dict:
        """
        Load an args.json file and returns as a dict.

        :param path: A saved YawningTitanRun path.
        :return: The args.json file as a dict.

        :raise ValueError: When an args.json file doesn't exist in the provided path. Or when it does exist but it's
            keys aren't correct.
        """
        args_path = os.path.join(path, 'args.json')
        msg = f'Cannot load trained agent as the args file ({args_path}) '
        if os.path.isfile(args_path):
            with open(args_path, 'r') as file:
                args = yaml.safe_load(file)

            if args.keys() == YawningTitanRun(auto=False)._args_dict().keys():
                args['network'] = Network.create(args['network'])
                args['game_mode'] = GameMode.create(args['game_mode'])
                args['red_agent_class'] = cls._get_agent_class_from_str(
                    args['red_agent_class']
                )
                args['blue_agent_class'] = cls._get_agent_class_from_str(
                    args['blue_agent_class']
                )
                return args
            else:
                # Args file keys don't match
                msg = f'{msg} is corrupted.'
                _LOGGER.error(msg)
                raise ValueError(msg)
        else:
            # Args file doesn't exist
            msg = f'{msg} does not exist.'
            _LOGGER.error(msg)
            raise ValueError(msg)

    @classmethod
    def load(cls, path: str, algo: str):
        """
        Load and return a saved YawningTitanRun.

        YawningTitanRun's that have auto=True will not be automatically ran on load.

        :param path: A saved YawningTitanRun path.
        :return: An instance of YawningTitanRun.
        """
        args = cls._load_args_file(path)

        uuid = args.pop('uuid')
        args.pop('auto')

        yt_run = YawningTitanRun(**args, auto=False)
        yt_run.uuid = uuid  # noqa - We'll allow it here :) #TODO: what?
        algo_key = str(algo).upper()
        if algo_key == "PPO":
            model_zip_path = os.path.join(path, 'ppo.zip')
            if not os.path.exists(model_zip_path):
                model_zip_path = os.path.join(path, 'ppo_0.zip')
        elif algo_key == "DQN":
            model_zip_path = os.path.join(path, 'dqn.zip')
            if not os.path.exists(model_zip_path):
                model_zip_path = os.path.join(path, 'dqn_0.zip')
        elif algo_key in {"TABULAR_Q", "Q_LEARNING", "QLEARNING"}:
            model_zip_path = os.path.join(path, 'tabular_q.pkl')
            if not os.path.exists(model_zip_path):
                model_zip_path = os.path.join(path, 'tabular_q_0.pkl')
        else:
            print(f"algo {algo} unknown")
            exit()

        yt_run.setup(new=False, model_zip_path=model_zip_path)

        return yt_run

    @classmethod
    def _verify_import_export_zip_file(cls, unzip_path) -> bool:
        """
        Verifies an INVENTORY file with the files contained in its parent dir.

        :param unzip_path: An unzipped exported YawningTitanRun path.
        :return: Whether the INVENTORY file matches the files.
        """
        with open(os.path.join(unzip_path, 'INVENTORY'), 'r') as inventory_file:
            for line in inventory_file.readlines()[1:]:
                line = line.rstrip('\n').split(',')
                print(line)
                file_name, st_size = line[0], int(line[1])
                print(unzip_path, file_name)
                target_file_path = os.path.join(unzip_path, file_name)
                print(target_file_path)
                _LOGGER.debug(f'Attempting to verify file: {target_file_path}')
                if os.path.isfile(target_file_path):
                    file_stat = os.stat(target_file_path)
                    if st_size != file_stat.st_size:
                        # File Size doesn't match
                        _LOGGER.debug(
                            f"   Verification failed, file size {file_stat.st_size} doesn't match {st_size}."
                        )
                        return False
                else:
                    # File doesn't exist
                    _LOGGER.debug("   Verification failed, file doesn't exist.")
                    return False
            _LOGGER.debug('   Verification successful.')
        return True

    @classmethod
    def import_from_export(
        cls, exported_zip_file_path: str, overwrite_existing: bool = False
    ) -> YawningTitanRun:
        """
        Import and return an exported YawningTitanRun.

        YawningTitanRun's that have auto=True will not be automatically ran on import.

        :param exported_zip_file_path: The path of an exported YawningTitanRun.
        :param overwrite_existing: If True, if the uuid of the imported agent already exists in the trainer agents dir
            it is overwritten.
        :return: The imported instance of YawningTitanRun.

        :raise YawningTitanRunError: When the INVENTORY file fails its verification.
        """
        _LOGGER.debug(f'Importing exported agent from {exported_zip_file_path}')
        # Unzip into trained agents folder
        unzip_path = pathlib.Path(
            os.path.join(
                AGENTS_DIR, 'trained', str(datetime.now().date()), str(uuid4())
            )
        )
        unzip_path.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(exported_zip_file_path, unzip_path, 'zip')

        # Verify the contents
        verified = cls._verify_import_export_zip_file(unzip_path)
        if not verified:
            msg = f'Failed to verify the contents while importing YawningTitanRun from {exported_zip_file_path}.'
            try:
                raise YawningTitanRunError(msg)
            except YawningTitanRunError as e:
                _LOGGER.critical(e)
                raise e

        # Rename unzip_dir using the UUID
        with open(os.path.join(unzip_path, 'UUID')) as file:
            uuid = file.read()
        new_unzip_path = pathlib.Path(
            os.path.join(AGENTS_DIR, 'trained', str(datetime.now().date()), uuid)
        )
        if not os.path.isdir(new_unzip_path):
            os.rename(unzip_path, new_unzip_path)
        else:
            # Has already been imported or was created on this machine
            if overwrite_existing:
                # Overwrite
                shutil.rmtree(new_unzip_path)
                os.rename(unzip_path, new_unzip_path)
                _LOGGER.debug(
                    f'Existing YawningTitanRun overwritten at {new_unzip_path}.'
                )

        # Pass new_unzip_path to .load and return
        return cls.load(str(new_unzip_path))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f"uuid='{self.uuid}', "
            f'network={self.network}, '
            f'game_mode={self.game_mode}, '
            f'red_agent_class={self._red_agent_class}, '
            f'blue_agent_class={self._blue_agent_class}, '
            f'print_metrics={self.print_metrics}, '
            f'show_metrics_every={self.show_metrics_every}, '
            f'collect_additional_per_ts_data={self.collect_additional_per_ts_data}, '
            f'eval_freq={self.eval_freq}, '
            f'total_timesteps={self.total_timesteps}, '
            f'training_runs={self.training_runs}, '
            f'n_eval_episodes={self.n_eval_episodes}, '
            f'deterministic={self.deterministic}, '
            f'warn={self.warn}, '
            f'render={self.render}, '
            f'verbose={self.verbose}'
            ')'
        )