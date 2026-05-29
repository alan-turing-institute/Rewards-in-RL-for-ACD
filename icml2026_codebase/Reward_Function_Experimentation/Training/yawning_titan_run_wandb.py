from __future__ import annotations

import json
import os.path
from typing import List, Dict, Any
import pathlib
import shutil
from datetime import datetime
from logging import Logger, getLogger
from typing import Dict, Final, List, Optional, Union
from uuid import uuid4

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
import numpy as np

_LOGGER = getLogger(__name__)
_LOGGER.setLevel(40)

class CustomEvalCallback(BaseCallback):
    """
    A single callback that does two things per episode:
      1) Logs a single "std_reward" (negative #compromised nodes) using final step's end_state
      2) Logs "avg_reward_delta" for the episode by comparing consecutive step rewards
    """

    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env

        # For logging final 'std_reward' each episode
        self.episode_std_rewards = []  # Stores the std_reward for each completed episode

        # For logging observation data each step & calculating reward deltas
        self.current_episode_data: List[Dict[str, Any]] = []
        self.reward_deltas: List[float] = []


    def _on_step(self) -> bool:
        """
        Called after each environment step in Training.
        We do two main things here:
          - Accumulate step data for reward-delta logging
          - Check for done=True; if so, compute std reward & final logging
        """
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        obs = self.locals["new_obs"]
        actions = self.locals["actions"]

        # 1) Convert `rewards` to a scalar if it's a list/array
        if isinstance(rewards, (list, np.ndarray)):
            step_reward = float(np.mean(rewards))
        else:
            step_reward = float(rewards)

        # 2) Calculate the reward delta
        if len(self.current_episode_data) > 0:
            last_reward = self.current_episode_data[-1]["rewards"]
            reward_delta = abs(step_reward - last_reward)
        else:
            reward_delta = 0.0

        self.reward_deltas.append(reward_delta)

        # 3) Store the step data (for reference or debugging)
        self.current_episode_data.append({
            "observations": obs,
            "actions": actions,
            "rewards": step_reward,
            "dones": dones,
            "infos": infos,
        })

        # 4) If the environment ended, log final std reward & episode metrics
        if np.any(dones):
            # -- Compute the final "std reward" from the final info
            final_info = infos[0]  # Single-env assumption
            end_state = final_info.get("end_state", {})
            no_compromised_nodes = sum(1 for v in end_state.values() if v == 1)
            std_reward = -no_compromised_nodes  # negative # of compromised nodes

            self.episode_std_rewards.append(std_reward)

            # -- Compute "avg_reward_delta" for the just-finished episode
            if len(self.reward_deltas) > 0:
                avg_reward_delta = float(np.mean(self.reward_deltas))
            else:
                avg_reward_delta = 0.0

            # -- Log to SB3's logger
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/avg_reward_delta", avg_reward_delta)
            self.logger.dump(self.num_timesteps)


            # -- Reset the episode-level trackers
            self.current_episode_data.clear()
            self.reward_deltas.clear()

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
        Called after the entire Training is done. We can compute an overall average
        std_reward if we wish. We'll do last 20% as per your existing logic:
        """
        if self.episode_std_rewards:
            # Compute average of last 20% of episodes
            last_20pct = max(1, len(self.episode_std_rewards) // 5)  # if n < 5, just use all episodes
            last20_subset = self.episode_std_rewards[-last_20pct:]
            mean_std_rew_score = sum(last20_subset) / len(last20_subset)
            self.logger.record("eval/last20pct_std_reward_mean", mean_std_rew_score)
            self.logger.dump(self.num_timesteps)

class GTScoreCallback(BaseCallback):
    """
    Logs a single "eval/GTScore" each time an episode ends (done=True),
    where GTScore is the sum over all steps in the episode of the maximum compromised count.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_gt_scores = []  # Store GTScore per episode
        self.current_episode_gtscore = 0  # Accumulator for the current episode

    def _on_step(self) -> bool:
        # Get the info for the current step; assuming single env, so index 0.
        info = self.locals["infos"][0]
        
        # Extract per-step compromised counts from both mid and end states.
        mid_compromised_states = info.get("mid_step_info", {}).get("mid_step", {}).get("mid_state_compromised", {})
        mid_compromised_count = sum(mid_compromised_states.values())

        compromised_states = info.get("end_state", {})
        compromised_count = sum(compromised_states.values())

        # For this step, use the maximum compromised count.
        step_max = max(mid_compromised_count, compromised_count)
        # Accumulate the value.
        self.current_episode_gtscore += step_max

        dones = self.locals["dones"]
        if np.any(dones):
            # At the end of the episode, log the accumulated GTScore.
            gt_score = -self.current_episode_gtscore  # Negative if you want higher scores for fewer compromises.
            # print(f"GTScore for this episode: {gt_score}")
            self.episode_gt_scores.append(gt_score)
            self.logger.record("eval/GTScore", gt_score)
            self.logger.dump(self.num_timesteps)
            
            # Reset for the next episode.
            self.current_episode_gtscore = 0

        return True  # Continue Training
    
    def _on_training_end(self) -> None:
        """
        After Training, compute and log the average GTScore for the last 20% of episodes.
        """
        if self.episode_gt_scores:
            # Compute average over the last 20% of episodes (or at least one).
            last_20pct = max(1, len(self.episode_gt_scores) // 5)
            last20_subset = self.episode_gt_scores[-last_20pct:]
            mean_gt_score = sum(last20_subset) / len(last20_subset)
            self.logger.record("eval/last20pct_GTScore_mean", mean_gt_score)
            self.logger.dump(self.num_timesteps)

class YawningTitanRun:
    """
    The ``YawningTitanRun`` class is the run class for Training YT agents from a given set of parameters.

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
        - Add multiple Training runs functionality for the same agent.
        - Add the ability to load a saved agent and continue Training it.
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
        if self.algorithm == "PPO":
            net_arch = [self.hidden_layer_size] * self.n_hidden_layers
            policy_kwargs = dict(net_arch=net_arch)

            return PPO(
                PPOMlp,
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
                batch_size=self.batch_size,
                policy_kwargs=policy_kwargs
            )
        elif self.algorithm == "DQN":
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
        else:
            print("Algorithm {self.algorithm} not implemented.")
            exit()

    def _load_existing_model(self, model_zip_path: str) -> BaseAlgorithm:
         """Load an existing model file into PPO/DQN/etc"""
        # TODO: Should hyperparams be set when loading?
         if self.algorithm == "PPO":
             return PPO.load(
                model_zip_path,
                self.env,
                verbose=self.verbose,
                tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
                seed=self.seed,
            )
         elif self.algorithm == "DQN":
             return DQN.load(
                model_zip_path,
                self.env,
                verbose=self.verbose,
                tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
                seed=self.seed,
            )
         else:
            print("Algorithm {self.algorithm} not implemented.")
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

        The setup needs to be performed before Training can occur.

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

        # Instantiate the custom callback
        custom_eval_callback = CustomEvalCallback(eval_env)
        gtScore_callback = GTScoreCallback(eval_env)

        self.eval_callback = EvalCallback(
            Monitor(self.env, str(self.output_dir)),
            eval_freq=self.eval_freq,
            deterministic=self.deterministic,
            render=self.render,
            verbose=self.verbose,
        )
        self.logger.debug(f'YT run  {self.uuid}: Eval callback set')

        # Add the custom callback to the list of callbacks
        self.callbacks = [self.eval_callback, WandbCallback(), custom_eval_callback, gtScore_callback]

    def train(self) -> Union[PPO, None]:
        """
        Trains the agent.

        :return: The trained instance of ``stable_baselines3.ppo.ppo.PPO``.
        """
        if self.env and self.agent and self.eval_callback:
            self.logger.debug(f'YT run  {self.uuid}: Performing agent Training')
            for i in range(self.training_runs):
                self.agent.learn(
                    total_timesteps=self.total_timesteps,
                    n_eval_episodes=self.n_eval_episodes,
                    callback=self.callbacks,
                    progress_bar=True,
                )
                self.logger.debug(f'YT run  {self.uuid}: Training run {i + 1} complete')

                self.env.reset()
                self.logger.debug(f'YT run  {self.uuid}: GenericNetworkEnv reset')

            self.logger.debug(f'YT run  {self.uuid}: Agent Training complete')
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

        The instance of PPO is saved to ppo_{index}.zip (where index starts at 0).
        The YawningTitanRun args are saved to args_{index}.json.
        The YawningTitanRun.uuid is saved to UUID_{index}.

        If ppo_0.zip already exists, the method tries ppo_1.zip, then ppo_2.zip, and so on.

        :return: The path to which the agent has been saved, or None if the agent is not trained.
        """
        if self.agent:
            # Find the next available index
            i = 0
            while True:
                agent_filename = f"ppo_{i}.zip"
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
        if algo == "PPO":
            model_zip_path=os.path.join(path, 'ppo.zip')
        elif algo == "DQN":
            model_zip_path=os.path.join(path, 'dqn.zip')
        else:
            print("algo {algo} unknown")
            exit()
        
        yt_run.setup(model_zip_path, new=False)

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