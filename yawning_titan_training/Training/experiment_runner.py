import os
from pathlib import Path

# Limit each worker to 1 thread — prevents thrashing when many processes share CPU cores.
# Must be set before torch/numpy are imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import json
import shutil
import wandb
import logging
import sys

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_SILENT"] = "True"

# Resolve paths relative to this file so the repo works anywhere.
_HERE = Path(__file__).resolve().parent                  # yawning_titan_training/Training/
_YT_TRAINING_ROOT = _HERE.parent                         # yawning_titan_training/
_REPO_ROOT = _YT_TRAINING_ROOT.parent                   # repo root
_YT_SRC = _REPO_ROOT / "YAWNING-TITAN" / "src"

for _p in [str(_HERE), str(_YT_TRAINING_ROOT), str(_YT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.networks.network import Network
from yawning_titan_run_wandb import YawningTitanRun
from Networks.N_node_generator import GenerateSubproblemNetwork
from Networks.Diamond_network import GenerateDiamondNetwork


def run_experiment(trial_name, group_name, eval_type, n_nodes, reward_function, n_steps, node_vulnerability,
                   red_agent_skill, n_timesteps, temp_dir, action_space_set, hyperparameters,
                   order, net_shape, algorithm, wandb_project_name, wandb_entity, output_location):
    # Create a per-process copy of the game-mode config so parallel runs don't race.
    config_file_path = os.path.join(temp_dir, f'Minimal_network_gamemode_{reward_function}_{trial_name}.json')
    shutil.copyfile(str(_HERE / 'Minimal_network_gamemode.json'), config_file_path)

    with open(config_file_path) as config:
        config_dict = json.load(config)
        config_dict['game_rules']['max_steps'] = n_steps
        config_dict['rewards']['function'] = reward_function
        config_dict['red']['agent_attack']['skill']['value'] = float(red_agent_skill)
        if action_space_set == 'simple_action_space':
            config_dict['blue']['action_set']['place_decoy'] = 'false'
        elif action_space_set == 'decoy_action_space':
            config_dict['blue']['action_set']['place_decoy'] = 'true'

    with open(config_file_path, 'w') as file:
        json.dump(config_dict, file, indent=4)

    game_mode = GameMode.create(dict=config_dict)

    if net_shape == 'diamond':
        net_dict = GenerateDiamondNetwork(num_nodes=n_nodes).network
    elif net_shape == 'linear':
        net_dict = GenerateSubproblemNetwork(num_nodes=n_nodes).network
    else:
        raise ValueError(f"Invalid network shape: {net_shape}. Must be 'linear' or 'diamond'.")

    network = Network.create(network_dict=net_dict)

    network_interface = NetworkInterface(game_mode=game_mode, network=network)
    if action_space_set == 'simple_action_space':
        network_interface.game_mode.blue.action_set.place_decoy.value = False
    elif action_space_set == 'decoy_action_space':
        network_interface.game_mode.blue.action_set.place_decoy.value = True

    output_dir = os.path.join(
        output_location, 'train_log', action_space_set, reward_function,
        f'{n_nodes}_nodes', algorithm, trial_name
    )

    run = wandb.init(
        name=trial_name,
        entity=wandb_entity,
        project=wandb_project_name,
        sync_tensorboard=True,
        group=group_name,
        settings=wandb.Settings(_service_wait=180),
    )

    seed = random.randint(0, 1000)
    algorithm_key = str(algorithm).upper()

    q_learning_rate = hyperparameters.get("q_learning_rate", 0.1)
    q_gamma = hyperparameters.get("q_gamma", 0.99)
    q_epsilon_start = hyperparameters.get("q_epsilon_start", 1.0)
    q_epsilon_end = hyperparameters.get("q_epsilon_end", 0.05)
    q_epsilon_decay_fraction = hyperparameters.get("q_epsilon_decay_fraction", 0.8)
    q_state_bins = hyperparameters.get("q_state_bins", 10)

    runner = YawningTitanRun(
        game_mode=game_mode,
        network=network,
        total_timesteps=n_timesteps,
        render=False,
        seed=seed,
        output_dir=output_dir,
        auto=False,
        print_metrics=False,
        verbose=0,
        learning_rate=(q_learning_rate if algorithm_key in {"TABULAR_Q", "Q_LEARNING", "QLEARNING"}
                       else hyperparameters.get("learning_rate", 3e-4)),
        n_hidden_layers=hyperparameters.get("n_hidden_layers", 2),
        gae_lambda=hyperparameters.get("gae_lambda", 0.95),
        clip_range=hyperparameters.get("clip_range", 0.2),
        gamma=(q_gamma if algorithm_key in {"TABULAR_Q", "Q_LEARNING", "QLEARNING"}
               else hyperparameters.get("gamma", 0.99)),
        vf_coef=hyperparameters.get("vf_coef", 0.5),
        n_epochs=hyperparameters.get("n_epochs", 10),
        hidden_layer_size=hyperparameters.get("hidden_layer_size", 64),
        n_steps=hyperparameters.get("n_steps", 2048),
        ent_coef=hyperparameters.get("ent_coef", 0.0),
        separate_networks=hyperparameters.get("separate_networks", False),
        separate_grad_clip=hyperparameters.get("separate_grad_clip", False),
        q_epsilon_start=q_epsilon_start,
        q_epsilon_end=q_epsilon_end,
        q_epsilon_decay_fraction=q_epsilon_decay_fraction,
        q_state_bins=q_state_bins,
        agent_order=order,
        algorithm=algorithm,
    )

    runner.setup()
    runner.train()
    runner.save()

    run.finish()


def run_experiment_with_logging(trial_name, group_name, eval_type, n_nodes, reward_function, n_steps,
                                node_vulnerability, red_agent_skill, n_timesteps, temp_dir, action_space_set,
                                hyperparameters, order, net_shape, algorithm,
                                wandb_project_name, wandb_entity, output_location):
    try:
        logging.info(f"Running experiment: {trial_name}")
        run_experiment(
            trial_name=trial_name,
            group_name=group_name,
            eval_type=eval_type,
            n_nodes=n_nodes,
            reward_function=reward_function,
            n_steps=n_steps,
            node_vulnerability=node_vulnerability,
            red_agent_skill=red_agent_skill,
            n_timesteps=n_timesteps,
            temp_dir=temp_dir,
            action_space_set=action_space_set,
            hyperparameters=hyperparameters,
            order=order,
            net_shape=net_shape,
            algorithm=algorithm,
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity,
            output_location=output_location,
        )
        logging.info(f"Experiment {trial_name} completed successfully.")
    except Exception as e:
        logging.error(f"Error in trial {trial_name}: {str(e)}")
        logging.error("Traceback details:", exc_info=True)
