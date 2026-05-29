import os
import random
import json
import shutil
import wandb
import logging
import tempfile
import sys
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

# Add sys path if you need to
# sys.path.append('../icml2026_code/YAWNING-TITAN/src/yawning_titan')

os.environ["WANDB_SILENT"] = "True"    # Don't print to console
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan_run_wandb import YawningTitanRun
from yawning_titan.networks.network import Network
from icml2026_codebase.Reward_Function_Experimentation.Network.N_node_generator import GenerateSubproblemNetwork

def run_experiment(trial_name, group_name, eval_type, n_nodes, reward_function, n_steps, node_vulnerability,
                   red_agent_skill, n_timesteps, temp_dir, action_space_set,
                   order, net_shape, algorithm, wandb_project_name, output_location):
    # Create a unique config file in the temporary directory
    config_file_path = os.path.join(temp_dir, f'Minimal_network_gamemode_{trial_name}.json')
    shutil.copyfile('Minimal_network_gamemode.json', config_file_path)

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

    net_dict = GenerateSubproblemNetwork(num_nodes=n_nodes).network

    network = Network.create(network_dict=net_dict)

    network_interface = NetworkInterface(game_mode=game_mode, network=network)
    if action_space_set == 'simple_action_space':
        network_interface.game_mode.blue.action_set.place_decoy.value = False
    elif action_space_set == 'decoy_action_space':
        network_interface.game_mode.blue.action_set.place_decoy.value = True

    output_dir = f'{output_location}/train_log/{action_space_set}/{reward_function}/{n_nodes}_nodes/{algorithm}/' + trial_name

    run = wandb.init(
        name=trial_name,
        entity='YTRewards',
        project=wandb_project_name,
        sync_tensorboard=True,
        group=group_name,
    )

    seed = random.randint(0, 1000)
    runner = YawningTitanRun(
        game_mode=game_mode,
        network=network,
        total_timesteps=n_timesteps,
        render=False,
        seed=seed,
        output_dir=output_dir,
        auto=False,
        print_metrics=False,
        verbose=0, # ! This shuts up the data overload
        agent_order=order,
        algorithm=algorithm
    )

    runner.setup()
    runner.train()
    runner.save()

    run.finish()


def run_experiment_with_logging(trial_name, group_name, eval_type, n_nodes, reward_function, n_steps,
                                node_vulnerability, red_agent_skill, n_timesteps, temp_dir, action_space_set,
                                order, net_shape, algorithm, wandb_project_name, output_location):
    """Wrapper function for run_experiment with error handling and logging."""
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
            order=order,
            net_shape=net_shape,
            algorithm=algorithm,
            wandb_project_name=wandb_project_name,
            output_location=output_location
        )
        logging.info(f"Experiment {trial_name} completed successfully.")
    except Exception as e:
        logging.error(f"Error in trial {trial_name}: {str(e)}")
        # Log the full traceback including the line number
        logging.error("Traceback details:", exc_info=True)