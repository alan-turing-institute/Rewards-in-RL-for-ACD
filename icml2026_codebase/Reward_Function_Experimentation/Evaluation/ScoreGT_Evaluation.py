import json
import random
from itertools import product
import sys

# add a system path if necessary
sys.path.append('')

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.networks.network import Network
from Reward_Function_Experimentation.Network.N_node_generator import GenerateSubproblemNetwork
import matplotlib.pyplot as plt
import copy
import time
import multiprocessing
from Reward_Function_Experimentation.Training.utils import evaluate_combination
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio
from tqdm import tqdm
from pprint import pprint
import os
import psutil
import os
import re

######################## Evaluation Score ########################
# This script is used to evaluate the performance of the sets of agents according to the ground truth evaluation metric.
# The evaluation gives you the ScoreGT and the upper and lower RF values too, printed to an eval file in output_dir,
# as specified below

# To evaluate a model or set of models, first check the global vars below are applicable to all (N_STEPS, N_EPISODES,
# NO_AGENTS, ORDER, ACTION_SPACE_SET).

# Then go to the parallel_evaluation function and fill out the node_values (say 2, 5, 10, 20, 50) and reward_values (
# say 'Negative Rewards', 'Positive Rewards', 'Scaffolded Rewards') with the agents you want to evaluate in parallel.
# (See utils for the reward function names config)

# Approx timings:
# 50 Nodes, 1000 episodes, 25 agents: 1 hour 35mins


# These are the Global variables that are universal to all agent evaluations
NODE_VULNERABILITY = 1
RED_AGENT_SKILL = 1
N_STEPS = 100
N_EPISODES = 1000
NO_AGENTS = 10 # No of agents to evaluate
EVAL_TYPE = "intra_step_eval"  # this version is evaluating intra step-wise - ground truth.
MODEL_LOCATION = "Reward_Engineering/Models"

def log_resource_usage(interval=1):
    """
    Logs system resource usage every `interval` seconds.
    """
    while True:
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.5)
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory_info.percent}% (Used: {memory_info.used // (1024 ** 2)} MB, "
              f"Available: {memory_info.available // (1024 ** 2)} MB)")
        print("-" * 50)
        time.sleep(interval)


def load_trained_agent(env: VecEnv, n_agent_index: int, n_nodes, reward_function, order, action_space) -> PPO:
    """
    Load a trained PPO agent from a zip file.

    Args:
        env (VecEnv): The environment to use for loading the agent.
        n_agent_index (int): The index of the agent to load.

    Returns:
        PPO: The loaded PPO agent.
    """
    agent_path = get_trained_agent_path(n_agent_index, n_nodes, reward_function, order, action_space)
    return PPO.load(agent_path, env)


def get_trained_agent_path(n_agent_index: int, n_nodes, reward_function, order, action_space) -> str:
    """
    Get the file path of the trained PPO agent.

    Args:
        n_agent_index (int): The index of the agent.

    Returns:
        str: The file path of the trained agent.
    """
    base_path = f'{MODEL_LOCATION}/train_log/{action_space}/{reward_function}' \
                f'/{n_nodes}_nodes/sb3_EpLen100_{order}_Skill1_Vul1_run_' \
                f'{n_agent_index + 1}'

    # Get all files in the directory
    files = os.listdir(base_path)

    # Check for "ppo.zip" (without a number)
    ppo_no_number = [f for f in files if f == "ppo.zip"]

    # Get files that match the pattern "ppo_*.zip" (with numbers)
    ppo_with_numbers = [f for f in files if re.match(r'ppo_\d+\.zip', f)]

    # If there's no "ppo.zip" or if there are other numbered files, return the highest numbered file
    if ppo_with_numbers:
        ppo_with_numbers.sort(key=lambda x: int(re.search(r'ppo_(\d+)\.zip', x).group(1)), reverse=True)
        return os.path.join(base_path, ppo_with_numbers[0])

    # If there are no numbered files but "ppo.zip" exists, return it
    if ppo_no_number:
        return os.path.join(base_path, ppo_no_number[0])

    # If no valid PPO files are found, raise an exception
    raise FileNotFoundError(f"No PPO files found in directory: {base_path}")


# Reliabilty Assessment
def calc_cvar(run_rollouts):
    """ Takes in the 1000 rollouts for each run evaluated.

    returns upper and lower CVaR for each rollout """

    from rl_reliability_metrics.metrics import metrics_offline

    lower_cvar_metric = metrics_offline.LowerCVaRAcrossRollouts()
    upper_cvar_metric = metrics_offline.UpperCVaRAcrossRollouts()

    lower_cvar_vals = lower_cvar_metric(run_rollouts)
    upper_cvar_vals = upper_cvar_metric(run_rollouts)

    return lower_cvar_vals, upper_cvar_vals


def evaluate_agent(agent: PPO, env, red, blue, network_interface, order, deterministic: bool = True, \
    current_episode:
int =
0):
    """
    Evaluate a trained agent in the given environment.

    Args:
        agent (PPO): The trained PPO agent.
        env (VecEnv): The environment in which to evaluate the agent.
        deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.

    Returns:
        tuple: A tuple containing the mean reward, standard deviation of rewards, action counts for Blue and Red,
               a list of compromised counts, dictionaries for Blue and Red action nodes, and a dictionary for
               tracking how many steps each node was compromised.
    """

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")

    # Divides episodes among different sub-environments in the vector as evenly as possible
    episode_count_targets = np.array([(1 + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    step_rews = []

    compromised_counts = []
    action_counts_blue = {}
    action_counts_red = {}
    action_nodes_blue = {}
    action_nodes_red = {}

    # Initialize a dictionary to keep track of node compromise durations
    node_compromise_duration = {}

    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():

        actions, states = agent.predict(observations, state=states, episode_start=episode_starts,
                                        deterministic=deterministic)

        observations, rewards, dones, notes = env.step(actions)

        current_rewards += rewards
        step_rews.append(rewards)
        current_lengths += 1

        # Mid step node states
        mid_compromised_states = notes[0]['mid_step_info']['mid_step']['mid_state_compromised']
        mid_compromised_count = sum(mid_compromised_states.values())

        # Extract node states
        compromised_states = notes[0]['end_state']
        compromised_count = sum(compromised_states.values())

        if EVAL_TYPE == "intra_step_eval":
            compromised_counts.append(max(mid_compromised_count, compromised_count))
        elif EVAL_TYPE == "stepwise_eval":
            compromised_counts.append(compromised_count)

        # Iterate through all nodes in both mid-step and end-step states
        all_nodes = set(mid_compromised_states.keys()).union(compromised_states.keys())

        for node in all_nodes:
            # Check if node is compromised in either mid-step or end-step
            is_mid_compromised = mid_compromised_states.get(node, 0)
            is_end_compromised = compromised_states.get(node, 0)

            # Initialize node in the compromise duration dictionary if not present
            if node not in node_compromise_duration:
                node_compromise_duration[node] = 0

            # Update the compromise duration if the node is compromised in either state
            if is_mid_compromised or is_end_compromised:
                node_compromise_duration[node] += 1

        blue_action = notes[0]['blue_action']
        red_action = notes[0]['red_info'][0]['Action']

        # Blue agent action tracking
        action_counts_blue[blue_action] = action_counts_blue.get(blue_action, 0) + 1
        blue_node = notes[0]['blue_node']

        if blue_action not in action_nodes_blue:
            action_nodes_blue[blue_action] = {}
        if blue_node not in action_nodes_blue[blue_action]:
            action_nodes_blue[blue_action][blue_node] = 0
        action_nodes_blue[blue_action][blue_node] += 1

        # Red agent action tracking
        action_counts_red[red_action] = action_counts_red.get(red_action, 0) + 1
        red_nodes = notes[0]['red_info'][0]['Target_Nodes']  # This is a list

        # Ensure red_action exists in action_nodes_red dictionary
        if red_action not in action_nodes_red:
            action_nodes_red[red_action] = {}

        # Iterate over each node in the red_nodes list (if it's not empty)
        for red_node in red_nodes:
            if red_node.name not in action_nodes_red[red_action]:
                action_nodes_red[red_action][red_node.name] = 0
            action_nodes_red[red_action][red_node.name] += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                observation = observations[i]
                reward = rewards[i]
                done = dones[i]
                info = notes[i]
                episode_starts[i] = done

                if dones[i]:
                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, action_counts_blue, action_counts_red, compromised_counts, action_nodes_blue, \
        action_nodes_red, node_compromise_duration, step_rews


def average_dicts(dicts_list: list) -> dict:
    """
    Average the values of a list of dictionaries.

    Args:
        dicts_list (list): A list of dictionaries to average.

    Returns:
        dict: A dictionary with averaged values.
    """
    average_dict = {}
    total_dict = {}

    for dictionary in dicts_list:
        for key, value in dictionary.items():
            if key in total_dict:
                total_dict[key] += value
            else:
                total_dict[key] = value

    for key, value in total_dict.items():
        average_dict[key] = value / len(dicts_list)

    return average_dict


def calculate_average_compromised(compromised_counts: list, n_steps: int) -> float:
    """
    Calculate the average number of compromised nodes per episode.

    Args:
        compromised_counts (list): List of compromised node counts across all timesteps.
        n_steps (int): Number of timesteps in each episode.

    Returns:
        float: The average number of compromised nodes per episode.
    """
    # Reshape compromised_counts into episodes of n_steps timesteps each
    episodes = np.array(compromised_counts).reshape(-1, n_steps)

    # Calculate the average number of compromised nodes per episode
    average_compromised_per_episode = np.mean(np.sum(episodes, axis=1) / n_steps)

    return average_compromised_per_episode


def main(n_nodes, reward_function, reward_description, reward_type, order, action_space):
    """
    Main function to evaluate multiple trained agents on a network environment and plot the results.
    """
    # Updating the JSON with the evaluation settings above
    with open('Minimal_network_gamemode.json', 'r') as file:
        base_config = json.load(file)

    # Create a unique copy for each evaluation
    eval_config = copy.deepcopy(base_config)
    eval_config['game_rules']['max_steps'] = N_STEPS
    eval_config['rewards']['function'] = reward_function
    if action_space == 'simple_action_space':
        eval_config['blue']['action_set']['place_decoy'] = 'false'
        eval_config['red']['agent_attack']['always_succeeds'] = 'true'
    elif action_space == 'decoy_action_space':
        eval_config['blue']['action_set']['place_decoy'] = 'true'


    # Lists to accumulate metrics
    all_action_counts_blue = []
    all_action_counts_red = []
    all_total_rewards = []
    all_compromised_counts = []
    all_node_compromise_duration = []

    # Dictionary to store average metrics for each agent
    agent_avg_compromised_counts = {}

    # List to store CVaR values for each agent
    all_lower_cvar_vals = []  # Store lower CVaR values for all agents
    all_upper_cvar_vals = []  # Store upper CVaR values for all agents

    # Set a fixed base seed (can be passed in as a parameter)
    BASE_SEED = 12345
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    # Calculate the total number of episodes you'll run
    total_episodes = NO_AGENTS * N_EPISODES

    # Generate a list of unique seeds (e.g., using random.randint)
    episode_seeds = [random.randint(0, 1_000_000) for _ in range(total_episodes)]

    # Now, when looping over agents and episodes, pick a seed from this list:
    seed_index = 0
    # Evaluate multiple agents
    for agent_index in tqdm(range(NO_AGENTS)):
        agent_compromised_counts = []  # To store compromised counts for this agent
        agent_rollouts = []  # To store rollouts for this agent
        for episode in range(N_EPISODES):
            current_seed = episode_seeds[seed_index]
            seed_index += 1
            # Load game mode configuration, altered for this eval
            game_mode = GameMode.create(dict=eval_config)
            net_dict = GenerateSubproblemNetwork(num_nodes=n_nodes).network
            network = Network.create(network_dict=net_dict)
            network_interface = NetworkInterface(game_mode=game_mode, network=network)
            if action_space == 'simple_action_space':
                network_interface.game_mode.blue.action_set.place_decoy.value = False
            elif action_space == 'decoy_action_space':
                network_interface.game_mode.blue.action_set.place_decoy.value = True

            red = RedInterface(network_interface)
            blue = BlueInterface(network_interface)

            env = GenericNetworkEnv(agent_order=order, red_agent=red, blue_agent=blue, network_interface=network_interface)
            env.random_seed = current_seed
            gen_env = env

            # print(blue.action_dict)
            check_env(gen_env, warn=True)

            # Monitor environment
            gen_env = Monitor(gen_env)

            # Load trained agent
            agent = load_trained_agent(gen_env, agent_index, n_nodes, reward_function, order, action_space)

            mean_reward, std_reward, action_counts_blue, action_counts_red, compromised_counts, action_nodes_blue, \
                action_nodes_red, node_compromise_duration, step_rews = evaluate_agent(
                                                                                       agent, gen_env, red, blue,
                                                                                       network_interface, order,
                                                                                       deterministic=True
                                                                                       )


            avg_comp_counts = sum(compromised_counts) / len(compromised_counts)

            # Save rewards for this episode (eval_metric, not actual rewards)
            agent_rollouts.append(avg_comp_counts)

            all_action_counts_blue.append(action_counts_blue)
            all_action_counts_red.append(action_counts_red)
            all_compromised_counts.extend(compromised_counts)
            all_total_rewards.append(mean_reward)
            all_node_compromise_duration.append(node_compromise_duration)

            # Append this episode's compromised counts to the agent's list
            agent_compromised_counts.extend(compromised_counts)

        rollout_indices = np.arange(len(agent_rollouts))
        agent_rollouts_array = np.array([rollout_indices, agent_rollouts])

        # Calculate CVaR for this agent
        lower_cvar_vals, upper_cvar_vals = calc_cvar([agent_rollouts_array])  # Pass rollouts for this agent

        # Store CVaR values
        all_lower_cvar_vals.append(lower_cvar_vals[0])  # Store lower CVaR for this agent
        all_upper_cvar_vals.append(upper_cvar_vals[0])  # Store upper CVaR for this agent

        # Compute the average number of nodes compromised for this agent
        agent_avg_compromised = round(np.mean(agent_compromised_counts), 2)
        agent_avg_compromised_counts[agent_index] = agent_avg_compromised  # Store the value

    # Calculate the final averaged CVaR values
    avg_lower_cvar = np.mean(all_lower_cvar_vals)
    avg_upper_cvar = np.mean(all_upper_cvar_vals)

    print(f"Average Lower CVaR across all agents: {avg_lower_cvar}")
    print(f"Average Upper CVaR across all agents: {avg_upper_cvar}")

    total_rew_mean = np.mean(all_total_rewards)

    # Average the metrics
    avg_action_counts_blue = average_dicts(all_action_counts_blue)
    avg_action_counts_red = average_dicts(all_action_counts_red)
    avg_node_compromise_duration = average_dicts(all_node_compromise_duration)
    print(f"len(all_compromised_counts): {len(all_compromised_counts)}")

    avg_no_nodes_compromised = round(np.mean(all_compromised_counts), 2)

    print(f"avg compromised nodes per episode: {avg_no_nodes_compromised}")

    # Evaluation info dictionary
    eval_info = {
        'N_NODES': n_nodes,
        'REWARD_TYPE': reward_type,
        'N_EPISODES': N_EPISODES,
        'N_STEPS': N_STEPS,
        'AVERAGE_COMPROMISED': avg_no_nodes_compromised,
        'NODE_VULNERABILITY': NODE_VULNERABILITY,
        'RED_AGENT_SKILL': RED_AGENT_SKILL,
    }

    # Ensure the directory exists
    output_dir = f'{MODEL_LOCATION}/eval_log/{EVAL_TYPE}/{action_space}/{reward_function}/{n_nodes}_nodes'
    os.makedirs(output_dir, exist_ok=True)

    # Write the JSON data to the appropriate file
    with open(f'{output_dir}/{n_nodes}_Node_{order}_{reward_function}_avg_evaluation.json', 'w') as file:
        json.dump({
            'avg_nodes_compromised': avg_no_nodes_compromised,
            'avg_action_counts_blue': avg_action_counts_blue,
            'avg_action_counts_red': avg_action_counts_red,
            'avg_node_compromise_duration': avg_node_compromise_duration
        }, file, indent=4)

    log_file_path = f'{output_dir}/Eval_scores_per_agent_{order}.json'
    with open(log_file_path, 'w') as file:
        # Write the overall average at the top
        file.write(f"Overall Average Compromised Nodes: {avg_no_nodes_compromised}\n\n")
        # Write per-agent averages
        file.write("Per-Agent Average Compromised Nodes:\n")
        for agent_index, avg in agent_avg_compromised_counts.items():
            file.write(f"Agent {agent_index}: {avg}\n")

    log_file_path = f'{output_dir}/cvar_scores_per_agent_{order}.json'
    with open(log_file_path, 'w') as file:
        # Write the overall average at the top
        file.write(f"Overall Average Lower and Upper CVaR values: "
                   f"{round(avg_lower_cvar, 3)} & {round(avg_upper_cvar, 3)}\n\n")
        # Write per-agent averages
        file.write("Per-Agent Average Compromised Nodes:\n")
        file.write(f"All lower cvar values: {all_lower_cvar_vals}\n")
        file.write(f"All upper cvar values: {all_upper_cvar_vals}")

    # Ensure the directory for the second file exists
    scores_dir = f'{MODEL_LOCATION}/eval_log/{EVAL_TYPE}'
    os.makedirs(scores_dir, exist_ok=True)

    # Append evaluation scores to the log file
    with open(f'{scores_dir}/Eval_scores.json', 'a') as file:
        file.write(f'{n_nodes}_Node_{order}_{reward_function}_{action_space}: {avg_no_nodes_compromised}, '
                   f'{avg_lower_cvar}, {avg_upper_cvar}\n')

    # Print results
    print("Average Action Counts (Blue):", avg_action_counts_blue)
    print("Average Action Counts (Red):", avg_action_counts_red)
    print("Average Node Compromise Duration:", avg_node_compromise_duration)
    print(f'new evaluate reward average: {total_rew_mean}, standard deviation: {std_reward} ')

    # Plot and save the results
    chart_filename = f"{MODEL_LOCATION}/eval_log/{EVAL_TYPE}/{action_space}/{reward_function}/{n_nodes}_nodes/plotly" \
                     f"_{reward_function}_" \
                     f"Rewards_{order}_{n_nodes}_Nodes.html"

    print(f"Dashboard saved")

def parallel_evaluation():
    node_values = [5]  # Example values for N_NODES
    reward_types = [
        'Positive Rewards',
        'Negative Rewards',
        'Scaffolded Rewards',
        # 'Complex Dense Rewards',
        # 'Simple Positive and Negative Rewards'
    ]
    action_space_set = ['simple_action_space', 'decoy_action_space']  # 'simple_action_space', 'decoy_action_space'
    order = ['Red_Blue', 'Blue_Red', 'Balanced']  # 'Red_Blue', 'Blue_Red', 'Balanced'

    # Generate all combinations of nodes, reward types, action spaces, and orders
    tasks = list(product(node_values, reward_types, order, action_space_set))
    num_cpus = multiprocessing.cpu_count()
    # Create a pool of workers
    with multiprocessing.get_context("spawn").Pool(processes=num_cpus) as pool:
        pool.starmap(evaluate_combination, tasks)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Ensure compatibility across platforms
    # start logging time
    start_time = time.time()
    parallel_evaluation()
    # end logging time
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
