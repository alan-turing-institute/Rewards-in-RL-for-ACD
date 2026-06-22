import json
import random
from itertools import product
import sys
from pathlib import Path

# Resolve paths relative to this file so the repo works on any machine.
_HERE = Path(__file__).resolve().parent                # yawning_titan_training/Evaluation/
_YT_TRAINING_ROOT = _HERE.parent                       # yawning_titan_training/
_REPO_ROOT = _YT_TRAINING_ROOT.parent                 # repo root
_YT_SRC = _REPO_ROOT / "YAWNING-TITAN" / "src"

for _p in [str(_HERE), str(_YT_TRAINING_ROOT), str(_YT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.networks.network import Network
from Networks.N_node_generator import GenerateSubproblemNetwork
import matplotlib.pyplot as plt
import copy
import time
import multiprocessing
from utils import evaluate_combination
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

from rl_reliability_metrics.metrics import metrics_offline

######################## Evaluation Score ########################
# This script is used to evaluate the performance of the sets of agents according to the ground truth evaluation metric.

# To evaluate a model or set of models, first check the global vars below are applicable to all (N_STEPS, N_EPISODES,
# NO_AGENTS, ORDER, ACTION_SPACE_SET).

# Then go to the parallel_evaluation function and fill out the node_values (say 2, 5, 10, 20, 50) and reward_values (
# say 'Negative Rewards', 'Positive Rewards', 'Scaffolded Rewards') with the agents you want to evaluate in parallel.
# (See utils for the reward function names config)
# You should be good to run the script.

# Approx timings so far:
# 50 Nodes, 1000 episodes, 25 agents: 1 hour 35mins
# 100 Nodes, 1000 episodes, 25 agents: 3 hours 10mins


# These are the Global variables that are universal to all agent evaluations
NODE_VULNERABILITY = 1
RED_AGENT_SKILL = 1
N_STEPS = 100
N_EPISODES = 500
NO_AGENTS = 20 # No of agents to evaluate
EVAL_TYPE = "intra_step_eval"  # this version is evaluating intra step-wise - ground truth. Do not change unless you
# want


# step-wise evaluation.

# These can be manually altered, or added to the evaluate_combination and parallel_evaluation functions tp be automated
# ORDER = "Blue_Red"  # "Red_Blue", "Balanced"
# ACTION_SPACE_SET = 'simple_action_space'
# 'decoy_action_space'

# Select reward configuration
# REWARD_TYPE = 'Negative Rewards'
# REWARD_TYPE = 'Positive Rewards'
# REWARD_TYPE = 'Scaffolded Rewards'
# REWARD_TYPE = 'Costly Negative Rewards'
# REWARD_TYPE = 'Costly Positive Rewards'
# REWARD_TYPE = 'Costly Scaffolded Rewards'

# EVAL_TYPE = "intra_step_eval" # this version is at a higher level than the previous eval, it's above the agent order
# abstraction and looks at the mid step
# EVAL_TYPE = "stepwise_eval" # this version is evaluating step-wise


# REWARD_FUNCTION = REWARD_CONFIGS[REWARD_TYPE]['function']
# REWARD_DESCRIPTION = REWARD_CONFIGS[REWARD_TYPE]['description']

# Dynamic module import based on N_NODES
def import_network_module(n_nodes):
    try:
        module_name = f"Reward_Engineering.Networks.Minimal_Network_{n_nodes}Nodes"
        module = __import__(module_name, fromlist=['GenerateSubproblemNetwork'])
        return module.GenerateSubproblemNetwork
    except ImportError:
        raise ValueError(f"No module found for {n_nodes} nodes")


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


def load_trained_agent(env: VecEnv, n_agent_index: int, n_nodes, reward_function, order, action_space,
                       agent_dir=None, start_run=1) -> PPO:
    """
    Load a trained PPO agent from a zip file.

    Args:
        env (VecEnv): The environment to use for loading the agent.
        n_agent_index (int): The index of the agent to load.
        agent_dir (str, optional): Base directory containing agent run subdirectories.

    Returns:
        PPO: The loaded PPO agent.
    """
    agent_path = get_trained_agent_path(n_agent_index, n_nodes, reward_function, order, action_space,
                                        agent_dir=agent_dir, start_run=start_run)
    # print(f"Loading agent from: {agent_path}")
    return PPO.load(agent_path, env)


def get_trained_agent_path(n_agent_index: int, n_nodes, reward_function, order, action_space,
                           agent_dir=None, start_run=1) -> str:
    """
    Get the file path of the trained PPO agent.

    Args:
        n_agent_index (int): The index of the agent (0-based).
        agent_dir (str, optional): Base directory containing run subdirectories. When provided,
            scans for the matching run folder and picks the highest-numbered ppo_N.zip.

    Returns:
        str: The file path of the trained agent.
    """
    run_num = n_agent_index + start_run

    if agent_dir is not None:
        run_dir = None
        pattern = re.compile(rf'sb3_EpLen100_{re.escape(order)}_Skill\d+_Vul\d+_run_{run_num}$')
        for subdir in os.listdir(agent_dir):
            if pattern.match(subdir):
                run_dir = os.path.join(agent_dir, subdir)
                break
        if run_dir is None:
            raise FileNotFoundError(
                f"No run directory found for agent {run_num}, order '{order}' in {agent_dir}")
    else:
        run_dir = (f"ICML_dense_pos/train_log/{action_space}/dense_positive"
                   f"/{n_nodes}_nodes/PPO/sb3_EpLen100_{order}_Skill1_Vul1_run_{run_num}")

    files = os.listdir(run_dir)
    ppo_files = [f for f in files if re.match(r'ppo_\d+\.zip', f)]
    if not ppo_files:
        raise FileNotFoundError(f"No PPO zip files found in {run_dir}")

    # Use the highest-numbered checkpoint (final saved model)
    ppo_files.sort(key=lambda x: int(re.search(r'ppo_(\d+)\.zip', x).group(1)), reverse=True)
    return os.path.join(run_dir, ppo_files[0])


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

    # gen_env = GenericNetworkEnv(red_agent=red, blue_agent=blue, network_interface=network_interface,
    #                             Agent_order=order)
    # gen_env.random_seed = seed
    # env = gen_env


    # print(blue.action_dict)
    # check_env(env, warn=True)

    # Monitor environment
    # env = Monitor(env)

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
        # print("rewards: ", rewards)

        # print(f"Observations: {observations}")
        # print(f'Red action: {notes[0]["red_info"][0]["Action"]}, success: {notes[0]["red_info"][0]["Successes"]}')
        # print(f'Blue action: {notes[0]["blue_action"]}')
        # print(f'mid step compromise: {notes[0]["mid_step_info"]["mid_step"]["mid_state_compromised"]}')
        # print(f'Sum of mid step compromise: {sum(notes[0]["mid_step_info"]["mid_step"]["mid_state_compromised"].values())}')
        # print(f'End of step compromise: {notes[0]["end_state"]}')
        # print(f'Sum of end of step compromise: {sum(notes[0]["end_state"].values())}')
        # print("~~~~~~~~~~~~~~~~~~~~~~")

        current_rewards += rewards
        step_rews.append(rewards)
        current_lengths += 1

        # Mid step node states
        mid_compromised_states = notes[0]['mid_step_info']['mid_step']['mid_state_compromised']
        # print(f"mid_compromised_states: {mid_compromised_states}")
        mid_compromised_count = sum(mid_compromised_states.values())
        # print(f"mid_compromised_count: {mid_compromised_count}")

        # Extract node states
        compromised_states = notes[0]['end_state']
        # print(f'end compromised_states: {compromised_states}')
        compromised_count = sum(compromised_states.values())
        # print(f'end compromised_count: {compromised_count}')

        if EVAL_TYPE == "intra_step_eval":
            # print(f"mid_compromised_count: {mid_compromised_count}, compromised_count: {compromised_count}, "
            #       f"blue_action: {notes[0]['blue_action']}, red_action: {notes[0]['red_info'][0]['Action']}")
            # whichever is biggest, mid_compromised_count or compromised_count, append this to compromised_counts
            intra_step_comp = max(mid_compromised_count, compromised_count)
            # print(f"intra_step_comp: {intra_step_comp}")
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
        # print(f'blue_action: {blue_action}')
        red_action = notes[0]['red_info'][0]['Action']
        # print(f'red_action from end of step log: {red_action}, on nodes: {notes[0]["red_info"][0]["Target_Nodes"]}, '
        #       f'from nodes:'
        #       f' {notes[0]["red_info"][0]["Attacking_Nodes"]}')

        red_action_mid = notes[0]['mid_step_info']['mid_step']['Action']
        red_success_mid = notes[0]['mid_step_info']['mid_step']['Success']

        # print(f'red_action from mid step log: {red_action_mid}, success:{red_success_mid} on nodes:'
        #       f' {notes[0]["mid_step_info"]["mid_step"]["Target_Nodes"]}')
        # print("~~~~~~~~~~~~~~~~~~~~~~")

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

    # print(f'episode_rewards: {episode_rewards}')
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # pprint(node_compromise_duration)

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


def plot_network_diagram(network: Network):
    """
    Plot the network diagram using Plotly.

    Args:
        network (Network): The network to plot.

    Returns:
        tuple: A tuple containing the Plotly traces for edges, nodes, entry nodes, and normal nodes.
    """
    G = nx.Graph()
    entry_nodes = []

    for node in network.nodes:
        G.add_node(node.name, pos=node.node_position)
        if node.entry_node:
            entry_nodes.append(node.name)
    for edge in network.edges:
        G.add_edge(edge[0].name, edge[1].name)

    pos = nx.get_node_attributes(G, 'pos')
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Edges',
        legendgroup='network',
        legendrank=1
    )

    node_x = []
    node_y = []
    entry_node_x = []
    entry_node_y = []
    normal_node_x = []
    normal_node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if node in entry_nodes:
            entry_node_x.append(x)
            entry_node_y.append(y)
        else:
            normal_node_x.append(x)
            normal_node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=[],
            size=20,
            line_width=2),
        text=[node for node in G.nodes()],
        textposition="top center",
        showlegend=False  # Disable legend for the combined trace
    )

    entry_node_trace = go.Scatter(
        x=entry_node_x, y=entry_node_y,
        mode='markers',
        hoverinfo='none',
        marker=dict(
            showscale=False,
            color='red',
            size=20,
            line_width=2),
        name='Entry Node',
        legendgroup='network',
        legendrank=3
    )

    normal_node_trace = go.Scatter(
        x=normal_node_x, y=normal_node_y,
        mode='markers',
        hoverinfo='none',
        marker=dict(
            showscale=False,
            color='orange',
            size=20,
            line_width=2),
        name='Normal Node',
        legendgroup='network',
        legendrank=4
    )

    return edge_trace, node_trace, entry_node_trace, normal_node_trace


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


def plot_results(compromised_counts: list, action_counts_blue: dict, action_counts_red: dict, network: Network,
                 n_nodes, reward_type, order, action_space,
                 filename_1: str, filename_2: str, filename_3: str, action_nodes_blue: dict, action_nodes_red: dict):
    """
    Plot the results as a bar chart and pie charts using Plotly.

    Args:
        compromised_counts (list): A list of compromised counts across episodes.
        action_counts_blue (dict): A dictionary of action counts for the Blue agent.
        action_counts_red (dict): A dictionary of action counts for the Red agent.
        network (Network): The network structure.
        filename (str): The filename for saving the plot as an HTML file.
        action_nodes_blue (dict): A dictionary mapping Blue agent actions to the nodes on which they were performed.
        action_nodes_red (dict): A dictionary mapping Red agent actions to the nodes on which they were performed.
    """
    if action_space == 'simple_action_space':
        # Define a color mapping for actions
        action_color_mapping = {
            'do_nothing': 'yellow',
            'basic_attack': 'red',
            'restore_node': 'blue',
            'scan': 'purple',
            'no_possible_targets': 'grey'
        }

        # Define Blue and Red actions separately
        blue_actions = ['restore_node', 'scan']
        red_actions = ['basic_attack', 'do_nothing', 'no_possible_targets']

    elif action_space == 'decoy_action_space':
        # Define a color mapping for actions
        action_color_mapping = {
            'do_nothing': 'yellow',
            'basic_attack': 'red',
            'restore_node': 'blue',
            'scan': 'purple',
            'place_decoy': 'green',
            'no_possible_targets': 'grey'
        }

        # Define Blue and Red actions separately
        blue_actions = ['restore_node', 'scan', 'place_decoy']
        red_actions = ['basic_attack', 'do_nothing', 'no_possible_targets']
    else:
        print("Invalid action space set")

    print(f'action_color_mapping: {action_color_mapping}')
    print(f'blue_actions: {blue_actions}')
    print(f'red_actions: {red_actions}')

    # print(f'Action counts blue: {action_counts_blue}')

    # print(f'compromised_counts: {compromised_counts}')

    # Reshape compromised_counts into episodes of 100 timesteps each
    episodes = np.array(compromised_counts).reshape(-1, N_STEPS)

    # print(episodes)

    # Count occurrences of 0, 1, 2, etc... nodes compromised in each episode
    counts_per_episode = np.array([[np.sum(episode == i) for i in range(n_nodes + 1)] for episode in episodes])

    # print(f'counts_per_episode: {counts_per_episode}')

    # Calculate average counts over all episodes
    average_counts = np.mean(counts_per_episode, axis=0)

    # Bar chart data
    bar_labels = [f'{i}' for i in range(n_nodes + 1)]
    bar_counts = average_counts.tolist()

    # Pie chart data for blue actions with node information
    blue_labels = []
    blue_values = []
    blue_hover_texts = []
    blue_colors = []

    for action in blue_actions:
        count = action_counts_blue.get(action, 0)
        blue_labels.append(action)
        blue_values.append(count)
        node_details = ', '.join([f"Node {node}: {node_count} times" for node, node_count in
                                  action_nodes_blue.get(action, {}).items()]) or "No specific nodes"
        blue_hover_texts.append(f"{count} times<br>{node_details}")
        blue_colors.append(action_color_mapping.get(action, 'grey'))

    # Pie chart data for red actions with node information
    red_labels = []
    red_values = []
    red_hover_texts = []
    red_colors = []

    for action in red_actions:
        count = action_counts_red.get(action, 0)
        red_labels.append(action)
        red_values.append(count)
        node_details = ', '.join([f"Node {node}: {node_count} times" for node, node_count in
                                  action_nodes_red.get(action, {}).items()]) or "No specific nodes"
        red_hover_texts.append(f"{count} times<br>{node_details}")
        red_colors.append(action_color_mapping.get(action, 'grey'))

    average_compromised_per_episode = round(np.mean(np.sum(episodes, axis=1) / N_STEPS), 2)

    # Create a figure with the bar chart spanning two rows and pie charts stacked on the right
    fig = sp.make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy", "rowspan": 2}, {"type": "domain"}],  # Bar chart spans two rows
               [None, {"type": "domain"}]],  # Pie charts stacked on the right
        row_heights=[0.5, 0.5],  # Equal heights for the pie charts
        column_widths=[0.6, 0.4],  # Wider left column for bar chart, narrower right for pie charts
        subplot_titles=("Average Nodes Compromised", "Blue Actions % Distribution", "Red Actions % Distribution")
    )

    # Add the bar chart for compromised node counts (spanning two rows)
    fig.add_trace(
        go.Bar(x=bar_labels, y=bar_counts, name='Node Compromise States', marker_color='orange'),
        row=1, col=1
    )

    # Add pie chart for blue actions (on top of the second column)
    fig.add_trace(
        go.Pie(
            labels=blue_labels,
            values=blue_values,
            name='Blue Actions',
            hole=0.3,
            showlegend=True,
            textinfo='percent',
            textposition='inside',
            hoverinfo='label+text',
            hovertext=blue_hover_texts,
            marker=dict(colors=blue_colors)
        ),
        row=1, col=2
    )

    # Add pie chart for red actions (below the blue pie chart in the second column)
    fig.add_trace(
        go.Pie(
            labels=red_labels,
            values=red_values,
            name='Red Actions',
            hole=0.3,
            showlegend=True,
            textinfo='percent',
            textposition='inside',
            hoverinfo='label+text',
            hovertext=red_hover_texts,
            marker=dict(colors=red_colors)
        ),
        row=2, col=2
    )

    # Update layout with titles, axis labels, and background box
    fig.update_layout(
        title_text=f'Model Evaluation - {n_nodes} Nodes - {reward_type} Rewards',
        title_font=dict(size=20),
        height=600,
        margin=dict(t=100),
        yaxis_title='Average Step Count per Episode',  # Adding y-axis title
        xaxis_title='Nodes Compromised at once',  # Adding x-axis title
        showlegend=True,  # Enable the legend
        legend=dict(  # Let the legend auto-position itself within the plot
            orientation='v',  # Vertical orientation of the legend
            yanchor='top',  # Align the legend at the top
            xanchor='left',  # Align the legend on the left inside the plot
        )
    )

    # # Adjust the position of only the pie chart titles
    # for annotation in fig['layout']['annotations']:
    #     if annotation['text'] in ["Blue Actions % Distribution", "Red Actions % Distribution"]:
    #         annotation['y'] += 0.05  # Shift these titles upward, adjust the value as needed

    # Update individual legends
    fig.update_traces(
        selector=dict(type='bar'),
        showlegend=True,
        legendgroup='bar',
        legendgrouptitle=dict(text='Node Compromise States')
    )
    fig.update_traces(
        selector=dict(type='pie', name='Blue Actions'),
        showlegend=True,
        legendgroup='blue',
        legendgrouptitle=dict(text='Blue Actions % Distribution')
    )
    fig.update_traces(
        selector=dict(type='pie', name='Red Actions'),
        showlegend=True,
        legendgroup='red',
        legendgrouptitle=dict(text='Red Actions % Distribution')
    )

    # Save the plotly figure as HTML
    if order == "Red_Blue":
        pio.write_html(fig, filename_1)
        print(f"Plotly chart saved as {filename_1}")

    if order == "Blue_Red":
        pio.write_html(fig, filename_3)
        print(f"Plotly chart saved as {filename_3}")

    if order == "Balanced":
        pio.write_html(fig, filename_1)
        print(f"Plotly chart saved as {filename_1}")

    # Show the figure
    fig.show()


def append_script_to_html(filename):
    """
    Appends a JavaScript snippet to the bottom of an HTML file.

    Args:
        filename (str): The path to the HTML file.
    """
    script = """
    <script>
      function sendHeight() {
        var height = document.body.scrollHeight;
        parent.postMessage({ height: height }, '*');
      }
      window.onload = sendHeight;
      window.onresize = sendHeight;
    </script>
    """

    with open(filename, 'a') as file:
        file.write(script)


def main(n_nodes, reward_function, reward_description, reward_type, order, action_space,
         agent_dir=None, eval_base_dir=None, eval_scores_root=None, n_agents=None, start_run=1):
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

    n_agents_actual = n_agents if n_agents is not None else NO_AGENTS

    # Calculate the total number of episodes you'll run
    total_episodes = n_agents_actual * N_EPISODES

    # Generate a list of unique seeds (e.g., using random.randint)
    episode_seeds = [random.randint(0, 1_000_000) for _ in range(total_episodes)]

    # Now, when looping over agents and episodes, pick a seed from this list:
    seed_index = 0
    # Evaluate multiple agents
    for agent_index in tqdm(range(n_agents_actual)):
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

            # print(
            #     f'First : n_nodes: {n_nodes}, reward_function: {reward_function}, order: {order}, action_space: {action_space}')

            # Load trained agent
            agent = load_trained_agent(gen_env, agent_index, n_nodes, reward_function, order, action_space,
                                       agent_dir=agent_dir, start_run=start_run)

            mean_reward, std_reward, action_counts_blue, action_counts_red, compromised_counts, action_nodes_blue, \
                action_nodes_red, node_compromise_duration, step_rews = evaluate_agent(
                                                                                       agent, gen_env, red, blue,
                                                                                       network_interface, order,
                                                                                       deterministic=True
                                                                                       )

            # print(f'step_rews: {step_rews}')
            # print(f"Compromised counts: {compromised_counts}")
            avg_comp_counts = sum(compromised_counts) / len(compromised_counts)
            # print(f'mean rewards: {mean_reward}')
            # print(f'avg compromised counts: {avg_comp_counts}')

            # Save rewards for this episode (eval_metric, not actual rewards)
            agent_rollouts.append(avg_comp_counts)

            all_action_counts_blue.append(action_counts_blue)
            all_action_counts_red.append(action_counts_red)
            all_compromised_counts.extend(compromised_counts)
            all_total_rewards.append(mean_reward)
            all_node_compromise_duration.append(node_compromise_duration)

            # Append this episode's compromised counts to the agent's list
            agent_compromised_counts.extend(compromised_counts)

        # Save agent rollouts in the required 2D numpy format
        # Create the required 2D numpy array for CVaR
        # Save agent rollouts in the required 2D numpy format
        # print(f'Agent {agent_index} rollouts: {agent_rollouts}')
        rollout_indices = np.arange(len(agent_rollouts))
        agent_rollouts_array = np.array([rollout_indices, agent_rollouts])

        # Calculate CVaR for this agent
        lower_cvar_vals, upper_cvar_vals = calc_cvar([agent_rollouts_array])  # Pass rollouts for this agent
        # print(f'Agent {agent_index} lower cvar: {lower_cvar_vals}, upper cvar: {upper_cvar_vals}')

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

    # print(f'all total rewards: {all_total_rewards}')
    total_rew_mean = np.mean(all_total_rewards)

    # pprint(all_node_compromise_duration)

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

    if eval_base_dir is not None:
        # eval_base_dir is already the full mirrored path, use it directly
        output_dir = eval_base_dir
        _scores_dir_default = eval_scores_root if eval_scores_root is not None else eval_base_dir
    else:
        output_dir = f'ICML_dense_pos/eval_log/{EVAL_TYPE}/{action_space}/{reward_function}/{n_nodes}_nodes'
        _scores_dir_default = f'ICML_dense_pos/eval_log/{EVAL_TYPE}'

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nEval output directory: {os.path.abspath(output_dir)}")

    # Write the JSON data to the appropriate file
    with open(f'{output_dir}/{n_nodes}_Node_{order}_{reward_function}_avg_evaluation.json', 'w') as file:
        json.dump({
            'avg_nodes_compromised': avg_no_nodes_compromised,
            'avg_action_counts_blue': avg_action_counts_blue,
            'avg_action_counts_red': avg_action_counts_red,
            'avg_node_compromise_duration': avg_node_compromise_duration
        }, file, indent=4)

    log_file_path = f'{output_dir}/Eval_scores_per_agent_{order}.json'
    write_mode = 'a' if start_run > 1 else 'w'
    with open(log_file_path, write_mode) as file:
        if start_run == 1:
            file.write(f"Overall Average Compromised Nodes (ScoreGT): {avg_no_nodes_compromised}\n\n")
            file.write("Per-Agent ScoreGT (avg compromised nodes):\n")
        for agent_index, avg in agent_avg_compromised_counts.items():
            file.write(f"Agent {agent_index + start_run}: {avg}\n")

    print(f"Per-agent ScoreGT saved to: {os.path.abspath(log_file_path)}")

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
    scores_dir = _scores_dir_default
    os.makedirs(scores_dir, exist_ok=True)

    # Append evaluation scores to the log file
    scores_file = f'{scores_dir}/Eval_scores.json'
    with open(scores_file, 'a') as file:
        file.write(f'{n_nodes}_Node_{order}_{reward_function}_{action_space}: {avg_no_nodes_compromised}, '
                   f'{avg_lower_cvar}, {avg_upper_cvar}\n')

    # Print results
    print("Average Action Counts (Blue):", avg_action_counts_blue)
    print("Average Action Counts (Red):", avg_action_counts_red)
    print("Average Node Compromise Duration:", avg_node_compromise_duration)
    print(f'new evaluate reward average: {total_rew_mean}, standard deviation: {std_reward} ')

    # Print ScoreGT summary
    print(f"\n=== ScoreGT Summary ({order}) ===")
    print(f"Average ScoreGT across all agents: {avg_no_nodes_compromised}")
    print("Per-Agent ScoreGT (avg compromised nodes, lower is better):")
    for idx, avg in sorted(agent_avg_compromised_counts.items()):
        print(f"  Agent {idx + 1}: {avg}")
    best_agent = min(agent_avg_compromised_counts, key=agent_avg_compromised_counts.get)
    print(f"Best agent: Agent {best_agent + 1} (ScoreGT: {agent_avg_compromised_counts[best_agent]})")
    print(f"Scores appended to: {os.path.abspath(scores_file)}")

    # Plot and save the results
    chart_filename = os.path.join(output_dir, f"plotly_{reward_function}_Rewards_{order}_{n_nodes}_Nodes.html")

    plot_results(
        all_compromised_counts,
        avg_action_counts_blue,
        avg_action_counts_red,
        network,
        n_nodes,
        reward_type,
        order,
        action_space,
        filename_1=chart_filename,
        filename_2=os.path.join(output_dir, f"{reward_function}_Rewards_{order}_{n_nodes}_Nodes_dashboard.html"),
        filename_3=chart_filename,
        action_nodes_blue=action_nodes_blue,
        action_nodes_red=action_nodes_red,
    )

    # Generate HTML report with dynamic content
    generate_html_from_template(
        template_path='evaluation_template.html',
        output_paths=[os.path.join(output_dir, f'{reward_function}_Rewards_{order}_{n_nodes}_Nodes.html')],
        chart_filename=chart_filename,
        eval_info=eval_info
    )

    print(f"Dashboard saved")


def generate_html_from_template(template_path, output_paths, chart_filename, eval_info):
    """
    Generate an HTML file by embedding the chart contents and evaluation information into the template.

    Args:
        template_path (str): Path to the HTML template.
        output_paths (str or list): Path(s) to save the final HTML file(s).
        chart_filename (str): Filename of the generated Plotly chart.
        eval_info (dict): Dictionary containing evaluation information (e.g., N_NODES, REWARD_TYPE).
    """
    # Read the contents of the template file
    with open(template_path, 'r') as template_file:
        html_content = template_file.read()

    # Read the contents of the Plotly chart file
    with open(chart_filename, 'r') as chart_file:
        chart_content = chart_file.read()

    # Replace the placeholders in the template with the actual evaluation information
    html_content = html_content.replace('{{N_NODES}}', str(eval_info['N_NODES']))
    html_content = html_content.replace('{{REWARD_TYPE}}', eval_info['REWARD_TYPE'])
    html_content = html_content.replace('{{N_EPISODES}}', str(eval_info['N_EPISODES']))
    html_content = html_content.replace('{{N_STEPS}}', str(eval_info['N_STEPS']))
    html_content = html_content.replace('{{AVERAGE_COMPROMISED}}', str(eval_info['AVERAGE_COMPROMISED']))
    html_content = html_content.replace('{{NODE_VULNERABILITY}}', str(eval_info['NODE_VULNERABILITY']))
    html_content = html_content.replace('{{RED_AGENT_SKILL}}', str(eval_info['RED_AGENT_SKILL']))

    # Replace the placeholder for the Plotly chart with the actual chart content
    html_content = html_content.replace('<div class="plot-container" id="plotly-chart"></div>', chart_content)

    # Ensure output_paths is a list for consistency
    if isinstance(output_paths, str):
        output_paths = [output_paths]

    # Save the final HTML file(s) to each specified path
    for path in output_paths:
        # Create the directory if it does not exist
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # # Write the content to the file
        with open(path, 'w') as output_file:
            output_file.write(html_content)


def parallel_evaluation():
    node_values = [10]  # Example values for N_NODES
    reward_types = [
        'Positive Rewards',
        # 'Negative Rewards',
        # 'Scaffolded Rewards',
        # 'Complex Dense Rewards',
        # 'Simple Positive and Negative Rewards'
    ]
    action_space_set = ['simple_action_space', 'decoy_action_space']  # 'simple_action_space', 'decoy_action_space'
    order = ['Blue_Red', 'Red_Blue', 'Balanced']  # 'Red_Blue', 'Blue_Red', 'Balanced'

    # Generate combinations of nodes and reward types
    # tasks = [(n_nodes, reward_type) for n_nodes in node_values for reward_type in reward_types]
    # Generate all combinations of nodes, reward types, action spaces, and orders
    tasks = list(product(node_values, reward_types, order, action_space_set))
    num_cpus = multiprocessing.cpu_count()
    # num_cpus = /
    # Create a pool of workers
    with multiprocessing.get_context("spawn").Pool(processes=num_cpus) as pool:
        pool.starmap(evaluate_combination, tasks)


def _parse_agent_dir(agent_dir, start_run=1, end_run=None):
    """
    Parse a PPO agent directory path and return (action_space, n_nodes, eval_base_dir, orders_and_counts).

    Expects a path of the form:
        .../train_log/<action_space>/<ablation>/<N>_nodes/PPO

    Returns:
        action_space (str)
        n_nodes (int)
        eval_base_dir (str)  — sibling eval_log directory
        orders_and_counts (dict) — {order_str: n_agents}
    """
    from pathlib import Path
    parts = Path(os.path.abspath(agent_dir)).parts

    try:
        tl_idx = parts.index('train_log')
    except ValueError:
        raise ValueError(f"'train_log' not found in path: {agent_dir}")

    # base dir is the parent of train_log (e.g. .../ICLR_Rebuttal_SP_Ablations)
    base_dir = str(Path(*parts[:tl_idx]))
    after_tl = parts[tl_idx + 1:]  # (action_space, ablation, N_nodes, PPO)

    if len(after_tl) < 3:
        raise ValueError(f"Expected at least 3 path components after 'train_log', got: {after_tl}")

    action_space = after_tl[0]
    n_nodes_str = after_tl[2]  # e.g. '10_nodes'
    n_nodes = int(n_nodes_str.replace('_nodes', ''))

    # Mirror the full path under eval_log instead of train_log
    eval_base_dir = os.path.join(base_dir, 'eval_log', *after_tl)
    eval_scores_root = os.path.join(base_dir, 'eval_log')

    # Scan subdirectories for run folders and group by order
    orders_and_counts = {}
    run_pattern = re.compile(r'sb3_EpLen100_(.+?)_Skill\d+_Vul\d+_run_(\d+)$')
    for subdir in os.listdir(agent_dir):
        m = run_pattern.match(subdir)
        if m:
            order = m.group(1)
            run_num = int(m.group(2))
            if run_num < start_run:
                continue
            if end_run is not None and run_num > end_run:
                continue
            orders_and_counts[order] = orders_and_counts.get(order, 0) + 1

    if not orders_and_counts:
        raise ValueError(f"No agent run directories found in {agent_dir}")

    return action_space, n_nodes, eval_base_dir, eval_scores_root, orders_and_counts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate a set of trained agents in a given directory.')
    parser.add_argument('--agent_dir', type=str, default=None,
                        help='Path to the PPO directory containing agent run subdirectories, e.g. '
                             'ICLR_Rebuttal_SP_Ablations/train_log/decoy_action_space/'
                             'simple_positive_ablation_6/10_nodes/PPO')
    parser.add_argument('--reward_type', type=str, default='Positive Rewards',
                        help='Reward type string (default: "Positive Rewards")')
    parser.add_argument('--reward_function', type=str, default=None,
                        help='Raw reward function name, bypasses --reward_type lookup (e.g. sp_sweep_c_neg050)')
    parser.add_argument('--n_episodes', type=int, default=None,
                        help='Override the number of evaluation episodes per agent')
    parser.add_argument('--start_run', type=int, default=1,
                        help='First run number to evaluate (default: 1). Use 16 to eval only runs 16+.')
    parser.add_argument('--end_run', type=int, default=None,
                        help='Last run number to evaluate (default: all). Use 30 to eval only up to run 30.')
    cli_args = parser.parse_args()

    multiprocessing.set_start_method('spawn')
    start_time = time.time()

    if cli_args.agent_dir is not None:
        if cli_args.n_episodes is not None:
            N_EPISODES = cli_args.n_episodes

        action_space, n_nodes, eval_base_dir, eval_scores_root, orders_and_counts = _parse_agent_dir(
            cli_args.agent_dir, start_run=cli_args.start_run, end_run=cli_args.end_run)

        reward_configs = {
            'Negative Rewards':                   {'function': 'simple_negative'},
            'Positive Rewards':                   {'function': 'simple_positive'},
            'Scaffolded Rewards':                 {'function': 'scaffolded'},
            'Complex Dense Rewards':              {'function': 'complex_dense'},
            'Simple Positive and Negative Rewards': {'function': 'simple_pos_neg'},
        }
        if cli_args.reward_function is not None:
            reward_function = cli_args.reward_function
            reward_type = cli_args.reward_type if cli_args.reward_type != 'Positive Rewards' else reward_function
        else:
            reward_type = cli_args.reward_type
            if reward_type not in reward_configs:
                raise ValueError(f"Unknown reward_type '{reward_type}'. Choose from: {list(reward_configs)}")
            reward_function = reward_configs[reward_type]['function']

        print(f"Agent directory : {os.path.abspath(cli_args.agent_dir)}")
        print(f"action_space    : {action_space}")
        print(f"n_nodes         : {n_nodes}")
        print(f"reward_type     : {reward_type} ({reward_function})")
        print(f"Orders found    : {orders_and_counts}")
        print(f"Eval output base: {os.path.abspath(eval_base_dir)}")
        print()

        for order, n_agents in orders_and_counts.items():
            print(f"--- Evaluating order '{order}' ({n_agents} agents, runs {cli_args.start_run}–{cli_args.end_run or 'end'}) ---")
            main(
                n_nodes=n_nodes,
                reward_function=reward_function,
                reward_description=reward_type,
                reward_type=reward_type,
                order=order,
                action_space=action_space,
                agent_dir=cli_args.agent_dir,
                eval_base_dir=eval_base_dir,
                eval_scores_root=eval_scores_root,
                n_agents=n_agents,
                start_run=cli_args.start_run,
            )
    else:
        parallel_evaluation()

    end_time = time.time()
    print(f"\nTime taken: {end_time - start_time:.1f} seconds")
