import os
import random
import json
import shutil
import wandb
import logging
import tempfile
import sys
from stable_baselines3.common.env_checker import check_env
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from icml2026_codebase.Reward_Function_Experimentation.Training.yawning_titan_run_wandb import YawningTitanRun
from yawning_titan.networks.network import Network
from icml2026_codebase.Reward_Function_Experimentation.Network.N_node_generator import GenerateSubproblemNetwork
from stable_baselines3.common.monitor import Monitor
from pprint import pprint

# This file is just to interact with the environment to check actions and outputs. This doesn't use any trained blue
# agents, it is like using the keyboard agent against the red agent.

N_NODES = 2
NODE_VULNERABILITY = 1
RED_AGENT_SKILL = 1
N_STEPS = 2

config = open(f'Minimal_network_gamemode.json')
config_dict = json.load(config)

game_mode = GameMode.create(dict=config_dict)

net_dict = GenerateSubproblemNetwork(num_nodes=N_NODES).network
network = Network.create(network_dict=net_dict)

network_interface = NetworkInterface(game_mode=game_mode, network=network)

red = RedInterface(network_interface)
blue = BlueInterface(network_interface)

env = GenericNetworkEnv(red_agent=red, blue_agent=blue, network_interface=network_interface)
check_env(env, warn=True)

env = Monitor(env)

env.reset()

for step in range(N_STEPS):
    if step == 0:
        action = 1
    else:
        action = 1

    observations, rewards, dones, notes = env.step(action)

    # pprint(notes)
    print("Mid step info:")
    print(f"Agent: {notes['mid_step_info']['mid_step']['Agent']}")
    print(f"Node: {notes['mid_step_info']['mid_step']['Target_Nodes']}")
    print(f"Action: {notes['mid_step_info']['mid_step']['Action']}")
    print(f"Success: {notes['mid_step_info']['mid_step']['Success']}")
    print(f"Reward: {notes['mid_step_info']['mid_step']['Reward']}")
    print(f"mid-Observations: {notes['mid_step_info']['mid_step']['observation']}")
    print(f"mid-compromised states: {notes['mid_step_info']['mid_step']['mid_state_compromised']}")

    print("~~~~")

    print(f"End Observations: {observations}")
    print(f"End Rewards: {rewards}")
    print(f"End blue action: {notes['blue_action']}, on the node: {notes['blue_node']}")
    print(f"End red action: {notes['red_info'][0]['Action']} on the node: {notes['red_info'][0]['Target_Nodes']} with "
          f"success: {notes['red_info'][0]['Successes']}")
    print(f"End state: {notes['end_state'].items()}")
    print("------------------")

