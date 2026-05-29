import multiprocessing
import os
import logging
import tempfile
import time
import datetime

from experiment_runner import run_experiment_with_logging  

os.environ["WANDB__SERVICE_WAIT"] = "300"

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
#os.environ["WANDB_MODE"] = "offline"  # Prevent online sync
os.environ["WANDB_SILENT"] = "True"    # Don't print to console

# Parameters of Training, edit to train what you want
NET_SHAPE = ['linear'] #
NODE_COMBINATIONS = [5]
REWARD_FUNCTIONS = ['scaffolded', 'complex_dense', 'simple_pos_neg', 'simple_positive', 'simple_negative']
# ['scaffolded', 'complex_dense', 'simple_pos_neg', 'simple_positive', 'simple_negative']
# Scaffolded refers to the Dense Negative reward function
N_STEPS = 100  # N steps in an episode
NODE_VULNERABILITY = 1
RED_AGENT_SKILL = 1
ORDER = ["Red_Blue", "Blue_Red", "Balanced"] # ["Red_Blue", "Blue_Red", "Balanced"]
EVAL_TYPE = "initial_eval"
ACTION_SPACE_SET = ["simple_action_space", "decoy_action_space"] # ["simple_action_space", "decoy_action_space"]
ALGO = ["PPO"] # ["DQN", "PPO"]
wandb_project_name = "YT_Reward_Engineering"
output_location= "Reward_Engineering/Models"
NO_RUNS = 10

# Create output location if it doesn't exist
if not os.path.exists(output_location):
    os.makedirs(output_location)


# Optional: Default to one device initially—this will be overridden per process.
GPU = 0
if GPU > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(GPU))

# Wrapper to set lower CPU priority - why?
def run_experiment_wrapper(*args, **kwargs):
    # try:
    #     os.nice(6)  # Increase the nice value to lower priority
    # except Exception:
    #     logging.error("Failed to set nice value: {}".format(e))
    run_experiment_with_logging(*args, **kwargs)

def run_parallel_experiments(n_nodes, reward_function, order, action_set, net_shape, algorithm, timesteps,
                             wandb_project_name, output_location):
    group_name = f'{n_nodes}_Nodes_{algorithm}_{net_shape}_YT_{order}_{reward_function}_{action_set}_Set_{N_STEPS}_Step_Episodes'

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Created temporary directory: {temp_dir}")

        processes = []
        MAX_PROCESSES = 10  # Try reducing further (e.g., 3 or 4) for better responsiveness, depending on your system

        for i in range(1, NO_RUNS+1):
            trial_name = f'sb3_EpLen{N_STEPS}_{order}_Skill1_Vul{NODE_VULNERABILITY}_run_{i}'
            logging.info(f"Starting process for trial {i}: {trial_name}")

            if GPU > 0:
                device_id = (i - 1) % GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

            # Use the wrapper that sets a lower nice value
            p = multiprocessing.Process(target=run_experiment_wrapper, args=(
                trial_name, group_name, EVAL_TYPE, n_nodes, reward_function, N_STEPS, NODE_VULNERABILITY,
                RED_AGENT_SKILL, timesteps, temp_dir, action_set, order, net_shape, algorithm, wandb_project_name, output_location))
            processes.append(p)
            p.start()

            while sum(1 for proc in processes if proc.is_alive()) >= MAX_PROCESSES:
                time.sleep(0.1)

        for p in processes:
            p.join()

        logging.info(f"Completed all experiments for {n_nodes} nodes with {reward_function} reward function.")

if __name__ == '__main__':
    # time.sleep(1200)
    # map number-of-nodes ➜ Training timesteps
    TIMESTEPS = {
        2: 500_000,
        5: 1_000_000,
        10: 1_500_000,
        20: 2_000_000,
        50: 2_500_000,
    }
    for algorithm in ALGO:
        for net_shape in NET_SHAPE:
            for action_set in ACTION_SPACE_SET:
                for n_nodes in NODE_COMBINATIONS:
                    try:
                        timesteps = TIMESTEPS[n_nodes]  # O(1) lookup
                    except KeyError:
                        raise ValueError(f"Invalid number of nodes: {n_nodes}")
                    for reward_function in REWARD_FUNCTIONS:
                        for order in ORDER:
                            print(f"Starting {algorithm} experiments for {n_nodes} {net_shape} nodes, "
                                  f"{reward_function} reward, {order} order, {timesteps} steps.")
                            print(f"Start time: {datetime.datetime.now()}")
                            run_parallel_experiments(n_nodes, reward_function, order, action_set, net_shape,
                                                     algorithm, timesteps, wandb_project_name, output_location)
                            print(f"Finish time: {datetime.datetime.now()}")