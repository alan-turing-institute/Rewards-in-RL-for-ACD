import wandb
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from rl_reliability_metrics.metrics import metrics_online

################## Reliability Evaluation  ##################

# To perform a reliability evaluation, fill in the entity, project, and group_name variables with the appropriate
# values in wandb (you'll have to login to wandb first).
# The script will fetch the reward data for all runs in the specified group and calculate the IQR across time and
# runs and store the results in the save_directory location below - adjust as needed.

# The script fetches reward data from W&B runs, computes IQR across time and runs,
# and saves results to save_directory.

# ═══════════════════════════════════════════════════════════
# CONFIGURATION — fill in your W&B details before running.
# ═══════════════════════════════════════════════════════════
nodes_evaluated = '5_Nodes'      # tag used to filter W&B run groups, e.g. '5_Nodes', '10_Nodes'
entity = ""                       # your W&B username or team name
project = "YT-Rewards-in-RL-for-ACD"  # your W&B project name
save_directory = f"../../results/IQR_Results/eval_log_{nodes_evaluated}_runs"
# ═══════════════════════════════════════════════════════════
os.makedirs(save_directory, exist_ok=True)

# Initialize the API
api = wandb.Api()

# Fetch all runs in the project
runs = api.runs(f"{entity}/{project}")

# Extract unique group names
group_names = set(run.group for run in runs if run.group and nodes_evaluated in run.group and 'Sweep' not in run.group)

# Define IQR calculation functions
def iqr_across_time(curves):
    metric = metrics_online.IqrWithinRuns()
    return metric(curves)


def iqr_across_runs(curves):
    if not curves or len(curves) == 0:
        raise ValueError("No learning curves provided.")

    # Truncate each curve to the last 20%
    truncated_curves = [curve[:, int(len(curve[1]) * 0.8):] for curve in curves]
    print(f"Number of truncated curves: {len(truncated_curves)}")
    print("truncated_curves shapes:", [curve.shape for curve in truncated_curves])
    print("truncated_curves sample data:", [curve[:, :5] for curve in truncated_curves])  # Print first 5 columns of each curve

    # Compute the IQR across runs
    metric = metrics_online.IqrAcrossRuns()
    iqr_across_runs_values = metric(truncated_curves)  # This should be a 1-D or 2-D array of IQR values
    print("IQR values across runs before normalization:", iqr_across_runs_values)

    # Normalize between 0 and 1
    min_val = np.min(iqr_across_runs_values)
    print("Min IQR value across runs:", min_val)
    max_val = np.max(iqr_across_runs_values)
    print("Max IQR value across runs:", max_val)

    if np.isclose(min_val, max_val):
        # If all values are the same, normalization would lead to division by zero
        # One simple fallback is to set the normalized array to zeros.
        normalized = np.zeros_like(iqr_across_runs_values)
    else:
        # Standard min-max normalization
        normalized = (iqr_across_runs_values - min_val) / (max_val - min_val)

    print("Normalized IQR values across runs:", normalized)

    # Return the mean of the normalized values
    return np.mean(normalized)


# Process each group
for group_name in tqdm(group_names):
    print(f"Processing group: {group_name}")

    # Filter runs by group name
    group_runs = [run for run in runs if run.group == group_name]

    # Determine the maximum training steps dynamically
    max_training_steps = 0
    for run in group_runs:
        history = run.history(keys=["global_step"], pandas=False)
        if history:
            steps = max(entry.get("global_step", 0) for entry in history)
            max_training_steps = max(max_training_steps, steps)

    if max_training_steps == 0:
        print(f"No valid training steps found for group {group_name}. Skipping.")
        continue

    reward_data_len = int(max_training_steps / 2048)  # Adjust reward data length dynamically
    print(f"Max training steps for group {group_name}: {max_training_steps}")

    # Extract data for each run
    all_rewards = {}
    for run in group_runs:
        history = run.history(samples=reward_data_len, keys=["rollout/ep_rew_mean"], pandas=False)
        mean_rewards = [entry["rollout/ep_rew_mean"] for entry in history if "rollout/ep_rew_mean" in entry]
        all_rewards[run.name] = {"rollout/ep_rew_mean": mean_rewards}

    # Prepare learning curves
    learning_curves = []
    run_names = []
    for run_name, rewards_dict in all_rewards.items():
        rewards = rewards_dict["rollout/ep_rew_mean"]
        if len(rewards) > 0:
            timepoints = np.linspace(0, max_training_steps, len(rewards))
            curve = np.vstack((timepoints, rewards))
            learning_curves.append(curve)
            run_names.append(run_name)

    # Calculate IQRs and save results
    if learning_curves:
        try:
            iqr_time_values = iqr_across_time(learning_curves)
            avg_iqr_time = np.mean(iqr_time_values)
            avg_iqr_run = iqr_across_runs(learning_curves)  # mean normalise each
            save_path = os.path.join(save_directory, f"{nodes_evaluated}_IQR_results.txt")
            with open(save_path, "a") as f:
                f.write(f"{group_name}: {avg_iqr_time:.10f}, {avg_iqr_run:.10f}\n")

            group_save_path = os.path.join(save_directory, f"{group_name}_IQR_results.txt")
            with open(group_save_path, "w") as f:
                f.write(f"Group: {group_name}\n")
                f.write(f"Average IQR Across Time: {avg_iqr_time:.10f}\n")
                f.write(f"Average IQR Across last 20% of Runs: {avg_iqr_run:.10f}\n\n")
                f.write("Individual Run IQR Values Across Time:\n")
                for run_name, iqr_value in zip(run_names, iqr_time_values):
                    f.write(f"{run_name}: {iqr_value[0]:.10f}\n")
            print(f"Results saved for group {group_name} at {group_save_path}.")
        except Exception as e:
            print(f"Failed to calculate IQR for group {group_name}: {e}")
    else:
        print(f"No learning curves available for group {group_name}.")

print("Processing complete for all groups.")
