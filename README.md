# Beyond Rewards in RL for Cyber Defence

**ICML 2026 code submission — anonymous version**

---
This code accompanies the ICML 2026 paper *“Beyond Rewards in RL for Cyber Defence.”*
The work explores how different reward‐function designs affect **autonomous cyber‑defence (ACD)** agents trained in 
the **[Yawning Titan](https://github.com/dstl/YAWNING-TITAN)** cyber‑gym and the **[MiniCAGE](https://github.com/alan-turing-institute/CybORG_plus_plus/tree/main/mini_CAGE)** environment from the 
CybORG++ toolkit.

This repository contains an alternate configuration of Yawning Titan, the miniCAGE environment, the codebase that 
accompanies "Measuring 
the Reliability of Reinforcement Learning Algorithms" by Chan et al. 
(2020) plus the training & evaluation 
scripts used in this paper.

---

## 🛠️ Re‑creating the Conda environment

```bash
# 1. Unpack
$ unzip icml2026_code.zip

# To train in the Yawning Titan env, you will need to set up a specific conda environment.

# For Mac users:
# 2. Create the Conda environment
$ conda env create -f yt_macos_environment.yml # or mamba if you prefer
# 3. Activate the environment
$ conda activate yt_rewards_macos
# Now you can run the Training and evaluation scripts

# For Linux users:
# 2. Create the Conda environment
$ conda env create -f yt_intel_linux_environment.yml 
# 3. Activate the environment
$ conda activate yt_rewards_intel_linux
# 4. Install the repo and Yawning Titan dependencies
$ pip install --no-deps -e .
$ pip install --no-deps -e ./YAWNING-TITAN
# Now you can run the Training and evaluation scripts in the Reward_Function_Experimentation directory


# To train in the MiniCAGE env, you will need to set up a separate conda environment.
# For Mac users:
# 2. Create the Conda environment
$ conda env create -f miniCAGE_macos_environment.yml # or mamba if you prefer
# 3. Activate the environment
$ conda activate miniCAGE_rewards_macos
# 4. Due to a dependency conflict, you will need to install a newer version of SB3 after creating the environment:
$ pip install stable-baselines3==2.3.2
# Now you can run the Training and evaluation scripts in the CybORG_plus_plus directory

# For Linux users:
# 2. Create the Conda environment
$ conda env create -f miniCAGE_intel_linux_environment.yml
# 3. Activate the environment
$ conda activate miniCAGE_rewards_linux
# 4. Due to a dependency conflict, you will need to install a newer version of SB3 after creating the environment:
$ pip install stable-baselines3==2.3.2
# Now you can run the Training and evaluation scripts in the CybORG_plus_plus directory
```

## 📌 Getting started in Yawning Titan
1. Once you've unzipped the code and configured your conda environment, now you can try training a blue agent. <br>
   Navigate to the Reward_Function_Experimentation/Training/parallel_training.py file and find the following lines of code. You can edit these to alter the 
   experiments you'd like to run:

   ```bash
    NET_SHAPE = ['linear']
    NODE_COMBINATIONS = [5]
    REWARD_FUNCTIONS = ['scaffolded', 'complex_dense', 'simple_pos_neg', 'simple_positive', 'simple_negative']
    N_STEPS = 100
    NODE_VULNERABILITY = 1
    RED_AGENT_SKILL = 1
    ORDER = ["Red_Blue", "Blue_Red", "Balanced"] 
    EVAL_TYPE = "initial_eval"
    ACTION_SPACE_SET = ["simple_action_space", "decoy_action_space"]
    ALGO = ["PPO"] 
    wandb_project_name = "YT_Reward_Engineering"
    output_location= "Reward_Engineering/Training/Models"
    NO_RUNS = 10

   ```

2. Run the training file from the command line:

   ```bash
   # cross‑platform, minimal
   python Training/parallel_training.py

   # Or create a nohup folder and run using nohup
   nohup python -u Training/parallel_training.py > nohup/training_1.log 2>&1 &

   ```

3. Evaluate the trained models using the scripts in the Evaluation folder.<br> To evaluate using the 
   ScoreGT_Evaluation.py script, first edit the following lines at the top of the file:

   ```bash
    NODE_VULNERABILITY = 1
    RED_AGENT_SKILL = 1
    N_STEPS = 100
    N_EPISODES = 1000
    NO_AGENTS = 10 
    MODEL_LOCATION = "Reward_Engineering/Models"

   ```
   After, edit the following lines in the final function "parallel_evaluation" to reflect the models you would like 
   to evaluate:
   ```bash
       node_values = [10] 
       reward_types = [
           'Positive Rewards',
           'Negative Rewards',
           'Scaffolded Rewards',
           'Complex Dense Rewards',
           'Simple Positive and Negative Rewards'
       ]
       action_space_set = ['simple_action_space', 'decoy_action_space']
       order = ['Red_Blue', 'Blue_Red', 'Balanced']
      ```

    Then run the script either from the command line or your python console. <br>
    <br>
    To evaluate using the Reliability_Evaluation.py script, first edit the following lines at the top of the file. It 
requires access to the wandb runs associated with each run.
    
    ```bash
        nodes_evaluated = '5_Nodes'  
        entity = "Your_wandb_entity"  
        project = "YT_Reward_Engineering" 
        save_directory = f"Reward_Function_Experimentation/Models/eval_log_{nodes_evaluated}_runs"
    ```
    Then run the script either from the command line or your python console. <br>
    
    The outputs of running these evaluation files can be seen in the **Models/eval** folder created, unless 
   otherwise 
    edited to save to a different location. <br>

## 📌 Getting started in MiniCAGE
1. Once you've unzipped the code and configured your conda environment, now you can try training a blue agent in MiniCAGE. <br>
   Navigate to the icml2026_codebase/CybORG_plus_plus/Training/SB3_training.py file and find the following lines of 
   code. You can edit these to alter the 
   experiments you'd like to run:

   ```bash
    NUM_RUNS: int = 25
    TOTAL_TIMESTEPS: int = 2_500_000

    USE_WANDB: bool = True           # flip to False to disable W&B logging
    USE_TENSORBOARD: bool = True     # if True, each run gets its own TB dir

    WANDB_PROJECT: str = "" # Add your wandb project name here
    WANDB_ENTITY: str | None = "" # Add your wandb entity/team name here
    GROUP_NAME: str = f"SB3_dqn_simple_default_Metric_rew_{TOTAL_TIMESTEPS}" # Add a group name for your runs here

    # These hyper-parameters are taken from the cardiff solution
    LEARNING_RATE: float = 0.002
    GAMMA: float = 0.99
    CLIP_RANGE: float = 0.2
    N_EPOCHS: int = 6
   ```
2. Adjust the model used depending on whether you'd like to use PPO or DQN.
    In the 'train_worker' function, there are two options for the model to use in training. Comment/uncommment the PPO or DQN model blocks depending on your preference.

3. Adjust the lines of code below in the same function depending on the algorithm used:
    ```bash
          run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=GROUP_NAME,
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,  
            config=dict(
                algorithm="DQN", # change to "PPO" if using PPO
                total_timesteps=TOTAL_TIMESTEPS,
                env="MiniCageBlue",
                seed=idx,
                # learning_rate=LEARNING_RATE,   # uncomment if using PPO
                # gamma=GAMMA,   # uncomment if using PPO
                # clip_range=CLIP_RANGE,   # uncomment if using PPO
                # n_epochs=N_EPOCHS,   # uncomment if using PPO
                exploration_final_eps=0.005,
                buffer_size=200_000,
            ),
        )
    ```
4. Run the training file from the command line. You will have to log on to wandb if you have enabled it.
    ```bash
    # cross‑platform, minimal
    python Training/SB3_training.py
    
    # Or create a nohup folder and run using nohup
    nohup python -u Training/SB3_training.py > nohup/training_1.log 2>&1 &
    
    ```

## 🗄️ Repository layout

```
icml2026_code/
├── icml2026_codebase/
│   ├── __init__.py
│   ├── CybORG_plus_plus/
│   │   ├── mini_CAGE/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── agents.py
│   │   │   ├── minimal.py
│   │   │   └── rl_red_agent.py
│   │   ├── rl-reliability-metrics/
│   │   └── Training/
│   │       ├── SB3_training.py
│   │       └── single_agent_gym_wrapper.py
│   │
│   └── Reward_Function_Experimentation/
│       ├── Evaluation/
│       │   ├── Basic_agent_runthrough.py
│       │   ├── Reliability_Evaluation.py
│       │   └── ScoreGT_Evaluation.py
│       ├── Minimal_network_gamemode.json
│       ├── Network/
│       │   └── N_node_generator.py
│       ├── Training/
│       │   ├── experiment_runner.py
│       │   ├── Minimal_network_gamemode.json
│       │   └── parallel_training.py
│       ├── Reward_Engineering/
│       ├── utils.py
│       └── yawning_titan_run_wandb.py
├── miniCAGE_intel_linux_environment.yml
├── miniCAGE_macos_environment.yml
├── pyproject.toml
├── README.md
├── rl-reliability-metrics/
└── YAWNING-TITAN/
├── yt_intel_linux_environment.yml
└── yt_macos_environment.yml                
```

---


## 👥 Acknowledgements

* **Yawning Titan** cyber-gym – Dstl, MIT Licence.
  *Original repo:* [YAWNING-TITAN](https://github.com/dstl/YAWNING-TITAN)<br>
  Our submission includes a lightly patched copy under the same MIT terms.
* **rl‑reliability‑metrics** – MIT Licence.
  Included verbatim (no code changes).
  *Original repo:* [rl-reliability-metrics](https://github.com/google-research/rl-reliability-metrics)
* **Stable-Baselines3** RL library – MIT Licence. *User guide:* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
* **CybORG++** Toolkit containing the MiniCAGE cyber gym. *Original repo:* [CybORG_plus_plus](https://github.com/alan-turing-institute/CybORG_plus_plus)
* This README file was made with help from ChatGPT-5.

Each subdirectory carries its original `LICENSE` file where required.

