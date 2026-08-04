def evaluate_combination(n_nodes, reward_type, agent_order, action_space):
    from Evaluation_score import main
    import numpy

    # Define reward configurations
    reward_configs = {
        'Negative Rewards': {
            'function': 'simple_negative',
            'description': 'Negative rewards: -1 only when all Nodes are compromised',
        },
        'Positive Rewards': {
            'function': 'simple_positive',
            'description': 'Positive rewards: +1 only when no Nodes are compromised',
        },
        'Dense Negative Rewards': {
            'function': 'dense_negative',
            'description': 'Dense negative rewards: -1 per compromised node per timestep',
        },
        'Complex Dense Negative Rewards': {
            'function': 'complex_dense_negative',
            'description': 'Complex dense negative rewards use the standard reward function given in the YT codebase',
        },
        'Simple Positive and Negative Rewards': {
            'function': 'simple_pos_neg',
            'description': 'Simple positive and negative rewards: +1 for no nodes compromised, -1 for all nodes compromised',
        },
    }

    # Set the reward function and description for the current task
    REWARD_FUNCTION = reward_configs[reward_type]['function']
    REWARD_DESCRIPTION = reward_configs[reward_type]['description']

    # Pass the variables directly to main
    print(f"Evaluating: {n_nodes} nodes with reward type '{reward_type}' ({REWARD_FUNCTION}), agent order '{agent_order}', and action space '{action_space}'")
    main(n_nodes=n_nodes, reward_function=REWARD_FUNCTION, reward_description=REWARD_DESCRIPTION,
         reward_type=reward_type, order=agent_order, action_space=action_space)
