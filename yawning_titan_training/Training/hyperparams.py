POSITIVE_RED_BLUE_HYPERPARAMETERS = {
    "learning_rate": 1e-3,
    "n_hidden_layers": 4,
    "hidden_layer_size": 128,
    "gae_lambda": 0.97,
    "clip_range": 0.16,
    "gamma": 0.92,
    "vf_coef": 0.49,
    "n_epochs": 6,
    "batch_size": 64
}

POSITIVE_ABLATION_HYPERPARAMS = {
    "learning_rate": lambda p: 5e-4 * p,
    "n_hidden_layers": 4,
    "hidden_layer_size": 64,
    "gae_lambda": 0.98,
    "clip_range": 0.16,
    "gamma": 0.89,
    "vf_coef": 0.49,
    "n_epochs": 6,
    "batch_size": 64,
    "n_steps": 4096
}

NEGATIVE_RED_BLUE_HYPERPARAMETERS = {
    "learning_rate": 4e-3,
    "n_hidden_layers": 4,
    "hidden_layer_size": 128,
    "gae_lambda": 0.98,
    "clip_range": 0.22,
    "gamma": 0.92,
    "vf_coef": 0.22,
    "n_epochs": 6,
    "batch_size": 64
}

POSNEG_RED_BLUE_HYPERPARAMETERS = {
    "learning_rate": 3e-4,
    "n_hidden_layers": 2,
    "hidden_layer_size": 128,
    "gae_lambda": 0.91,
    "clip_range": 0.26,
    "gamma": 0.91,
    "vf_coef": 0.79,
    "n_epochs": 6,
    "batch_size": 64
}

SCAFFOLD_RED_BLUE_HYPERPARAMETERS = {
    "learning_rate": 2e-3,
    "n_hidden_layers": 4,
    "hidden_layer_size": 64,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "gamma": 0.97,
    "vf_coef": 0.87,
    "n_epochs": 6,
    "batch_size": 64
}

COMPLEX_RED_BLUE_HYPERPARAMETERS = {
    "learning_rate": 3e-3,
    "n_hidden_layers": 3,
    "hidden_layer_size": 128,
    "gae_lambda": 0.86,
    "clip_range": 0.28,
    "gamma": 0.90,
    "vf_coef": 0.22,
    "n_epochs": 6,
    "batch_size": 64
}

REWARD_TO_HYPERPARAMS = {
    # 'simple_positive': POSITIVE_RED_BLUE_HYPERPARAMETERS,
    'simple_positive': POSITIVE_RED_BLUE_HYPERPARAMETERS,
    'simple_positive_ablation_4': POSITIVE_ABLATION_HYPERPARAMS,
    'simple_positive_ablation_5': POSITIVE_ABLATION_HYPERPARAMS,
    'simple_negative': NEGATIVE_RED_BLUE_HYPERPARAMETERS,

    'simple_pos_neg':  POSNEG_RED_BLUE_HYPERPARAMETERS,
    'dense_negative':       SCAFFOLD_RED_BLUE_HYPERPARAMETERS,
    # 'complex_dense_negative':  COMPLEX_RED_BLUE_HYPERPARAMETERS
    'complex_dense_negative':  COMPLEX_RED_BLUE_HYPERPARAMETERS,
    'missing_key':    POSITIVE_RED_BLUE_HYPERPARAMETERS
}