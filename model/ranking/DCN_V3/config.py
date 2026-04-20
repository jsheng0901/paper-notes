data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "batch_size": 4096
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 300,
    "trials": 2,
    "device": "cpu",
    "embed_dim": 10,
    "num_linear_cross_layers": 2,
    "num_exp_cross_layers": 3,
    "exp_net_dropout": 0.1,
    "linear_net_dropout": 0.1,
    "layer_norm": True,
    "batch_norm": True,     # this will accelerate training time with fewer steps
    "num_heads": 1
}


class DeepCrossNetworkV3ModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
