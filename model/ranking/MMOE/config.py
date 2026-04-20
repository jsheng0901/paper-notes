data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "batch_size": 4096,
    "num_tasks": 2
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 300,
    "trials": 10,
    "device": "cpu",
    "embed_dim": 10,
    "expert_mlp_dims": [128, 64, 32],
    "gate_mlp_dims": [32, 16],
    "tower_mlp_dims": [64, 32],
    "num_experts": 3,
    "dropout": 0.2
}


class MultigateMixtureOfExpertsModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
