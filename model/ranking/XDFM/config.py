data_config = {
    "path": "../../data/",
    "sample_size": 10000,
    "batch_size": 64
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 300,
    "trials": 100,
    "device": "cpu",
    "embed_dim": 4,
    "mlp_dims": [32, 16],
    "dropout": 0.2,
    "cross_layer_sizes": [16, 16]
}


class ExtremeDeepFactorizationMachineModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
