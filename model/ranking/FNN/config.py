data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "batch_size": 4096
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 600,
    "trials": 20,
    "device": "cpu",
    "k_dims": 10,
    "mlp_dims": [16, 16, 16],
    "dropout": 0.2
}


class FNNModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
