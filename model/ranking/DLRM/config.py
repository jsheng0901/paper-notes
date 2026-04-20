data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "batch_size": 4096
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 300,
    "trials": 5,
    "device": "cpu",
    "embed_dim": 10,
    "mlp_dims": [64, 32],
    "dropout": 0.2,
    "interaction_op": "cat"
}


class DLRMConfig:
    all_config = {
        **data_config,
        **model_config,
    }
