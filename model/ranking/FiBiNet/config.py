data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "batch_size": 4096
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 100,
    "trials": 2,
    "device": "cpu",
    "embed_dim": 8,
    "mlp_dims": [32, 16],
    "dropout": 0.4,
    "reduction_ratio": 8,
    "bilinear_type": "each"
}


class FiBiNETModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
