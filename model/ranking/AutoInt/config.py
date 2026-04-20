data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "batch_size": 4096
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 300,
    "trials": 10,
    "device": "cpu",
    "embed_dim": 10,
    "atten_embed_dim": 32,
    "mlp_dims": [64, 32],
    "dropouts": [0.2, 0.1],
    "num_layers": 2,
    "num_heads": 2,
    "has_residual": True
}


class AutomaticFeatureInteractionModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
