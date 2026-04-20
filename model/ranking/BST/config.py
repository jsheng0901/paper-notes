data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "sequence_length": 40,
    "batch_size": 4096
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 600,
    "trials": 100,
    "device": "cpu",
    "embed_dim": 10,
    "mlp_dims": [200, 80],
    "num_heads": 2,
    "stacked_transformer_layers": 1,
    "attn_dropout": 0.1,
    "net_dropout": 0.1,
    "use_position_emb": True,
    "layer_norm": True,
    "use_residual": True,
    "seq_pooling_type": "mean"
}


class BehaviorSequenceTransformerModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
