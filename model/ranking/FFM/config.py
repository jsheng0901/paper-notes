data_config = {
    "path": "../../data/",
    "sample_size": 100000,
    "batch_size": 4096
}

model_config = {
    "learning_rate": 1e-4,
    "regularization": 1e-6,
    "num_epochs": 100,
    "trials": 10,
    "device": "cpu",
    "k_dims": 10
}


class FFMModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
