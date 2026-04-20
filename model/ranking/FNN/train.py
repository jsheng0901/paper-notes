import logging
import time
import torch
import torch.nn as nn

from notes.ranking.FM.model_v2 import FactorizationMachine
from model import FactorizationSupportedNeuralNetworkModel
from notes.utils.dataloader import create_dataset
from config import FNNModelConfig
from notes.utils.trainer import Trainer

logging.basicConfig(level=logging.INFO)


def train(config):
    # load all parameters
    device = config['device']
    k_dims = config['k_dims']
    learning_rate = config['learning_rate']
    regularization = config['regularization']
    num_epochs = config['num_epochs']
    trials = config['trials']
    sample_size = config['sample_size']
    batch_size = config['batch_size']
    mlp_dims = config['mlp_dims']
    dropout = config['dropout']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('criteo', sample_num=sample_size, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init FM model
    torch.manual_seed(1337)
    fm_model = FactorizationMachine(field_dims, k_dims).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(fm_model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train the FM model
    trainer = Trainer(fm_model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"FM model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    # init FNN model
    fnn_model = FactorizationSupportedNeuralNetworkModel(field_dims, k_dims, mlp_dims, dropout).to(device)

    # assign the embedding output from FM model to FNN model
    # get all parameter state dict
    fm_state_dict = fm_model.state_dict()
    fnn_state_dict = fnn_model.state_dict()
    # assign the weight according to torch.nn.embedding layer defined name
    fnn_state_dict['linear_embedding.embedding.weight'] = fm_state_dict['linear_layer.linear.weight']
    fnn_state_dict['cross_embedding.embedding.weight'] = fm_state_dict['cross_layer.embedding_layer.embedding.weight']
    fnn_state_dict['mlp.mlp.0.bias'] = torch.zeros_like(fnn_state_dict['mlp.mlp.0.bias']).fill_(
        fm_state_dict['linear_layer.bias'].item())
    fnn_model.load_state_dict(fnn_state_dict)

    # start train the fnn model
    trainer = Trainer(fnn_model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"FNN model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Factorization Supported Neural Network Model Train')
    fnn_config = FNNModelConfig.all_config
    train(fnn_config)
