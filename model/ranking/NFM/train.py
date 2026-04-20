import logging
import time
import torch
import torch.nn as nn

from notes.ranking.NFM.model import NeuralFactorizationMachineModelV1, NeuralFactorizationMachineModelV2
from notes.utils.dataloader import create_dataset
from config import NeuralFactorizationMachineModelConfig
from notes.utils.trainer import Trainer

logging.basicConfig(level=logging.INFO)


def train(config):
    # load all parameters
    device = config['device']
    embed_dim = config['embed_dim']
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

    # init nfm v1 model
    torch.manual_seed(1337)
    model_v1 = NeuralFactorizationMachineModelV1(field_dims, mlp_dims, dropout, embed_dim).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model_v1.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train nfm v1 model
    trainer = Trainer(model_v1, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"Neural Factorization Machine V1 model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    # init nfm v2 model
    model_v2 = NeuralFactorizationMachineModelV2(field_dims, embed_dim, mlp_dims, dropout).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model_v2.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train nfm v2 model
    trainer = Trainer(model_v2, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"Neural Factorization Machine V2 model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Neural Factorization Machine Model Train')
    nfm_config = NeuralFactorizationMachineModelConfig.all_config
    train(nfm_config)
