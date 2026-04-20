import logging
import time
import torch
import torch.nn as nn

from notes.ranking.FM.model_v2 import FactorizationMachine
from notes.utils.dataloader import create_dataset
from config import FMModelConfig
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

    # loading the data
    t1 = time.time()
    dataset = create_dataset('criteo', sample_num=sample_size, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init model
    torch.manual_seed(1337)
    model = FactorizationMachine(field_dims, k_dims).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train the fm model
    trainer = Trainer(model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Factorization Machine Model Train')
    fm_config = FMModelConfig.all_config
    train(fm_config)
