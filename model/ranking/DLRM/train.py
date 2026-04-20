import logging
import time
import torch
import torch.nn as nn

from notes.ranking.DLRM.model import DLRM
from notes.utils.dataloader import create_dataset
from config import DLRMConfig
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
    interaction_op = config['interaction_op']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('criteo', sample_num=sample_size, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init dlrm model
    torch.manual_seed(1337)
    model = DLRM(field_dims, embed_dim, mlp_dims, dropout, interaction_op).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train dlrm model
    trainer = Trainer(model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"DLRM test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start DLRM Train')
    dlrm_config = DLRMConfig.all_config
    train(dlrm_config)
