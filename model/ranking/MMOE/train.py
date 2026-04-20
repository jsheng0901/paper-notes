import logging
import time
import torch
import torch.nn as nn

from notes.ranking.MMOE.model import MultigateMixtureOfExpertsModel
from notes.utils.dataloader import create_dataset
from config import MultigateMixtureOfExpertsModelConfig
from notes.utils.trainer import MultitaskTrainer

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
    expert_mlp_dims = config['expert_mlp_dims']
    gate_mlp_dims = config['gate_mlp_dims']
    tower_mlp_dims = config['tower_mlp_dims']
    num_experts = config['num_experts']
    dropout = config['dropout']
    num_tasks = config['num_tasks']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('adult', sample_num=sample_size, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init model
    torch.manual_seed(1337)
    model = MultigateMixtureOfExpertsModel(field_dims, embed_dim, num_experts, num_tasks, expert_mlp_dims,
                                           gate_mlp_dims, tower_mlp_dims, dropout).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train MMOE model
    trainer = MultitaskTrainer(model, optimizer, criterion, batch_size, num_tasks)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    for i in range(trainer.num_tasks):
        logging.info(f"Multigate Mixture Of Experts model task{i + 1}: "
                     f"test loss: {test_loss[i]:.5f} | test_auc: {test_auc[i]:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Multigate Mixture Of Experts Model Train')
    mmoe_config = MultigateMixtureOfExpertsModelConfig.all_config
    train(mmoe_config)
