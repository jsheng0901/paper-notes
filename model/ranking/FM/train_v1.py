import logging
import time

from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
import numpy as np
from model_v1 import *
from notes.utils.dataloader import kaggle_fm_loader
from config import FMModelConfig

logging.basicConfig(level=logging.INFO)


def train(config):
    # load all parameters
    path = config['path']
    device = config['device']
    k_dims = config['k_dims']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    eval_interval = config['eval_interval']

    # loading the data
    t1 = time.time()
    x_train, x_val, y_train, y_val = kaggle_fm_loader(path)
    # transfer all data to tensor with float32 since later layer parameters will be float32 and put on device
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32).to(device)
    x_val = torch.tensor(np.array(x_val), dtype=torch.float32).to(device)
    # target need be same float32 type for loss function calculate loss
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
    y_val = torch.tensor(np.array(y_val), dtype=torch.float32).to(device)
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init model
    n_features = x_train.shape[1]
    torch.manual_seed(1337)
    model = FactorizationMachineModel(n_features, k_dims)
    # create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # train and eval
    for epoch in range(num_epochs):
        t3 = time.time()
        # start train mode
        model.train()

        # run model with whole dataset, no batch here, output -> [n_samples, 1]
        train_output = model(x_train)
        # calculate loss, target need same size as output -> [n_samples, 1], use view change size
        # here need binary cross entropy with logits loss, since output is raw without sigmoid and 1 class output
        train_loss = F.binary_cross_entropy_with_logits(train_output, y_train.view(-1, 1))
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_output = model(x_val)
                val_loss = F.binary_cross_entropy_with_logits(val_output, y_val.view(-1, 1))
                train_predicts = torch.sigmoid(train_output)
                val_predicts = torch.sigmoid(val_output)
                train_auc = roc_auc_score(y_train, train_predicts)
                val_auc = roc_auc_score(y_val, val_predicts)

            t4 = time.time()
            dt = t4 - t3
            logging.info(f"step {epoch}, dt: {dt * 1000:.2f}ms, train loss: {train_loss.item():.4f},"
                         f"train auc: {train_auc:.2f}, val loss： {val_loss.item():.4f}, val auc: {val_auc:.2f}")

    return


if __name__ == '__main__':
    logging.info('Start Factorization Machine Model Train')
    fm_config = FMModelConfig.all_config
    train(fm_config)
