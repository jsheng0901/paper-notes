import logging
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from notes.ranking.DIEN.config import DeepInterestEvolutionNetworkModelConfig
from notes.ranking.DIEN.model import DeepInterestEvolutionNetworkModel
from notes.utils.dataloader import create_dataset
from notes.utils.trainer import Trainer, EarlyStopper, BatchLoader

logging.basicConfig(level=logging.INFO)


class DIENTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, batch_size=None):
        super().__init__(model, optimizer, criterion, batch_size)

    def train(self, train_x_list, train_y, epoch=100, trials=None, valid_x=None, valid_y=None):
        if self.batch_size:
            train_loader = BatchLoader(train_x_list[1], train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_x_list[1], train_y]]

        if trials:
            self.early_stopper = EarlyStopper(self.model, trials)

        train_loss_list = []

        for step in tqdm(range(epoch)):
            t1 = time.time()
            # train part
            self.model.train()
            # accumulate loss by batch
            batch_train_loss = 0
            for b_x, b_y in train_loader:
                self.optimizer.zero_grad()
                # sequence length is half, since we stack train and negative label as input
                seq_len = b_x.shape[1] // 2
                # first input is train x, second input is negative label
                # get predict y label -> [batch_size, 1], auxiliary score from inner product -> [2, all_timestep]
                pred_y, auxiliary_y = self.model(b_x[:, :seq_len + 1], b_x[:, -seq_len + 1:])
                # one is positive label, zero is negative label, create true label, only contains 0, 1
                auxiliary_true = torch.cat([torch.ones_like(auxiliary_y[0]), torch.zeros_like(auxiliary_y[1])],
                                           dim=0).view(2, -1)
                # auxiliary_y -> [2, all_timestep], auxiliary_true -> [2, all_timestep] then apply log loss (BCELoss)
                auxiliary_loss = self.criterion(auxiliary_y, auxiliary_true)
                auxiliary_loss.backward(retain_graph=True)

                train_loss = self.criterion(pred_y, b_y)
                train_loss.backward()
                self.optimizer.step()
                # here loss already calculate avg in batch, so we need time batch size back to calculate total loss
                batch_train_loss += train_loss.detach() * len(b_x)

            # record each epoch avg loss
            train_loss_list.append(batch_train_loss / len(train_x_list[1]))

            # valid part
            if trials:
                valid_loss, valid_metric = self.test(valid_x, valid_y)
                # valid_loss_list.append(valid_loss)
                # train loss and metric, here only input original train_x since no need negative sample when do the test
                train_loss, train_metric = self.test(train_x_list[0], train_y)
                t2 = time.time()
                dt = t2 - t1
                logging.info(f"step {step}, dt: {dt * 1000:.2f}ms, train loss: {train_loss:.4f},"
                             f"train metric: {train_metric:.3f}, val loss： {valid_loss:.4f},"
                             f"val auc: {valid_metric:.3f}")
                if self.early_stopper.is_continuable(valid_metric) is False:
                    break

        # if trials:
        #     self.model.load_state_dict(early_stopper.best_state)
        #     plt.plot(valid_loss_list, label='valid_loss')
        #
        # plt.plot(train_loss_list, label='train_loss')
        # plt.legend()
        # plt.show()
        #
        # print('train_loss: {:.5f} | train_metric: {:.5f}'.format(*self.test(train_X, train_y)))
        #
        # if trials:
        #     print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(*self.test(valid_X, valid_y)))


def auxiliary_sample(x, sample_set):
    # sample_set -> [1348, ] contains all behaviors feature index
    # x -> [batch_size, num_behaviors + 1(ad)] ex: [80000, 41]
    # pos_sample is next timestep behaviors, num_behaviors - 1 is next timestep sequence. Last field is ad, not behavior
    # pos_sample -> [batch_size, num_behaviors - 1] ex: [80000, 39]
    pos_sample = x[:, 1: -1]
    # neg_sample is same size as pos_sample, make a placeholder first -> [batch_size, num_behaviors - 1]
    neg_sample = torch.zeros_like(pos_sample)
    # loop through each sample and each field
    for i in tqdm(range(pos_sample.shape[0])):
        for j in range(pos_sample.shape[1]):
            # if pos_sample has value which is not padding index value 0
            if pos_sample[i, j] > 0:
                # random pick one behavior index as negative label
                idx = np.random.randint(len(sample_set))
                # if pick index is equal to positive index label, keep random pick one
                while sample_set[idx] == pos_sample[i, j]:
                    idx = np.random.randint(len(sample_set))
                # assign the same index negative placeholder label
                neg_sample[i, j] = sample_set[idx]
            # if is 0 value which is padding index, then negative label is same 0
            else:
                break
    # return negative label -> [batch_size, num_behaviors - 1] ex: [80000, 39]
    return neg_sample


def train(config):
    # load all parameters
    device = config['device']
    embed_dim = config['embed_dim']
    learning_rate = config['learning_rate']
    regularization = config['regularization']
    num_epochs = config['num_epochs']
    trials = config['trials']
    sample_size = config['sample_size']
    sequence_length = config['sequence_length']
    batch_size = config['batch_size']
    mlp_dims = config['mlp_dims']
    activation_dim = config['activation_dim']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('amazon-books', sample_num=sample_size, sequence_length=sequence_length, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    # create auxiliary negative sample -> [batch_size, num_behaviors - 1] ex: [80000, 39]
    neg_sample = auxiliary_sample(train_x, dataset.cate_set)
    # horizontal stack both train and neg label ->
    # [batch_size, num_behaviors + 1 + num_behaviors - 1] -> [batch_size, num_behaviors * 2]
    train_x_neg = torch.hstack([train_x, neg_sample])
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init din model
    torch.manual_seed(1337)
    model = DeepInterestEvolutionNetworkModel(field_dims, embed_dim, activation_dim, mlp_dims).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train dien model
    trainer = DIENTrainer(model, optimizer, criterion, batch_size)
    trainer.train([train_x, train_x_neg], train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"Deep Interest Evolution Network model test_loss: {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Deep Interest Evolution Network Model Train')
    dien_config = DeepInterestEvolutionNetworkModelConfig.all_config
    train(dien_config)
