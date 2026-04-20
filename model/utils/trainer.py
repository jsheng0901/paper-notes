import logging
import time
from copy import deepcopy

import torch
from sklearn import metrics
from tqdm import tqdm

from notes.utils.dataloader import BatchLoader


class EarlyStopper:
    """
    Early stopper object.
    If metric is improved or metric not continues to improve smaller than number of trials, then keep training.
    Otherwise, stop training.
    """

    def __init__(self, model, num_trials=50, num_tasks=1):
        # init for all type of jobs including multitask job
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_metric = num_tasks * [-1e9]
        self.best_state = num_tasks * [deepcopy(model.state_dict())]
        self.model = model
        self.num_tasks = num_tasks
        self.continuable = None

    def is_continuable(self, metric):
        # check if it's single task, then metric is just float not list, convert into list for later loop
        if isinstance(metric, float):
            metric = [metric]
        # each time check init with True first, if one task meet below case then will change to False
        self.continuable = False
        # maximize metric
        # if metric is keep increase
        for i in range(self.num_tasks):
            if metric[i] > self.best_metric[i]:
                # update best metric
                self.best_metric[i] = metric[i]
                # init trail counter
                self.trial_counter = 0
                # record model state
                self.best_state[i] = deepcopy(self.model.state_dict())
                self.continuable = True
            # if metric not improve times smaller than trials
            elif self.trial_counter + 1 < self.num_trials:
                # update number of trial counter
                self.trial_counter += 1
                self.continuable = True

        # if all tasks not meet above continuable case then stop training
        return self.continuable


class Trainer:

    def __init__(self, model, optimizer, criterion, batch_size=None, task='classification'):
        assert task in ['classification', 'regression']
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.task = task

        self.early_stopper = None

    def train(self, train_x, train_y, epoch=100, trials=None, valid_x=None, valid_y=None):
        # if batch loader
        if self.batch_size:
            train_loader = BatchLoader(train_x, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_x, train_y]]

        if trials:
            self.early_stopper = EarlyStopper(self.model, trials)

        train_loss_list = []
        valid_loss_list = []

        for step in tqdm(range(epoch)):
            t1 = time.time()
            # train mode
            self.model.train()
            # accumulate loss by batch
            batch_train_loss = 0
            for b_x, b_y in train_loader:
                pred_y = self.model(b_x)
                train_loss = self.criterion(pred_y, b_y)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                # here loss already calculate avg in batch, so we need time batch size back to calculate total loss
                batch_train_loss += train_loss.detach() * len(b_x)

            # record each epoch avg loss
            train_loss_list.append(batch_train_loss / len(train_x))

            # valid mode, check early stopper or not
            if trials:
                # valid loss and metric
                valid_loss, valid_metric = self.test(valid_x, valid_y)
                # valid_loss_list.append(valid_loss)
                # train loss and metric
                train_loss, train_metric = self.test(train_x, train_y)
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

        # print('train_loss: {:.5f} | train_metric: {:.5f}'.format(*self.test(train_X, train_y)))

        # if trials:
        #     print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(*self.test(valid_X, valid_y)))

    def test(self, test_x, test_y):
        # eval mode
        self.model.eval()
        # calculate pred value and loss
        with torch.no_grad():
            pred_y = self.model(test_x)
            test_loss = self.criterion(pred_y, test_y).detach()

        # calculate different task metric
        if self.task == 'classification':
            test_metric = metrics.roc_auc_score(test_y.cpu(), pred_y.cpu())
        if self.task == 'regression':
            test_metric = -test_loss

        return test_loss, test_metric


class MultitaskTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, batch_size=None, num_tasks=1):
        super().__init__(model, optimizer, criterion, batch_size)
        self.num_tasks = num_tasks

    def train(self, train_x, train_y, epoch=100, trials=None, valid_x=None, valid_y=None):
        # if batch loader
        if self.batch_size:
            train_loader = BatchLoader(train_x, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_x, train_y]]

        if trials:
            self.early_stopper = EarlyStopper(self.model, trials, self.num_tasks)

        train_loss_list = []
        for step in tqdm(range(epoch)):
            t1 = time.time()
            # train part
            self.model.train()
            # accumulate loss by batch
            batch_train_loss = 0
            for b_x, b_y in train_loader:
                pred_y = self.model(b_x)
                # multitask loss simple sum together as one loss
                train_loss = [self.criterion(pred_y[:, i], b_y[:, i]) for i in range(self.num_tasks)]
                total_train_loss = sum(train_loss)
                self.optimizer.zero_grad()
                total_train_loss.backward()
                self.optimizer.step()
                # here loss already calculate avg in batch, so we need time batch size back to calculate total loss
                batch_train_loss += total_train_loss.detach() * len(b_x)

            # record each epoch avg loss
            train_loss_list.append(batch_train_loss / len(train_x))

            # valid mode, check early stopper or not
            if trials:
                # valid loss and metric
                valid_loss, valid_metric = self.test(valid_x, valid_y)
                # valid_loss_list.append(valid_loss)
                # train loss and metric
                train_loss, train_metric = self.test(train_x, train_y)
                t2 = time.time()
                dt = t2 - t1
                logging.info(f"step {step}, dt: {dt * 1000:.2f}ms, train total loss: {total_train_loss:.4f}")
                for i in range(self.num_tasks):
                    logging.info(
                        f"task{i + 1}: train loss: {train_loss[i]}, train metric: {train_metric[i]:.3f},"
                        f"val loss： {valid_loss[i]:.4f}, val auc: {valid_metric[i]:.3f}"
                    )
                if self.early_stopper.is_continuable(valid_metric) is False:
                    break

    def test(self, test_x, test_y):
        # eval mode
        self.model.eval()
        # calculate pred value and loss
        with torch.no_grad():
            pred_y = self.model(test_x)
            # for test record multitask loss separate
            test_loss = [self.criterion(pred_y[:, i], test_y[:, i]).detach() for i in range(self.num_tasks)]

        # calculate different task metric
        if self.task == 'classification':
            test_metric = [metrics.roc_auc_score(test_y[:, i].cpu(), pred_y[:, i].cpu()) for i in range(self.num_tasks)]
        if self.task == 'regression':
            test_metric = [-test_loss[i] for i in range(self.num_tasks)]

        return test_loss, test_metric
