import logging
import time
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm

from notes.ranking.DCN_V3.model import DeepCrossNetworkV3Model
from notes.utils.dataloader import create_dataset, BatchLoader
from config import DeepCrossNetworkV3ModelConfig
from notes.utils.trainer import Trainer, EarlyStopper

logging.basicConfig(level=logging.INFO)


class DeepCrossNetworkV3Trainer(Trainer):

    def __init__(self, model, optimizer, criterion, batch_size=None, num_tasks=1):
        super().__init__(model, optimizer, criterion, batch_size)
        self.num_tasks = num_tasks

    def get_loss(self, output_dict, y_true):
        """
        Get total loss for three outputs
        """
        # get three outputs logit
        y_pred = output_dict["y_pred"]
        y_exp = output_dict["y_exp"]
        y_linear = output_dict["y_linear"]
        # calculate each tower output loss
        loss = self.criterion(y_pred, y_true)
        loss_exp = self.criterion(y_exp, y_true)
        loss_linear = self.criterion(y_linear, y_true)
        # calculate each loss weight
        weight_exp = loss_exp - loss
        weight_linear = loss_linear - loss
        # when exp tower has better predict than linear and total loss, exp tower loss weight will be 0
        # which means model focus more on linear tower and total loss optimization
        weight_exp = torch.where(weight_exp > 0, weight_exp, torch.zeros(1))
        weight_linear = torch.where(weight_linear > 0, weight_linear, torch.zeros(1))
        # calculate total loss
        loss = loss + loss_exp * weight_exp + loss_linear * weight_linear

        return loss

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
        for step in tqdm(range(epoch)):
            t1 = time.time()
            # train mode
            self.model.train()
            # accumulate loss by batch
            batch_train_loss = 0
            for b_x, b_y in train_loader:
                # get three outputs y_pred, y_exp, y_linear logit
                output_dict = self.model(b_x)
                # get total train loss
                train_loss = self.get_loss(output_dict, b_y)
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

    def test(self, test_x, test_y):
        # eval mode
        self.model.eval()
        # calculate pred value and loss
        with torch.no_grad():
            # get three test outputs y_pred, y_exp, y_linear logit
            output_dict = self.model(test_x)
            # get test loss
            test_loss = self.get_loss(output_dict, test_y).detach().numpy()[0]

            # calculate different task metric
        if self.task == 'classification':
            test_metric = metrics.roc_auc_score(test_y.cpu(), output_dict["y_pred"].cpu())
        if self.task == 'regression':
            test_metric = -test_loss

        return test_loss, test_metric


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
    num_linear_cross_layers = config['num_linear_cross_layers']
    num_exp_cross_layers = config['num_exp_cross_layers']
    exp_net_dropout = config['exp_net_dropout']
    linear_net_dropout = config['linear_net_dropout']
    layer_norm = config['layer_norm']
    batch_norm = config['batch_norm']
    num_heads = config['num_heads']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('criteo', sample_num=sample_size, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init model
    torch.manual_seed(1337)
    model = DeepCrossNetworkV3Model(field_dims, embed_dim, num_linear_cross_layers, num_exp_cross_layers,
                                    exp_net_dropout, linear_net_dropout, layer_norm, batch_norm, num_heads).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train dcn_v3 model
    trainer = DeepCrossNetworkV3Trainer(model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"Deep Cross Network V3 model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Deep Cross Network V3 Model Train')
    dcn_v3_config = DeepCrossNetworkV3ModelConfig.all_config
    train(dcn_v3_config)
