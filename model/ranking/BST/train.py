import logging
import time
import torch
import torch.nn as nn

from notes.ranking.BST.config import BehaviorSequenceTransformerModelConfig
from notes.ranking.BST.model import BehaviorSequenceTransformerModel
from notes.utils.dataloader import create_dataset
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
    sequence_length = config['sequence_length']
    batch_size = config['batch_size']
    mlp_dims = config['mlp_dims']
    num_heads = config['num_heads']
    stacked_transformer_layers = config['stacked_transformer_layers']
    attn_dropout = config['attn_dropout']
    net_dropout = config['net_dropout']
    use_position_emb = config['use_position_emb']
    layer_norm = config['layer_norm']
    use_residual = config['use_residual']
    seq_pooling_type = config['seq_pooling_type']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('amazon-books', sample_num=sample_size, sequence_length=sequence_length, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init BST model
    torch.manual_seed(1337)
    model = BehaviorSequenceTransformerModel(field_dims, sequence_length, mlp_dims, num_heads,
                                             stacked_transformer_layers, attn_dropout, embed_dim, net_dropout,
                                             layer_norm, use_residual, seq_pooling_type, use_position_emb).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train BST model
    trainer = Trainer(model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"Behavior Sequence Transformer model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Behavior Sequence Transformer Model Train')
    dim_config = BehaviorSequenceTransformerModelConfig.all_config
    train(dim_config)
