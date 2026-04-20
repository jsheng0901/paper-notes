from notes.utils.dataloader import kaggle_loader
from model import *
import time


if __name__ == '__main__':
    t1 = time.time()
    # loading the data
    path = "../../data/"
    data, categorical_fea, numerical_fea = kaggle_loader(path)
    t2 = time.time()
    print(f"Loading data takes {(t2 - t1) * 1000}ms")

    # train the model
    # 训练和预测LR模型
    lr_model(data.copy(), categorical_fea, numerical_fea)
    t3 = time.time()
    print(f"Train and predict LR model takes {(t3 - t2) * 1000}ms")

    # 模型训练和预测GBDT模型
    gbdt_model(data.copy(), categorical_fea)
    t4 = time.time()
    print(f"Train and predict GBDT-LGBM model takes {(t4 - t3) * 1000}ms")

    # 训练和预测GBDT+LR模型
    gbdt_lr_model(data.copy(), categorical_fea, numerical_fea)
    t5 = time.time()
    print(f"Train and predict GBDT+LR model takes {(t5 - t4) * 1000}ms")
