from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
import gc


# logistic regression 模型
def lr_model(data, categorical_fea, numerical_fea):
    # 连续特征归一化
    scaler = MinMaxScaler()
    for col in numerical_fea:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    # 离散特征one-hot编码
    for col in categorical_fea:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 把训练集和测试集分开
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    # 建立模型
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # calculate log_loss = −(y * log(p) + (1−y) * log(1−p))
    train_log_loss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    val_log_loss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print(f"train log loss: {train_log_loss}")
    print(f"validation log loss: {val_log_loss}")

    # 模型预测
    # predict_proba 返回[n, k]的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    y_pred = lr.predict_proba(test)[:, 1]
    # 看前10个， 预测为点击的概率
    print(f"first 10 ads predict click probability: {y_pred[:10]}")


# GBDT 模型
def gbdt_model(data, categorical_fea):
    # 连续特征不需要归一化处理，对于树类型model，scale的影响不大
    # 离散特征one-hot编码
    for col in categorical_fea:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 训练集和测试集分开
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    # 建模
    lgbm = lgb.LGBMClassifier(boosting_type='gbdt',
                              objective='binary',
                              subsample=0.8,
                              min_child_weight=0.5,
                              colsample_bytree=0.7,
                              num_leaves=100,
                              max_depth=12,
                              learning_rate=0.01,
                              n_estimators=10000
                              )
    lgbm.fit(x_train, y_train,
             eval_set=[(x_train, y_train), (x_val, y_val)],
             eval_names=['train', 'val'],
             eval_metric='binary_logloss',
             callbacks=[lgb.early_stopping(stopping_rounds=100)]
             )

    train_log_loss = log_loss(y_train, lgbm.predict_proba(x_train)[:, 1])
    val_log_loss = log_loss(y_val, lgbm.predict_proba(x_val)[:, 1])
    print(f"train log loss: {train_log_loss}")
    print(f"validation log loss: {val_log_loss}")

    # 模型预测，同lr model
    y_pred = lgbm.predict_proba(test)[:, 1]
    # 看前10个， 预测为点击的概率
    print(f"first 10 ads predict click probability: {y_pred[:10]}")


# LR + GBDT建模
# 下面就是把上面两个模型进行组合， GBDT负责对各个特征进行交叉和组合， 把原始特征向量转换为新的离散型特征向量， 然后在使用逻辑回归模型
def gbdt_lr_model(data, categorical_fea, numerical_fea):
    # 原始数据一开始通过GBDT处理训练，所以同上不需要对数值类型特征归一化处理
    # 离散特征one-hot编码
    for col in categorical_fea:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 训练集和测试集分开
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    lgbm = lgb.LGBMClassifier(objective='binary',
                              subsample=0.8,
                              min_child_weight=0.5,
                              colsample_bytree=0.7,
                              num_leaves=100,
                              max_depth=12,
                              learning_rate=0.01,
                              n_estimators=1000,
                              )

    lgbm.fit(x_train, y_train,
             eval_set=[(x_train, y_train), (x_val, y_val)],
             eval_names=['train', 'val'],
             eval_metric='binary_logloss',
             callbacks=[lgb.early_stopping(stopping_rounds=100)]
             )

    # 拿到所有构建的单一树
    model = lgbm.booster_
    # 提取基于树的特征组合情况
    # ex: [1599, 147]，每个样本落在每棵树的叶子节点的index，比如12，意思是当前这棵树的所有叶子节点中index为12的叶子节点
    gbdt_feats_train = model.predict(train, pred_leaf=True)
    gbdt_feats_test = model.predict(test, pred_leaf=True)
    # 给每个树的叶子结点对应的特征命名
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    # 把这些所有的类别特征组合成一个dataframe
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    # 构建新的训练数据集，把之前train用的所有特征和新提取出来的特征组合合并起来构成一个新的训练集
    train = pd.concat([train, df_train_gbdt_feats], axis=1)
    test = pd.concat([test, df_test_gbdt_feats], axis=1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    # 因为要重新应用lr model，需要对连续数据处理一下
    # 连续特征归一化
    scaler = MinMaxScaler()
    for col in numerical_fea:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    # 对基于树的特征组合进行离散特征one-hot编码，
    # ex: 12表示当前树的所有叶子节点第12个叶子节点，转化成[0, .., 1, ..]的当前特征所有维度的one-hot，
    # 比如此特征有20个叶子节点，则维度变成0-1的20维度的vector，第12个是1，其它都是0
    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 训练集和测试集分开
    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)

    # 建立模型
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    train_log_loss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    val_log_loss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print(f"train log loss: {train_log_loss}")
    print(f"validation log loss: {val_log_loss}")
    # 模型预测
    y_pred = lr.predict_proba(test)[:, 1]
    # 看前10个， 预测为点击的概率
    print(f"first 10 ads predict click probability: {y_pred[:10]}")
