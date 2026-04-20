import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OrdinalEncoder
from tqdm import tqdm
import numpy as np
import torch


def kaggle_loader(path):
    """
    数据读取与预处理，kaggle criteo 数据集
    """
    # 数据读取
    df_train = pd.read_csv(path + 'kaggle_train.csv')
    df_test = pd.read_csv(path + 'kaggle_test.csv')

    # 简单的数据预处理
    # 去掉id列， 把测试集和训练集合并， 填充缺失值
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)

    df_test['Label'] = -1

    # train 和 test 一起处理
    data = pd.concat([df_train, df_test])
    data.fillna(-1, inplace=True)

    # 下面把特征列分开处理
    numerical_fea = ['I' + str(i + 1) for i in range(13)]
    categorical_fea = ['C' + str(i + 1) for i in range(26)]

    return data, categorical_fea, numerical_fea


def kaggle_fm_loader(path):
    """
    为FM模型，准备数据读取与预处理，kaggle criteo 数据集
    """
    data = pd.read_csv(path + 'kaggle_train.csv')
    # dense 特征开头是I，sparse特征开头是C，Label是标签
    cols = data.columns.values
    dense_feats = [f for f in cols if f[0] == 'I']
    sparse_feats = [f for f in cols if f[0] == 'C']

    df = data.copy()
    # dense, apply log
    df_dense = df[dense_feats].fillna(0.0)
    for f in tqdm(dense_feats):
        df_dense[f] = df_dense[f].apply(lambda x: np.log(1 + x) if x > -1 else -1).astype('float32')

    # sparse, apply label encoder first
    df_sparse = df[sparse_feats].fillna('-1')
    for f in tqdm(sparse_feats):
        lbe = LabelEncoder()
        df_sparse[f] = lbe.fit_transform(df_sparse[f])

    # apply one hot encoder, change as int not boolean, since will be pass as tensor later
    df_sparse_arr = []
    for f in tqdm(sparse_feats):
        data_new = pd.get_dummies(df_sparse.loc[:, f].values, dtype='int32')
        data_new.columns = [f + "_{}".format(i) for i in range(data_new.shape[1])]
        df_sparse_arr.append(data_new)

    df_new = pd.concat([df_dense] + df_sparse_arr, axis=1)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(df_new, data['Label'], test_size=0.2, random_state=2020)

    return x_train, x_val, y_train, y_val


def create_dataset(dataset='criteo', read_part=True, sample_num=100000, task='classification', sequence_length=40,
                   device=torch.device('cpu')):
    """
    Create dataset according to file name
    """
    if dataset == 'criteo':
        return CriteoDataset('../../data/criteo-100k.txt', read_part=read_part, sample_num=sample_num).to(device)
    elif dataset == 'adult':
        return AdultDataset('../../data/adult_train.csv', '../../data/adult_test.csv',
                            read_part=read_part, sample_num=sample_num).to(device)
    elif dataset == 'amazon-books':
        return AmazonBooksDataset('../../data/amazon-books-100k.txt', read_part=read_part, sample_num=sample_num,
                                  sequence_length=sequence_length).to(device)
    else:
        raise Exception('No such dataset!')


class Dataset:
    """
    Dataset basic object
    """

    def __init__(self):
        self.device = torch.device('cpu')
        self.data = None

    def to(self, device):
        self.device = device
        return self

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        """
        train valid and test split function
        """
        # each features number of unique value
        # ex: fea1: [0, 1, 2, 3] -> 4 dim, fea2: [0, 1, 2] -> 3 dim, fea3: [0, 1, 2, 3, 4] -> 5 dim
        # then field_dims -> [4, 3, 5]
        # not last one is label column in data so will be exclude
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2021)

        valid_size = valid_size / (test_size + valid_size)
        valid, test = train_test_split(valid_test, train_size=valid_size, random_state=2021)

        train_x = torch.tensor(train[:, :-1], dtype=torch.long).to(self.device)
        valid_x = torch.tensor(valid[:, :-1], dtype=torch.long).to(self.device)
        test_x = torch.tensor(test[:, :-1], dtype=torch.long).to(self.device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(self.device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(self.device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(self.device)

        return field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class CriteoDataset(Dataset):
    """
    Criteo Dataset loader
    """

    def __init__(self, file, read_part=True, sample_num=100000):
        super().__init__()

        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']

        if read_part:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names, nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names)

        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        features = sparse_features + dense_features

        # 缺失值填充，数值特征填充为0，类别特征填充为 -1 的string
        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)

        # 连续型特征等间隔分箱
        est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        data_df[dense_features] = est.fit_transform(data_df[dense_features])

        # 离散型特征转换成连续数字，为了在与参数计算时使用索引的方式计算，而不是向量乘积
        # 这里用ordinal而不是label，为了保存原始的顺序意义并且作用在所有类别特征上，label一般作用在target上
        data_df[features] = OrdinalEncoder().fit_transform(data_df[features])

        self.data = data_df[features + ['label']].values


class AmazonBooksDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=100000, sequence_length=40):
        super().__init__()

        if read_part:
            data_df = pd.read_csv(file, sep=',', nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep=',')

        data_df['hist_item_list'] = data_df.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
        data_df['hist_cate_list'] = data_df.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

        # cate encoder
        cate_list = list(data_df['cateID'])
        data_df.apply(lambda x: cate_list.extend(x['hist_cate_list']), axis=1)
        cate_set = set(cate_list + ['0'])
        cate_encoder = LabelEncoder().fit(list(cate_set))
        self.cate_set = cate_encoder.transform(list(cate_set))

        # cate pad and transform
        hist_limit = sequence_length
        col = ['hist_cate_{}'.format(i) for i in range(hist_limit)]

        def deal(x):
            if len(x) > hist_limit:
                return pd.Series(x[-hist_limit:], index=col)
            else:
                pad = hist_limit - len(x)
                x = x + ['0' for _ in range(pad)]
                return pd.Series(x, index=col)

        cate_df = data_df['hist_cate_list'].apply(deal).join(data_df[['cateID']]).apply(cate_encoder.transform).join(
            data_df['label'])
        self.data = cate_df.values

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        field_dims = [self.data[:-1].max().astype(int) + 1]
        num_data = len(self.data)
        num_train = int(train_size * num_data)
        num_test = int(test_size * num_data)
        train = self.data[:num_train]
        valid = self.data[num_train: -num_test]
        test = self.data[-num_test:]

        device = self.device
        train_x = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_x = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_x = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class AdultDataset(Dataset):
    """
    Here we only support transfer all input features value into id,
    no difference between sparse or dense features.
    """
    def __init__(self, train_file, test_file, read_part=True, sample_num=100000):
        super().__init__()
        # features column name
        names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
                 'native_country', 'income_bracket']

        # read part or not
        if read_part:
            data_df_train = pd.read_csv(train_file, header=None, names=names, nrows=sample_num)
            data_df_test = pd.read_csv(test_file, header=None, names=names, nrows=sample_num)
        else:
            data_df_train = pd.read_csv(train_file, header=None, names=names)
            data_df_test = pd.read_csv(test_file, header=None, names=names)

        # concat together to feature preprocess same time
        data = pd.concat([data_df_train, data_df_test], axis=0)

        # create two labels to do mtl job
        # first label use income column, if over 50k then 1, otherwise is 0
        data['label_income'] = data['income_bracket'].map({' >50K.': 1, ' >50K': 1, ' <=50K.': 0, ' <=50K': 0})
        # second label use marital column, if no married then 1, otherwise is 0
        data['label_marital'] = data['marital_status'].apply(lambda x: 1 if x == ' Never-married' else 0)
        # drop original columns since already transfer as label
        data.drop(labels=['marital_status', 'income_bracket'], axis=1, inplace=True)

        # define dense and sparse features
        columns = data.columns.values.tolist()
        dense_features = ['fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        sparse_features = [col for col in columns if
                           col not in dense_features and col not in ['label_income', 'label_marital']]
        features = sparse_features + dense_features

        # 缺失值填充，数值特征填充为0，类别特征填充为 -1 的string
        data[sparse_features] = data[sparse_features].fillna('-1')
        data[dense_features] = data[dense_features].fillna(0)

        # 连续型特征等间隔分箱
        est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        data[dense_features] = est.fit_transform(data[dense_features])

        # 离散型特征转换成连续数字，为了在与参数计算时使用索引的方式计算，而不是向量乘积
        # 这里用ordinal而不是label，为了保存原始的顺序意义并且作用在所有类别特征上，label一般作用在target上
        data[features] = OrdinalEncoder().fit_transform(data[features])

        self.data = data[features + ['label_income'] + ["label_marital"]].values

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        """
        train valid and test split function, note here we have two labels
        """
        # each features number of unique value
        # ex: fea1: [0, 1, 2, 3] -> 4 dim, fea2: [0, 1, 2] -> 3 dim, fea3: [0, 1, 2, 3, 4] -> 5 dim
        # then field_dims -> [4, 3, 5]
        # not last two are label column in data so will be exclude
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-2]

        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2021)

        valid_size = valid_size / (test_size + valid_size)
        valid, test = train_test_split(valid_test, train_size=valid_size, random_state=2021)

        train_x = torch.tensor(train[:, :-2], dtype=torch.long).to(self.device)
        valid_x = torch.tensor(valid[:, :-2], dtype=torch.long).to(self.device)
        test_x = torch.tensor(test[:, :-2], dtype=torch.long).to(self.device)
        # here we don't need to un-squeeze dim, since we slice 2 columns not select 1 column
        train_y = torch.tensor(train[:, -2:], dtype=torch.float).to(self.device)
        valid_y = torch.tensor(valid[:, -2:], dtype=torch.float).to(self.device)
        test_y = torch.tensor(test[:, -2:], dtype=torch.float).to(self.device)

        return field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class BatchLoader:
    """
    Loading the dataset by batch
    """

    def __init__(self, x, y, batch_size=128, shuffle=True):
        # check length should be equal
        assert len(x) == len(y)

        self.batch_size = batch_size

        # shuffle the dataset
        if shuffle:
            seq = list(range(len(x)))
            # shuffle the dataset each sample index
            np.random.shuffle(seq)
            self.x = x[seq]
            self.y = y[seq]
        else:
            self.x = x
            self.y = y

    def __iter__(self):
        # return an iteration yield result
        def iteration(x, y, batch_size):
            start = 0
            end = batch_size
            while start < len(x):
                yield x[start: end], y[start: end]
                start = end
                end += batch_size

        return iteration(self.x, self.y, self.batch_size)
