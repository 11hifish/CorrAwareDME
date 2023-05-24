import numpy as np
from scipy.spatial.distance import cdist
from encoders import *


def compute_R(X):
    n = X.shape[0]
    D = cdist(X, X, metric=lambda x, y: np.dot(x, y))
    print(D)
    R1 = np.sum(D[np.arange(n), np.arange(n)])
    D[np.arange(n), np.arange(n)] = 0
    R2 = np.sum(D)
    R = R2 / R1
    return R


def split_data_across_clients(X, n_clients):
    batch = X.shape[0] // n_clients
    X_list = []
    for i in range(n_clients):
        X_i = X[i * batch: (i+1) * batch]
        X_list.append(X_i)
        print('Client {}, X_i shape: {}'.format(i, X_i.shape))
    return X_list

def split_data_across_clients_Xy(X, y, n_clients):
    batch = X.shape[0] // n_clients
    X_list, y_list = [], []
    for i in range(n_clients):
        X_i = X[i * batch: (i+1) * batch]
        X_list.append(X_i)
        y_i = y[i * batch: (i+1) * batch]
        y_list.append(y_i)
        print('Client {}, X_i shape: {}, y_i shape: {}'.format(i, X_i.shape, y_i.shape))
    return X_list, y_list


def split_data_across_clients_Non_IID(X, y, n_clients):
    # sorted data X
    n_classes = len(np.unique(y))
    n_cls_samples = len(np.where(y == 0)[0])  # 1000
    print('n_cls_samples: ', n_cls_samples)
    # each client gets 2 shards of samples
    n_in_shard = X.shape[0] // n_clients // 2  # num samples in a shard
    print('n in shard: ', n_in_shard)
    n_shards_per_cls = n_cls_samples // n_in_shard
    print('n_shards_per_cls: ', n_shards_per_cls)
    S = []
    for i in range(n_classes):
        cls_shards = []
        cls_idx = np.where(y == i)[0]
        X_cls = X[cls_idx]
        for shard_idx in range(n_shards_per_cls):
            cls_shards.append(X_cls[shard_idx * n_in_shard:(shard_idx + 1) * n_in_shard])
        S.append(cls_shards)
    # now assign shards to clients
    X_list = []
    for client_idx in range(n_clients):
        cls_1_idx = client_idx % (n_classes // 2) * 2
        cls_2_idx = client_idx % (n_classes // 2) * 2 + 1
        shard_idx = client_idx // (n_classes // 2)
        print(client_idx, (cls_1_idx, shard_idx), (cls_2_idx, shard_idx))
        D1 = S[cls_1_idx][shard_idx]
        D2 = S[cls_2_idx][shard_idx]
        D = np.vstack((D1, D2))
        X_list.append(D)
    return X_list


def split_data_across_clients_Xy_Non_IID(X, y, n_clients):
    batch = X.shape[0] // n_clients
    sorted_idx = np.argsort(y)
    y_sorted = y[sorted_idx]
    X_sorted = X[sorted_idx]
    X_list, y_list = [], []
    for i in range(n_clients):
        X_i = X_sorted[i * batch: (i+1) * batch]
        X_list.append(X_i)
        y_i = y_sorted[i * batch: (i+1) * batch]
        y_list.append(y_i)
        print('Client {}, X_i shape: {}, y_i shape: {}'.format(i, X_i.shape, y_i.shape))
    return X_list, y_list


encoder_map = {
    'rand_k': RandkEncoder,
    'rand_k_spatial': RandkSpatialEncoder,
    'rand_proj_spatial': RandProjSpatialEncoderSRHT,
    'rand_k_wangni': RandkWangni,
    'induced': InducedTopkRandk
}

encoder_rename = {
    'rand_k': 'Rand-k',
    'rand_k_spatial': 'Rand-k-Spatial(Avg)',
    'rand_proj_spatial': 'Rand-Proj-Spatial(Avg)',
    'rand_k_wangni': 'Rand-k(Wangni)',
    'induced': 'Induced Compressor'
}


# if __name__ == '__main__':
    # with open('data/fashion_mnist_test_32x32_power_iter_sorted.pkl', 'rb') as f:
    #     X, y = pickle.load(f)
    # n_clients = 10
    # split_data_across_clients_Non_IID(X, y, n_clients)
    # with open('data/UJIndoorLoc_lin_reg.pkl', 'rb') as f:
    #     X, y = pickle.load(f)
    # n_clients = 10
    # split_data_across_clients_Xy_Non_IID(X, y, n_clients)

