import numpy as np
from utils import split_data_across_clients, split_data_across_clients_Non_IID, encoder_map
import pickle
import os
import time
from sklearn.cluster._kmeans import kmeans_plusplus
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


DATA_FILE = os.path.join('data', 'fashion_mnist_test_32x32_k_means.pkl')
CENTER_PATH = os.path.join('data', 'fashion_mnist_test_k_means_centers.pkl')

def metric_fn(x, y):
    return np.linalg.norm(x - y) ** 2

def get_init_center(X, n_clusters, save_path=None):
    if save_path and os.path.isfile(save_path):
        with open(save_path, 'rb') as f:
            center, idx = pickle.load(f)
    else:
        center, idx = kmeans_plusplus(X, n_clusters=n_clusters)
        with open(save_path, 'wb') as f:
            pickle.dump((center, idx), f)
    return center, idx


def server_aggregation(x_hat_list, G_list, encoder):
    if encoder is None:
        x_hat = np.mean(x_hat_list, axis=0)
    else:
        x_hat = encoder.decode(x_hat_list, G_list)
    return x_hat


def client_subroutine(X, centers, encoder):
    n_clusters, d = centers.shape

    D = cdist(X, centers, metric=metric_fn)  # n x c
    new_center_idx = np.argmin(D, axis=1)  # n
    new_centers = []
    for cls_no in range(n_clusters):
        idx = np.where(new_center_idx == cls_no)[0]
        if len(idx) == 0:  # no data gets assigned to this cluster
            new_center_cls = np.zeros(d)
        else:
            X_cls = X[idx]
            new_center_cls = np.mean(X_cls, axis=0)
        new_centers.append(new_center_cls)
    new_centers = np.vstack(new_centers)
    # encode
    if encoder is None:  # no reduction
        encoded_centers = new_centers
        G_i_list = [np.eye(X.shape[1]) for _ in range(n_clusters)]
    else:
        encoded_centers = []
        G_i_list = []
        for i in range(n_clusters):
            x_hat_i, G_i = encoder.encode(new_centers[i])
            encoded_centers.append(x_hat_i)
            G_i_list.append(G_i)
        encoded_centers = np.vstack(encoded_centers)
    return encoded_centers, G_i_list, new_centers


def compute_objective(X, centers):
    D = cdist(X, centers, metric=metric_fn)
    Dmin = np.min(D, axis=1).ravel()
    obj_val = np.sum(Dmin)
    return obj_val

def compute_avg_squared_error(true_centers_list, new_centers):
    # mean across 10 clusters
    # true centers list has size (n_clients, n_centers, d)
    # new centers has size (n_clusters, d)
    total_se = 0
    n_clients = len(true_centers_list)
    n_clusters = new_centers.shape[0]
    for cls_no in range(n_clusters):
        center_bar = np.mean([true_centers_list[client_idx][cls_no] for client_idx in range(n_clients)], axis=0)
        center_hat = new_centers[cls_no]
        squared_error = np.sum((center_bar - center_hat) ** 2)
        total_se += squared_error
    return total_se / n_clusters


def perform_k_means(X, X_list, n_clients, k, n_iter, init_vec, encoder_name):
    encoder = None
    if encoder_name is not None:
        print('Generating encoder : {}'.format(encoder_name))
        encoder_fn = encoder_map[encoder_name]
        encoder = encoder_fn(n_clients, X.shape[1], k)
    # X_list = split_data_across_clients(X, n_clients)
    all_se = np.zeros(n_iter)
    all_obj_val = np.zeros(n_iter+1)
    all_obj_val[0] = compute_objective(X, init_vec)
    centers = init_vec
    n_clusters = init_vec.shape[0]
    iter_idx = 0
    # for iter_idx in range(n_iter):
    while iter_idx < n_iter:
        t1 = time.time()
        # encode
        encoded_centers_list, G_list = [], []  # encoded_centers_list list has size (n_clients, n_centers)
        true_centers_list = []
        for client_idx in range(n_clients):
            encoded_centers, G_i_list, true_centers_i = client_subroutine(X_list[client_idx], centers, encoder)
            encoded_centers_list.append(encoded_centers)
            G_list.append(G_i_list)
            true_centers_list.append(true_centers_i)
        # decode
        new_centers = np.zeros((n_clusters, X.shape[1]))  # aggregated centers
        try:
            for cls_no in range(n_clusters):
                x_hat_cls_list = [encoded_centers_list[client_idx][cls_no] for client_idx in range(n_clients)]
                G_cls_list = [G_list[client_idx][cls_no] for client_idx in range(n_clients)]
                x_hat_cls = server_aggregation(x_hat_cls_list, G_cls_list, encoder)
                new_centers[cls_no] = x_hat_cls
        except np.linalg.LinAlgError:
            print('LinAlgError at iteration: {}'.format(iter_idx))
            continue
        # compute avg SE
        avg_se = compute_avg_squared_error(true_centers_list, new_centers)
        all_se[iter_idx] = avg_se
        # compute objective
        obj_val = compute_objective(X, new_centers)
        all_obj_val[iter_idx + 1] = obj_val
        # update centers
        centers = new_centers
        t2 = time.time()
        print('iter: {}, SE: {:.4f}, Obj val: {:.8f}, time: {:.4f} s'
              .format(iter_idx, avg_se, obj_val, t2 - t1))
        iter_idx += 1
    return all_se, all_obj_val


def main():
    IID = False
    n_iter = 30
    n_clients = 10
    k = 5
    encoder_names = ['rand_proj_spatial']
    # fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    num_exp = 10
    # get data
    if IID:
        with open(DATA_FILE, 'rb') as f:
            X, y = pickle.load(f)
        X_list = split_data_across_clients(X, n_clients)
    else:
        with open('data/fashion_mnist_test_32x32_power_iter_sorted.pkl', 'rb') as f:
            X, y = pickle.load(f)
        X_list = split_data_across_clients_Non_IID(X, y, n_clients)
    print('X shape: ', X.shape)
    n_clusters = len(np.unique(y))
    print('n_clusters: ', n_clusters)

    for exp_idx in range(num_exp):
    # while exp_idx < num_exp:
        for encoder_name in encoder_names:
            # encoder_name = None
            # get init centers
            init_center, idx = get_init_center(X, n_clusters=n_clusters, save_path=CENTER_PATH)
            all_se, all_obj_val = perform_k_means(X=X, X_list=X_list, n_clients=n_clients, k=k, n_iter=n_iter,
                                                  init_vec=init_center, encoder_name=encoder_name)
            save_folder = 'fashion_mnist_32x32_k_means_res_noniid'
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            save_path = os.path.join(save_folder, 'fashion_mnist_test_{}_iter_{}_n_{}_k_{}_exp_{}.pkl'
                                     .format(encoder_name, n_iter, n_clients, k, exp_idx))
            with open(save_path, 'wb') as f:
                pickle.dump((all_se, all_obj_val), f)
            # exp_idx += 1

    #     axes[0].plot(np.arange(n_iter), all_se, label=encoder_name)
    #     axes[1].plot(np.arange(n_iter + 1), all_obj_val, label=encoder_name)
    # axes[0].legend()
    # axes[1].legend()
    # plt.show()


def main_k_means_single_machine():
    with open(DATA_FILE, 'rb') as f:
        X, y = pickle.load(f)
    n_clusters = len(np.unique(y))
    center, idx = get_init_center(X, n_clusters=n_clusters, save_path=CENTER_PATH)
    n_iter = 50
    all_obj_vals = np.zeros(n_iter)
    for iter_idx in range(n_iter):
        km = KMeans(n_clusters=n_clusters, init=center, n_init=1, max_iter=1)
        km.fit(X)
        center = km.cluster_centers_
        obj_val = km.inertia_
        print('iter : {}, obj val: {}'.format(iter_idx, obj_val))
        all_obj_vals[iter_idx] = obj_val
    import matplotlib.pyplot as plt
    plt.plot(np.arange(n_iter), all_obj_vals)
    plt.show()


if __name__ == '__main__':
    # main_k_means_single_machine()
    main()

