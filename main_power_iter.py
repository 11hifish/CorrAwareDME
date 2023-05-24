import numpy as np
from utils import split_data_across_clients, split_data_across_clients_Non_IID, encoder_map
import pickle
import os
import time


def server_aggregation(x_hat_list, G_list, encoder):
    if encoder is None:
        x_hat = np.mean(x_hat_list, axis=0)
    else:
        x_hat = encoder.decode(x_hat_list, G_list)
    return x_hat


def client_subroutine(C, vec, encoder):
    v = C @ vec
    # encode v
    if encoder is None:  # no reduction
        v_hat = v
        G = np.eye(len(vec))
    else:
        v_hat, G = encoder.encode(v)
    return v_hat, G


def compute_squared_error(Cov_list, vec, x_hat):
    x_bar = np.mean([Cov_list[i] @ vec for i in range(len(Cov_list))], axis=0)
    squared_error = np.sum((x_hat - x_bar) ** 2)
    return squared_error


def get_top_eigvec(C):
    print('Cov shape: ', C.shape)
    eigvals, V = np.linalg.eig(C)
    eigvals = eigvals.real
    V = V.real
    max_idx = np.argmax(eigvals)
    top_eigvec = V[:, max_idx]
    return top_eigvec


def compute_objective(v_top, x_hat):
    x_hat_normalized = x_hat / np.linalg.norm(x_hat)
    return np.linalg.norm(v_top - x_hat_normalized)


def perform_power_iteration(X_list, n_clients, k, n_iter, init_vec, encoder_name):
    # note X should be centered at 0 here
    encoder = None
    if encoder_name is not None:
        print('Generating encoder : {}'.format(encoder_name))
        encoder_fn = encoder_map[encoder_name]
        encoder = encoder_fn(n_clients, X_list[0].shape[1], k)
    # X_list = split_data_across_clients(X, n_clients)
    Cov_list = []
    for i in range(n_clients):
        X_i = X_list[i]
        Cov_list.append(X_i.T @ X_i)
    # send Cov in cov list to clients
    # cov_vec_list = [Cov_list[i] @ init_vec for  i in range(n_clients)]
    Cov = np.sum(Cov_list, axis=0)
    top_eigvec = get_top_eigvec(Cov)
    vec = init_vec
    # recorders
    all_se = np.zeros(n_iter)
    all_obj_val = np.zeros(n_iter+1)
    all_obj_val[0] = compute_objective(top_eigvec, vec)
    for iter_idx in range(n_iter):
        t1 = time.time()
        # encode
        x_hat_list, G_list = [], []
        for client_idx in range(n_clients):
            x_hat, G = client_subroutine(Cov_list[client_idx], vec, encoder)
            x_hat_list.append(x_hat)
            G_list.append(G)
        # decode
        x_hat = server_aggregation(x_hat_list, G_list, encoder)
        # compute SE
        squared_error = compute_squared_error(Cov_list, vec, x_hat)
        all_se[iter_idx] = squared_error
        # compute objective error
        obj_val = compute_objective(top_eigvec, x_hat)
        all_obj_val[iter_idx+1] = obj_val
        # update vec
        vec = x_hat / np.linalg.norm(x_hat)
        t2 = time.time()
        print('iter: {}, SE: {:.4f}, Obj val: {:.8f}, time: {:.4f} s'
              .format(iter_idx, squared_error, obj_val, t2 - t1))
    return all_se, all_obj_val


def main():
    IID = False
    # hyperparameters
    n_iter = 30
    n_clients = 50
    k = 20
    # get data
    if IID:
        with open(os.path.join('data', 'fashion_mnist_test_32x32_power_iter.pkl'),  'rb') as f:
            X = pickle.load(f)
            # X = X.astype(np.float64)
            # X /= 255  # normalize each pixel value to be within [0, 1]
            X = X - np.mean(X, axis=0)  # make it center at 0
        X_list = split_data_across_clients(X, n_clients)
    else:  # Non-IID
        with open(os.path.join('data', 'fashion_mnist_test_32x32_power_iter_sorted.pkl'),  'rb') as f:
            X, y = pickle.load(f)
            X = X - np.mean(X, axis=0)  # make it center at 0
        X_list = split_data_across_clients_Non_IID(X, y, n_clients)
    print('X shape: ', X.shape)
    # compute init vec
    init_vec = np.ones(X.shape[1]) / np.sqrt(X.shape[1])
    print('init vec shape: ', init_vec.shape)
    # encoder_names = ['rand_k', 'rand_k_spatial', 'rand_proj_spatial']
    # encoder_names = ['rand_k_spatial', 'rand_proj_spatial']
    encoder_names = ['rand_k', 'induced']
    num_exp = 10
    exp_idx = 0
    while exp_idx < num_exp:
    # for exp_idx in range(num_exp):
        for encoder_name in encoder_names:
            # encoder_name = None
            try:
                all_se, all_obj_val = perform_power_iteration(X_list, n_clients=n_clients, k=k, n_iter=n_iter, init_vec=init_vec,
                                                              encoder_name=encoder_name)
            except np.linalg.LinAlgError:
                continue
            suffix = '' if IID else '_noniid'
            save_folder = 'fashion_mnist_32x32_power_iter_res{}'.format(suffix)
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            save_path = os.path.join(save_folder, 'fashion_mnist_test_iid_{}_iter_{}_n_{}_k_{}_exp_{}.pkl'
                                     .format(encoder_name, n_iter, n_clients, k, exp_idx))
            with open(save_path, 'wb') as f:
                pickle.dump((all_se, all_obj_val), f)
            exp_idx += 1


def main_power_iter_single_machine():
    # get data
    data_file_name = 'fashion_mnist_test_32x32_power_iter.pkl'
    # data_file_name = 'fashion_mnist_test_d_512_power_iter_iid.pkl'
    with open(os.path.join('data', data_file_name), 'rb') as f:
        X = pickle.load(f)
    # X = X.astype(np.float64)
    # X /= 255
    d = X.shape[1]
    print('X shape:', X.shape)
    centered_X = X - np.mean(X, axis=0)
    C = centered_X.T @ centered_X  # d x d covariance matrix
    top_eigvec = get_top_eigvec(C)
    n_iter = 30
    # vec = np.ones(d) / np.sqrt(d)
    vec = np.zeros(d)
    vec[0] = 1
    for iter_idx in range(n_iter):
        vec = C @ vec
        vec /= np.linalg.norm(vec)
        diff = np.linalg.norm(vec - top_eigvec)
        print('iter: {}, diff: {}'.format(iter_idx, diff))


def check_data():
    # get data
    with open(os.path.join('data', 'fashion_mnist_test_d_512_power_iter_iid.pkl'), 'rb') as f:
        X = pickle.load(f)
    print('X max: {}, X min: {}'.format(np.max(X), np.min(X)))


if __name__ == '__main__':
    # check_data()
    main()
    # main_power_iter_single_machine()


