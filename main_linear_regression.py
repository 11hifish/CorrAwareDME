import numpy as np
from utils import split_data_across_clients_Xy, split_data_across_clients_Xy_Non_IID, encoder_map, compute_R
import pickle
import os
import time


DATA_FILE = os.path.join('data', 'UJIndoorLoc_lin_reg.pkl')
# DATA_FILE = os.path.join('data', 'puma32_linear_regression.pkl')
LEARNING_RATE = 0.001



def server_aggregation(x_hat_list, G_list, encoder):
    if encoder is None:
        x_hat = np.mean(x_hat_list, axis=0)
    else:
        x_hat = encoder.decode(x_hat_list, G_list)
    return x_hat

def compute_gradient(X, y, w):
    return X.T @ (X @ w - y) / X.shape[0]

def client_subroutine(X, y, w, encoder):
    grad = compute_gradient(X, y, w)
    # encode v
    if encoder is None:  # no reduction
        encoded_grad = grad
        G = np.eye(X.shape[1])
    else:
        encoded_grad, G = encoder.encode(grad)
    return encoded_grad, G


def compute_squared_error(X_list, y_list, w, grad_hat):
    N = np.vstack([compute_gradient(X_list[i], y_list[i], w) for i in range(len(X_list))])
    grad_bar = np.mean(N, axis=0)
    squared_error = np.sum((grad_bar - grad_hat) ** 2)
    return squared_error


def compute_objective(X, y, w):
    return 0.5 * np.mean((X @ w - y) ** 2)


def perform_linear_regression(X, y, X_list, y_list, n_clients, k, n_iter, init_vec, encoder_name):
    encoder = None
    if encoder_name is not None:
        print('Generating encoder : {}'.format(encoder_name))
        encoder_fn = encoder_map[encoder_name]
        encoder = encoder_fn(n_clients, X.shape[1], k)
    # X_list, y_list = split_data_across_clients_Xy(X, y, n_clients)
    w = init_vec
    # recorders
    all_se = np.zeros(n_iter)
    all_obj_val = np.zeros(n_iter + 1)
    all_obj_val[0] = compute_objective(X, y, w)
    for iter_idx in range(n_iter):
        t1 = time.time()
        # encode
        grad_hat_list, G_list = [], []
        for client_idx in range(n_clients):
            grad_hat_i, G_i = client_subroutine(X_list[client_idx], y_list[client_idx], w, encoder)
            grad_hat_list.append(grad_hat_i)
            G_list.append(G_i)
        # decode
        grad_hat = server_aggregation(grad_hat_list, G_list, encoder)
        # compute SE
        squared_error = compute_squared_error(X_list, y_list, w, grad_hat)
        all_se[iter_idx] = squared_error
        # update param
        w -= LEARNING_RATE * grad_hat
        # compute objective error
        obj_val = compute_objective(X, y, w)
        all_obj_val[iter_idx + 1] = obj_val
        t2 = time.time()
        print('iter: {}, SE: {:.4f}, Obj val: {:.8f}, time: {:.4f} s'
              .format(iter_idx, squared_error, obj_val, t2 - t1))
    return all_se, all_obj_val


def main():
    IID = False
    # hyperparameters
    n_iter = 50
    n_clients = 10
    k = 5
    # get data
    with open(DATA_FILE, 'rb') as f:
        X, y = pickle.load(f)
    X = X.astype(np.float64) / 100
    print('X shape: ', X.shape)
    if IID:
        X_list, y_list = split_data_across_clients_Xy(X, y, n_clients)
    else:
        X_list, y_list = split_data_across_clients_Xy_Non_IID(X, y, n_clients)

    # encoder_names = ['rand_k', 'rand_k_spatial', 'rand_k_wangni', 'induced', 'rand_proj_spatial']
    encoder_names = ['rand_proj_spatial']
    num_exp = 10
    import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    exp_idx = 0
    # for exp_idx in range(num_exp):
    while exp_idx < num_exp:
        for encoder_name in encoder_names:
            # encoder_name = None
            init_w = np.ones(X.shape[1])
            try:
                all_se, all_obj_val = perform_linear_regression(X=X, y=y, X_list=X_list, y_list=y_list, n_clients=n_clients, k=k, n_iter=n_iter,
                                                                init_vec=init_w, encoder_name=encoder_name)
            except np.linalg.LinAlgError:
                continue
            save_folder = 'indoor_res_noniid'
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            save_path = os.path.join(save_folder, 'indoor_{}_iter_{}_n_{}_k_{}_exp_{}.pkl'
                                     .format(encoder_name, n_iter, n_clients, k, exp_idx))
            with open(save_path, 'wb') as f:
                pickle.dump((all_se, all_obj_val), f)

            exp_idx += 1
    #         axes[0].plot(np.arange(n_iter), all_se, label=encoder_name)
    #         axes[1].plot(np.arange(n_iter + 1)[10:], all_obj_val[10:], label=encoder_name)
    # axes[0].legend()
    # axes[1].legend()
    # plt.show()

def linear_regression_single_machine():
    with open(DATA_FILE, 'rb') as f:
        X, y = pickle.load(f)
    # X = X.astype(np.float64) / 100
    print(X.shape)
    X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))
    print(X.shape)
    n_iter = 30
    w = np.ones(X.shape[1])
    all_obj_val = np.zeros(n_iter)
    for iter_idx in range(n_iter):
        grad = X.T @ (X @ w - y) / X.shape[0]
        w -= LEARNING_RATE * grad
        obj_val = 0.5 * np.mean((X @ w - y) ** 2)
        print('iter: {}, obj : {}'.format(iter_idx, obj_val))
        all_obj_val[iter_idx] = obj_val
    import matplotlib.pyplot as plt
    plt.plot(np.arange(n_iter), all_obj_val)
    plt.show()


if __name__ == '__main__':
    # linear_regression_single_machine()
    main()

