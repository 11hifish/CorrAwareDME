import numpy as np
import matplotlib.pyplot as plt
from rand_matrix_vec import generate_corr_vecs
from rand_k_spatial import compute_MSE_Rand_k_Spatial
from general_decoding_srht import compute_MSE_srht_decoding
import os


def compute_MSE_varying_degree(n, k, d, R_vals, num_exp=1000):
    mse_rand_k = np.zeros(len(R_vals))
    mse_rk_spatial_best = np.zeros(len(R_vals))
    mse_general = np.zeros(len(R_vals))
    for i, R in enumerate(R_vals):
        print('R: ', R)
        X = generate_corr_vecs(n, d, R)
        mse_rand_k[i] = (d / k - 1) / (n ** 2) * np.sum(X ** 2)
        mse_rk_spatial_best[i] = compute_MSE_Rand_k_Spatial(X, k, R)
        fn = lambda x: 1 + R / (n - 1) * (x - 1)
        mse_general[i] = compute_MSE_srht_decoding(X, k, num_exp=num_exp, **{'fn': fn})
    return mse_rand_k, mse_rk_spatial_best, mse_general


def compute_and_save_beta_bar_srht_decoding(n, k, d, R, num_exp=1000):
    print('n = {}, k = {}, d = {}, R = {}'.format(n, k, d, R))
    fn = lambda x: 1 + R / (n - 1) * (x - 1)
    X = generate_corr_vecs(n, d, R)
    save_folder = 'results_2'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    beta_bar_path = os.path.join(save_folder, 'beta_bar_n_{}_k_{}_d_{}_R_{}.pkl'.format(n, k, d, R))
    mse_path = os.path.join(save_folder, 'mse_n_{}_k_{}_d_{}_R_{}.pkl'.format(n, k, d, R))
    A_path = os.path.join(save_folder, 'A_n_{}_k_{}_d_{}_R_{}.pkl'.format(n, k, d, R))
    dic = {
        'fn': fn,
        'beta_bar_path': beta_bar_path,
        'mse_path': mse_path,
        'A_path': A_path
    }
    mse = compute_MSE_srht_decoding(X, k, num_exp=num_exp, **dic)
    print('n : {}, k : {}, d : {}, R : {}'.format(n, k, d, R))
    print('mse: {}'.format(mse))


def main_21():
    n = 21
    d = 1024
    R_vals = [0, 4, 8, 12, 16, 20]
    k_vals = [10, 20, 30, 40]
    for i, k in enumerate(k_vals):
        print('k = {}'.format(k))
        num_exp = 10
        mse_rand_k, mse_rk_spatial_best, mse_general = compute_MSE_varying_degree(n, k, d, R_vals, num_exp=num_exp)
        print('k: ')
        print(mse_rand_k)
        print(mse_rk_spatial_best)
        print(mse_general)


if __name__ == '__main__':
    # main_21()
    n = 5
    k = 1
    d = 1024
    R = 1
    num_exp = 1000
    compute_and_save_beta_bar_srht_decoding(n=n, k=k, d=d, R=R, num_exp=num_exp)
