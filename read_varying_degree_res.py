import numpy as np
import pickle
import os
from rand_matrix_vec import generate_corr_vecs
from rand_k_spatial import compute_MSE_Rand_k_Spatial
import matplotlib.pyplot as plt

d = 1024

# n = 21
# k_vals = [10, 20, 30, 40]
# R_vals = [4, 8, 12, 16]
# save_folder = 'genspa_res'

# n = 51
# k_vals = [4, 8, 12, 16]
# R_vals = [10, 20, 30, 40]
# save_folder = 'genspa_51_res'

# n = 11
n = 5
# k_vals = [1, 5, 10, 15]
k_vals = [50, 100, 150, 200]
# k_vals = [30,50,70,90]
R_vals = [1,2,3,4]
# R_vals = [2,4,6,8]
save_folder = 'genspa_{}_res'.format(n)

fig, axes = plt.subplots(1, len(R_vals), figsize=(16, 3))
for k_idx, k in enumerate(k_vals):
    mse_rand_k = np.zeros(len(R_vals))
    mse_rk_spatial_best = np.zeros(len(R_vals))
    mse_general = np.zeros(len(R_vals))
    for R_idx, R in enumerate(R_vals):
        mse_path = os.path.join(save_folder, 'mse_n_{}_k_{}_d_{}_R_{}.pkl'.format(n, k, d, R))
        with open(mse_path, 'rb') as f:
            all_mse = pickle.load(f)
        mse_general[R_idx] = np.mean(all_mse)
        X = generate_corr_vecs(n, d, R)
        mse_rand_k[R_idx] = (d / k - 1) / (n ** 2) * np.sum(X ** 2)
        mse_rk_spatial_best[R_idx] = compute_MSE_Rand_k_Spatial(X, k, R)
    ax = axes[k_idx]
    size = 5
    ax.plot(R_vals, mse_rand_k, 'bo-', label='Rand-k', markersize=size)
    ax.plot(R_vals, mse_rk_spatial_best, 'go-', label='Rand-k-Spatial(Opt)', markersize=size)
    ax.plot(R_vals, mse_general, 'ro-', label='Rand-Proj-Spatial', markersize=size)
    ax.grid()
    fsize = 15
    ax.set_xticks(R_vals)
    ax.set_xticklabels(R_vals, fontsize=fsize)
    ax.legend(fontsize=12)
    labels = ax.get_yticklabels()
    ax.set_yticklabels(labels, fontsize=fsize)
    ax.set_title('n = {}, k = {}'.format(n, k), fontsize=fsize)
    ax.set_xlabel('$\mathcal{R}$', fontsize=fsize)
    if k_idx == 0:
        ax.set_ylabel('MSE', fontsize=fsize)

# plt.show()
plt.savefig('varying_deg_corr_n_{}_k_large.pdf'.format(n), bbox_inches='tight', pad_inches=0.1)
plt.close()
