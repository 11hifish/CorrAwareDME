import numpy as np
from rand_k_spatial import compute_c1_c2_beta_bar
import matplotlib.pyplot as plt


def compute_MSE_full_corr(d, n, k_vals):
    mse_rand_k = d / (n * k_vals) - 1 / n
    mse_general = d / (n * k_vals) - 1
    c1_vals = np.zeros(len(k_vals))
    c2_vals = np.zeros(len(k_vals))
    for i, k in enumerate(k_vals):
        _, c1, c2 = compute_c1_c2_beta_bar(n=n, dim=d, k=k, T_fn=lambda x: x)
        c1_vals[i] = c1
        c2_vals[i] = c2
    mse_rk_spatial = d / (n * k_vals) - (1 - c1_vals + c2_vals * (n - 1)) / n
    return mse_rand_k, mse_rk_spatial, mse_general


def compare_and_plot_MSE_full_corr():
    d_vals = [1024, 1024, 1024, 1024]
    n_vals = [10, 20, 50, 100]
    k_vals = np.array([[20, 40, 60, 80, 100],
                       [10, 20, 30, 40, 50],
                       [4, 8, 12, 16, 20],
                       [2, 4, 6, 8, 10]])
    fig, axes = plt.subplots(1, len(n_vals), figsize=(16, 3))
    for idx in range(len(n_vals)):
        d = d_vals[idx]
        n = n_vals[idx]
        kv = k_vals[idx]
        mse_rand_k, mse_rk_spatial, mse_general = compute_MSE_full_corr(d, n, kv)
        ax = axes[idx]
        size = 5
        ax.plot(kv, mse_rand_k, 'bo-', label='Rand-k', markersize=size)
        ax.plot(kv, mse_rk_spatial, 'go-', label='Rand-k-Spatial(Max)', markersize=size)
        ax.plot(kv, mse_general, 'ro-', label='Rand-Proj-Spatial(Max)', markersize=size)
        ax.grid()
        fsize = 15
        ax.set_xticks(kv)
        ax.set_xticklabels(kv, fontsize=fsize)
        ax.legend(fontsize=12)
        labels = ax.get_yticklabels()
        ax.set_yticklabels(labels, fontsize=fsize)
        ax.set_title('n = {}, d = {}'.format(n, d), fontsize=fsize)
        ax.set_xlabel('k', fontsize=fsize)
        ax.set_ylabel('MSE', fontsize=fsize)
    # plt.show()
    plt.savefig('full_corr_comp.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    compare_and_plot_MSE_full_corr()
