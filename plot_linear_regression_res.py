import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from utils import encoder_rename


save_folder = 'indoor_res_noniid'
n_iter = 50
# n_clients, k = 10, 50
# n_clients, k = 5, 100
num_exp = 10
# n_client_k = [(10, 50), (5, 100)]
# n_client_k = [(50, 1), (50, 5)]
n_client_k = [(50, 10)]
encoder_names = ['rand_k', 'rand_k_spatial', 'rand_k_wangni', 'induced', 'rand_proj_spatial']

fig, axes = plt.subplots(1, len(n_client_k) * 2, figsize=(8, 3))
fsize = 12
legend_size = 10
for idx, (n_clients, k) in enumerate(n_client_k):
    for encoder_name in encoder_names:
        all_se = np.zeros((num_exp, n_iter))
        all_obj_val = np.zeros((num_exp, n_iter + 1))
        for exp_idx in range(num_exp):
            if k == 1 and encoder_name == 'induced':
                encoder_name_here = 'rand_k'
            else:
                encoder_name_here = encoder_name
            save_path = os.path.join(save_folder, 'indoor_{}_iter_{}_n_{}_k_{}_exp_{}.pkl'
                                    .format(encoder_name_here, n_iter, n_clients, k, exp_idx))
            with open(save_path, 'rb') as f:
                se, obj_val = pickle.load(f)
            all_se[exp_idx] = se
            all_obj_val[exp_idx] = obj_val
        mean_se, std_se = np.mean(all_se, axis=0), np.std(all_se, axis=0)
        mean_obj_val, std_obj_val = np.mean(all_obj_val, axis=0), np.std(all_obj_val, axis=0)
        start_idx = 10
        axes[idx * 2].plot(np.arange(n_iter)[start_idx:], mean_se[start_idx:], label=encoder_rename[encoder_name])
        axes[idx * 2].fill_between(np.arange(n_iter)[start_idx:], mean_se[start_idx:] + std_se[start_idx:],
                                   mean_se[start_idx:] - std_se[start_idx:], alpha=0.2)
        axes[idx * 2 + 1].plot(np.arange(n_iter+1)[start_idx:], mean_obj_val[start_idx:], label=encoder_rename[encoder_name])
        axes[idx * 2 + 1].fill_between(np.arange(n_iter+1)[start_idx:], mean_obj_val[start_idx:] + std_obj_val[start_idx:],
                                       mean_obj_val[start_idx:] - std_obj_val[start_idx:], alpha=0.2)
    axes[idx * 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[idx * 2].set_title('n={}, k={}, d=512'.format(n_clients, k), fontsize=fsize)
    axes[idx * 2].set_ylabel('Squared Error', fontsize=fsize)
    axes[idx * 2].set_xlabel('Iteration No.', fontsize=fsize)
    axes[idx * 2].legend(fontsize=legend_size)
    axes[idx * 2].grid()
    axes[idx * 2 + 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[idx * 2 + 1].set_title('n={}, k={}, d=512'.format(n_clients, k), fontsize=fsize)
    axes[idx * 2 + 1].set_ylabel('Loss', fontsize=fsize, labelpad=0)
    axes[idx * 2 + 1].set_xlabel('Iteration No.', fontsize=fsize)
    axes[idx * 2 + 1].legend(fontsize=legend_size)
    axes[idx * 2 + 1].grid()
# plt.show()
fig.suptitle('Distributed Linear Regression (Non-IID)', fontsize=16)
plt.subplots_adjust(top=0.8)
plt.savefig('linear_regression_noniid_n_50_2.pdf', bbox_inches='tight', pad_inches=0.2)
plt.close()
