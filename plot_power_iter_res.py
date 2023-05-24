import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from utils import encoder_rename

IID = False
suffix = '' if IID else '_noniid'
# save_folder = 'fashion_mnist_32x32_power_iter_res'
save_folder = 'fashion_mnist_32x32_power_iter_res{}'.format(suffix)
n_iter = 30
# n_clients, k = 10, 50
# n_clients, k = 5, 100
# num_exp = 10
exp_indices = [0,2,4,6,8]
num_exp=len(exp_indices)
# n_client_k = [(10, 50), (5, 100)]
# n_client_k = [(10, 102), (50, 20)]
n_client_k = [(50, 5), (50, 10)]
# n_client_k = [(50, 20)]
encoder_names = ['rand_k', 'rand_k_spatial', 'rand_k_wangni', 'induced', 'rand_proj_spatial']

fig, axes = plt.subplots(1, len(n_client_k) * 2, figsize=(16, 3))
fsize = 12
legend_size = 10
for idx, (n_clients, k) in enumerate(n_client_k):
    for encoder_name in encoder_names:
        all_se = np.zeros((num_exp, n_iter))
        all_obj_val = np.zeros((num_exp, n_iter + 1))
        # for exp_idx in range(num_exp):
        for j, exp_idx in enumerate(exp_indices):
            if IID:
                middle = 'iid'
            else:
                middle = ''
            save_path = os.path.join(save_folder, 'fashion_mnist_test{}_{}_iter_{}_n_{}_k_{}_exp_{}.pkl'
                                    .format(middle, encoder_name, n_iter, n_clients, k, exp_idx))
            with open(save_path, 'rb') as f:
                se, obj_val = pickle.load(f)
            all_se[j] = se
            all_obj_val[j] = obj_val
        mean_se, std_se = np.mean(all_se, axis=0), np.std(all_se, axis=0)
        mean_obj_val, std_obj_val = np.mean(all_obj_val, axis=0), np.std(all_obj_val, axis=0)
        axes[idx * 2].plot(np.arange(n_iter), mean_se, label=encoder_rename[encoder_name])
        axes[idx * 2].fill_between(np.arange(n_iter), mean_se + std_se, mean_se - std_se, alpha=0.2)
        axes[idx * 2 + 1].plot(np.arange(n_iter+1), mean_obj_val, label=encoder_rename[encoder_name])
        axes[idx * 2 + 1].fill_between(np.arange(n_iter+1), mean_obj_val + std_obj_val, mean_obj_val - std_obj_val, alpha=0.2)
    axes[idx * 2].set_title('n={}, k={}, d=1024'.format(n_clients, k), fontsize=fsize)
    axes[idx * 2].set_ylabel('Squared Error', fontsize=fsize, labelpad=0)
    axes[idx * 2].set_xlabel('Iteration No.', fontsize=fsize)
    axes[idx * 2].legend(fontsize=legend_size)
    axes[idx * 2].grid()
    axes[idx * 2 + 1].set_title('n={}, k={}, d=1024'.format(n_clients, k), fontsize=fsize)
    axes[idx * 2 + 1].set_ylabel('Loss', fontsize=fsize, labelpad=0)
    axes[idx * 2 + 1].set_xlabel('Iteration No.', fontsize=fsize)
    axes[idx * 2 + 1].legend(fontsize=legend_size)
    axes[idx * 2 + 1].grid()
# plt.show()
fig.suptitle('Distributed Power Iteration (Non-IID)', fontsize=16)
plt.subplots_adjust(top=0.8)
plt.savefig('power_iter_fmnist_32x32_noniid_n_50_2.pdf', bbox_inches='tight', pad_inches=0.2)
plt.close()
