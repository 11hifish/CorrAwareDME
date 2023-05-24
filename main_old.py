from rand_matrix_vec import generate_testing_vecs
from rand_k_spatial import compute_MSE_Rand_k, compute_MSE_Rand_k_Spatial
import numpy as np
import matplotlib.pyplot as plt


def main():
    n = 4
    d = 32
    k_vals = [2, 4, 6, 8, 10]
    mse_rand_k = np.zeros(len(k_vals))
    vec_type = 'ortho'
    for i, k in enumerate(k_vals):
        X = generate_testing_vecs(n, d, vec_type=vec_type)
        mse_rand_k[i] = compute_MSE_Rand_k(X, k)
    plt.plot(k_vals, mse_rand_k, label='rand k')
    plt.legend()
    plt.show()

def test_beta_bar():
    from general_decoding_srht import simulate_beta_bar
    from rand_k_spatial import compute_c1_c2_beta_bar
    n = 5
    d = 32
    k_vals = [2, 4, 6, 8, 10]
    for k in k_vals:
        beta_bar = simulate_beta_bar(n, k, d, matrix_type='simple', num_exp=2000)
        beta_bar_spatial, _, _ = compute_c1_c2_beta_bar(n, d, k, T_fn=lambda x: 1)
        print('n = {}, k = {}, d = {}, beta bar: {}, beta bar spatial: {}, d/k: {}'
              .format(n, k, d, beta_bar, beta_bar_spatial, d / k))

def test_GTG():
    n = 2
    k = 2
    d = 4
    from rand_matrix_vec import generate_random_matrix
    Gs = [generate_random_matrix(k, d, 'simple') for i in range(n)]
    G = np.vstack(Gs)
    print(G)
    print(G.T @ G)
    print(np.linalg.pinv(G.T @ G))


def test_testing_corr_vec():
    from rand_matrix_vec import generate_corr_vecs
    from utils import compute_R
    d = 1024
    R = 8
    n = 11
    X = generate_corr_vecs(n, d, R)
    R_ = compute_R(X)
    print(R, R_)


if __name__ == '__main__':
    # main()
    # test_beta_bar()
    # test_GTG()
    test_testing_corr_vec()
