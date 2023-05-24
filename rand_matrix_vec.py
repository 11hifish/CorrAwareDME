## generate random vectors and random matrices
import numpy as np
from scipy.linalg import hadamard


def generate_random_matrix(k, d, matrix_type, **kwargs):
    # generate a single G_i matrix of size k x d
    if matrix_type.lower() == 'srht':
        # had_dim = int(np.ceil(np.log2(d)))
        H = 1 / np.sqrt(d) * hadamard(d, dtype=np.float64)
        diag_entries = 2 * np.random.randint(2, size=d) - 1
        D = np.diag(diag_entries)
        sel_idx = np.random.choice(d, size=k, replace=False)
        E = np.zeros((k, d))
        E[np.arange(k), sel_idx] = 1
        G = E @ H @ D
    elif matrix_type.lower() == 'red':
        # random dedamacher matrix
        G = 2 * np.random.randint(2, size=(k, d)) - 1
        G = G / np.sqrt(d)
    elif matrix_type.lower() == 'gaussian':
        G = np.random.randn(k, d)
        if 'normalize' in kwargs and kwargs['normalize']:   # normalize each row
            print('Gaussian row normalized')
            G /= np.linalg.norm(G, axis=1)[:, None]  # normalize each row of G
    elif matrix_type.lower() == 'ldpc':
        s = kwargs['s'] if 's' in kwargs else 1   # set num of ones to 1 by default
        G = np.zeros((k, d))
        for i in range(k):
            sel_indices = np.random.choice(d, size=s, replace=False)
            G[i, sel_indices] = 1
        if 'normalize' in kwargs and kwargs['normalize']:
            G /= 1 / np.sqrt(s)
    elif matrix_type == 'simple':
        # this one corresponds to rand-k / spatial max's random matrix, for testing purpose only
        G = np.zeros((k, d))
        sel_indices = np.random.choice(d, size=k, replace=False)
        G[np.arange(k), sel_indices] = 1
    elif matrix_type == 'sparse_red':
        G = np.zeros((k, d))
        # target_num = 2
        s = kwargs['s'] if 's' in kwargs else 1  # set num of ones to 1 by default
        for i in range(k):
            sel_idx = np.random.choice(d, size=s, replace=False)
            val = np.random.randint(0, 2, size=s) * 2 - 1
            G[i, sel_idx] = val
        G = G / np.sqrt(s)
    else:
        raise Exception('Unsupported matrix type {}'.format(matrix_type))
    return G


def generate_testing_vecs(n, d, vec_type='ortho', **kwargs):
    if vec_type == 'ortho':
        X = np.zeros((n, d))
        X[np.arange(n), np.arange(n)] = 1
    elif vec_type == 'all_ones':
        X = np.ones((n, d))
        X = X / np.sqrt(d)
    else:
        raise Exception('Unknown vector type {}!'.format(vec_type))
    return X


def generate_corr_vecs(n, d, R):
    if n == 21:
        if R == 0:  # no corr
            X = generate_testing_vecs(n, d, 'ortho')
        elif R == 4:  # need 2n = 42 corr
            X = np.zeros((n, d))
            X[:9, 0] = 1  # 9 choose 2 = 36 corr
            X[9:13, 1] = 1  # 4 choose 2 = 6 corr
            for i in range(13, n):
                X[i, i] = 1
        elif R == 8:  # need 4n = 84 corr
            X = np.zeros((n, d))
            X[:13, 0] = 1  # 13 choose 2 = 78 corr
            X[13:17, 1] = 1  # 4 choose 2 = 6 corr
            for i in range(17, n):
                X[i, i] = 1
        elif R == 12:  # need 6n = 126 corr
            X = np.zeros((n, d))
            X[:16, 0] = 1  # 16 choose 2 = 120 corr
            X[16:20, 1] = 1  # 4 choose 2 = 6 corr
            X[20, 20] = 1
        elif R == 16:  # need 8n = 168 corr
            X = np.zeros((n, d))
            X[:19, 0] = 1  # 18 choose 2 = 153 corr
            for i in range(19, n):
                X[i, i] = 1
        elif R == 20:
            X = np.zeros((n, d))
            X[:, 0] = 1
        else:
            raise Exception('Invalid R {} for n = 21'.format(R))
    elif n == 51:
        if R == 10:  # need 5 x 51 = 255 corr
            X = np.zeros((n, d))
            X[:23, 0] = 1  # 23 choose 2 = 253 corr
            X[23, 1] = 1
            X[24, 1] = 1
            X[25, 2] = 1
            X[26, 2] = 1
            for i in range(27, n):
                X[i, i] = 1
        elif R == 20:  # need 10 x 51 = 510 corr
            X = np.zeros((n, d))
            X[:32, 0] = 1  # 32 choose 2 = 496 corr
            X[32:36, 1] = 1  # 4 choose 2 = 6 corr
            X[36:40, 2] = 1  # 4 choose 2 = 6 corr
            X[40:42, 3] = 1  # 2 choose 2 = 1
            X[42:44, 4] = 1  # 2 choose 2 = 1
            for i in range(44, n):
                X[i, i] = 1
        elif R == 30:  # need 15 x 51 = 765 corr
            X = np.zeros((n, d))
            X[:39, 0] = 1  # 39 choose 2 = 741 corr
            X[39:46, 1] = 1  # 7 choose 2 = 21 corr
            X[46:49, 2] = 1  # 3 choose 2 = 3 corr
            for i in range(49, n):
                X[i, i] = 1
        elif R == 40:  # need 20 x 51 = 1020 corr
            X = np.zeros((n, d))
            X[:46, 1] = 1   # 46 choose 2 = 1035 corr
            for i in range(46, n):
                X[i, i] = 1
        else:
            raise Exception('Invalid R {} for n = 51'.format(R))
    elif n == 5:
        if R == 1:
            X = np.zeros((n, d))
            X[0, 0] = 0.5
            X[0, 1:4] = 0.5
            X[1, 0] = 1
            X[2, 1] = 1
            X[3, 1] = 1
            X[4, 4] = 1
        elif R == 2:
            X = np.zeros((n, d))
            X[:3, 0] = 1
            X[3, 1] = 1
            X[4, 0:4] = 0.5
        elif R == 3:
            X = np.zeros((n, d))
            X[:4, 0] = 1
            X[4, 0] = 0.375
            X[4, 1] = np.sqrt(1 - 0.375 ** 2)
        elif R == 4:
            X = np.zeros((n, d))
            X[:, 0] = 1
        else:
            raise Exception('Invalid R {} for n = 5'.format(R))
    elif n == 11:
        if R == 2:
            X = np.zeros((n, d))
            X[0:5, 0] = 1
            X[5:7, 1] = 1
            for i in range(7, n):
                X[i, i] = 1
        elif R == 4:
            X = np.zeros((n, d))
            X[0:7, 0] = 1
            X[7:9, 1] = 1
            for i in range(9, n):
                X[i, i] = 1
        elif R == 6:
            X = np.zeros((n, d))
            X[:8, 0] = 1
            X[8, 0:4] = 0.5
            X[9:11, 5] = 1
        elif R == 8:
            X = np.zeros((n, d))
            X[:9, 0] = 1
            X[9, 0] = 8 / 9
            X[9, 1] = np.sqrt(1 - (8 / 9) ** 2)
            X[10, 5] = 1
        else:
            raise Exception('Invalid R {} for n = 11'.format(R))
    else:
        raise Exception('# clients {} not supported! '.format(n))
    return X

