import numpy as np
from rand_k_spatial import compute_c1_c2_beta_bar
import os
import pickle
from general_decoding_srht import simulate_beta_bar_srht
from rand_matrix_vec import generate_random_matrix


class Encoder(object):

    def __init__(self, n, d, k):
        assert(k <= d)
        self.k = k
        self.d = d
        self.n = n

    def encode(self, x):
        assert len(x) == self.d
        hat_x = x
        return hat_x, np.eye(self.d)

    def decode(self, x_hat_list, G_list=None):
        V = np.vstack(x_hat_list)
        return np.mean(V, axis=0)

def rand_k_encoding(x, k):
    d = len(x)
    sel_idx = np.random.choice(np.arange(d), size=k, replace=False)
    G = np.zeros((k, d))
    G[np.arange(k), sel_idx] = 1
    x_hat = G @ x
    return x_hat, G


def cross_client_decoding(x_hat_list, G_list, beta_bar, T_fn):
    n = len(x_hat_list)
    Gs = np.vstack(G_list)
    S = Gs.T @ Gs
    eigvals, V = np.linalg.eig(S)
    eigvals = eigvals.real
    V = V.real
    S = V @ np.diag(T_fn(eigvals)) @ V.T
    S_pinv = np.linalg.pinv(S)
    encoded = np.sum([G_list[cidx].T @ x_hat_list[cidx] for cidx in range(n)], axis=0)
    x_hat = beta_bar * S_pinv @ encoded / n
    return x_hat


class RandkEncoder(Encoder):

    def encode(self, x):
        # x: (d, )
        assert (len(x) == self.d)
        return rand_k_encoding(x, self.k)

    def decode(self, x_hat_list, G_list=None):
        assert len(x_hat_list) == self.n
        x_hat = np.zeros(self.d)
        for client_idx in range(self.n):
            x_hat += G_list[client_idx].T @ x_hat_list[client_idx]
        x_hat = self.d / self.k * x_hat / self.n
        return x_hat


class RandkSpatialEncoder(Encoder):

    def __init__(self, n, d, k):
        super().__init__(n, d, k)
        self.T_fn = lambda m: 1 + n * (m - 1) / (n - 1) / 2
        self.beta_bar, _, _ = compute_c1_c2_beta_bar(n=n, dim=d, k=k, T_fn=self.T_fn)

    def encode(self, x):
        assert (len(x) == self.d)
        return rand_k_encoding(x, self.k)

    def decode(self, x_hat_list, G_list=None):
        assert len(x_hat_list) == self.n
        assert G_list is not None
        assert len(G_list) == self.n
        x_hat = cross_client_decoding(x_hat_list, G_list, self.beta_bar, self.T_fn)
        return x_hat


class RandProjSpatialEncoderSRHT(Encoder):

    def __init__(self, n, d, k):
        super().__init__(n, d, k)
        self.T_fn = lambda m: 1 + n * (m - 1) / (n - 1) / 2
        folder = 'beta_bar'
        path = os.path.join(folder, 'beta_bar_n_{}_k_{}_d_{}.pkl'.format(n, k, d))
        if os.path.isdir(folder) and os.path.isfile(path):
            with open(path, 'rb') as f:
                self.beta_bar = pickle.load(f)
        else:
            if not os.path.isdir(folder):
                os.mkdir(folder)
            num_exp = 100
            A_path = os.path.join(folder, 'A_n_{}_k_{}_d_{}.pkl'.format(n, k, d))
            dic = {'beta_bar_path': path, 'fn': self.T_fn, 'A_path': A_path}
            self.beta_bar = simulate_beta_bar_srht(n=n, k=k, d=d, num_exp=num_exp, **dic)

    def encode(self, x):
        assert (len(x) == self.d)
        G = generate_random_matrix(self.k, self.d, matrix_type='srht')
        x_hat = G @ x
        return x_hat, G

    def decode(self, x_hat_list, G_list=None):
        assert len(x_hat_list) == self.n
        assert G_list is not None
        assert len(G_list) == self.n
        x_hat = cross_client_decoding(x_hat_list, G_list, self.beta_bar, self.T_fn)
        return x_hat


class RandkWangni(Encoder):

    def __init__(self, n, d, k):
        super().__init__(n, d, k)
        self.rho = k / d

    def encode(self, x):
        assert (len(x) == self.d)
        abs_x = np.abs(x)
        if np.sum(abs_x) == 0:
            return x, None
        # compute probability of encoding
        prob = np.minimum(self.k * abs_x / np.sum(abs_x), np.ones(self.d))
        j = 0
        while True:
            # identify active set
            active_idx = np.where(prob < 1)[0]
            active_prob = prob[active_idx]
            # compute the scaling variable
            c = (self.k - self.d + len(active_idx)) / np.sum(active_prob)
            if c <= 1:
                break
            # recalibrate
            prob = np.minimum(c * prob, np.ones(self.d))
            j += 1
        # after getting prob vec, now can encode x
        x_hat = np.zeros(self.d)
        for coord_i in range(self.d):
            p_i = prob[coord_i]
            if p_i < 1e-5:  # p_i is treated as 0
                x_hat[coord_i] = 0
            else:
                x_hat[coord_i] = np.random.choice([0, x[coord_i]], p=[1-p_i, p_i]) / p_i
        return x_hat, None


    def decode(self, x_hat_list, G_list=None):
        assert len(x_hat_list) == self.n
        return np.mean(x_hat_list, axis=0)


class InducedTopkRandk(Encoder):

    def __init__(self, n, d, k):
        super().__init__(n, d, k)
        self.half_k = k // 2

    def encode(self, x):
        assert len(x) == self.d
        # encode using top-k first
        abs_x = np.abs(x)
        sorted_idx = np.argsort(abs_x)[::-1]  # indices sorted from largest to smallest
        x_top_k = np.zeros(self.d)
        top_k_idx = sorted_idx[:self.half_k]
        x_top_k[top_k_idx] = x[top_k_idx]
        # now encode using rand k
        x_remaining = x - x_top_k
        sel_idx = np.random.choice(self.d, size=self.half_k, replace=False)
        x_rand_k = np.zeros(self.d)
        x_rand_k[sel_idx] = self.d / self.half_k * x_remaining[sel_idx]
        x_hat = x_top_k + x_rand_k
        return x_hat, None

    def decode(self, x_hat_list, G_list=None):
        assert len(x_hat_list) == self.n
        return np.mean(x_hat_list, axis=0)


def test_encoder():
    n = 2
    d = 8
    k = 2
    # encoder = RandkEncoder(n, d, k)
    encoder = RandkSpatialEncoder(n, d, k)
    X = []
    for cidx in range(n):
        X.append(np.arange(d) * (cidx + 1))
    X = np.stack(X)
    print(X)
    x_hat_list = []
    G_list = []
    for cidx in range(n):
        x_hat, G_i = encoder.encode(X[cidx])
        x_hat_list.append(x_hat)
        G_list.append(G_i)
        print('client {}, encoded: {}'.format(cidx, x_hat))
        print(G_i)
    x_hat = encoder.decode(x_hat_list, G_list)
    print('est mean vec: ', x_hat)


def test_wangni_encoder():
    n = 2
    d = 8
    k = 4

    encoder = RandkWangni(n, d, k)
    X = []
    for cidx in range(n):
        X.append(np.arange(d) * (cidx + 1))
    X = np.stack(X)
    print(X)
    x_hat_list = []
    for cidx in range(n):
        x_hat, _ = encoder.encode(X[cidx])
        x_hat_list.append(x_hat)
    x_hat = encoder.decode(x_hat_list)
    print(x_hat)

def test_induced_top_k_rand_k_encoder():
    n = 2
    d = 8
    k = 4

    encoder = InducedTopkRandk(n, d, k)
    X = []
    for cidx in range(n):
        X.append(np.arange(1, d + 1) * (cidx + 1))
    X = np.stack(X)
    print(X)
    x_hat_list = []
    for cidx in range(n):
        x_hat, _ = encoder.encode(X[cidx])
        x_hat_list.append(x_hat)
    x_hat = encoder.decode(x_hat_list)
    print(x_hat)



if __name__ == '__main__':
    # test_encoder()
    # test_wangni_encoder()
    test_induced_top_k_rand_k_encoder()
