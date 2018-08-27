import numpy as np
from tqdm import tqdm

def generate_data(n_data, data_dim, noise_std=0.1):
    """Generate some random data for Bayesian Linear Regression"""
    w = np.random.randn(data_dim + 1)
    x = np.ones((n_data, data_dim + 1))
    x[:, 1:] = np.random.randn(n_data, data_dim)
    y = np.dot(x, w) + np.random.normal(0, noise_std, size=n_data)
    return (x, y), w


def estimate_gibbs(X, y, tau_0=None, mu_0=None, a_0=1, b_0=1, n_iter=100):
    """Create a Gibbs sampler

    Args:
        X ((n,p) array): the design matrix
        y (n array): the outputs
        tau_0 (p array): prior precision for w
        mean_0 (p ndarray): prior mean for w
        a_0 (float): parameter for Gamma prior on tau
        b_0 (float): paramter for Gamma prior on tau
    """
    n, p = X.shape
    tau_0 = tau_0 or np.ones(p)
    mu_0 = mu_0 or np.zeros(p)

    XtX = np.matmul(X.transpose(), X)
    yX = np.matmul(y, X)

    def sample_w(i, w, tau):
        loc = tau_0[i]*mu_0[i] + yX[i] - np.sum([XtX[i,j]*w[j] for j in range(p) if j != i])
        loc /= (tau_0[i] + XtX[i,i])
        precision = tau * (tau_0[i] + XtX[i,i])
        return np.random.normal(loc, 1/np.sqrt(precision))

    def sample_tau(w):
        a_new = a_0 + n/2
        r_1 = y - np.matmul(X, w)
        r_2 = mu_0 - w
        b_new = b_0 + np.dot(r_1, r_1)/2 + np.dot(r_2, r_2) / 2
        return np.random.gamma(a_new, 1 / b_new)

    trace = []
    w = np.zeros(p)
    tau = 2
    trace.append((w.copy(), tau))
    for _ in tqdm(range(n_iter)):
        w = [sample_w(i, w, tau) for i in range(p)]
        tau = sample_tau(w)
        trace.append((w.copy(), tau))
    return trace
