import  numpy as np
import pytest

from main import generate_data, estimate_gibbs

def test_generate_data():
    (x, y), w = generate_data(10, 2)
    assert x.shape == (10, 3)
    assert y.shape == (10,)
    assert w.shape == (3,)


@pytest.mark.flaky(reruns=3)
def test_estimate_gibbs():
    tau = 1
    (X, y), w = generate_data(10000, 2, noise_std=1/np.sqrt(tau))
    trace = estimate_gibbs(X, y, a_0=1, b_0=1, n_iter=1000)
    burn_in = 800
    samples = np.array([_t[0] for _t in trace[burn_in:]])
    w_mean = np.mean(samples, axis=0)
    w_stddv = np.sqrt(np.var(samples, axis=0))
    tau_mean = np.mean([_t[1] for _t in trace[burn_in:]])
    tau_stddv = np.sqrt(np.var([_t[1] for _t in trace[burn_in:]]))

    r = 1.5
    for i in range(X.shape[1]):
        assert w_mean[i] - r*w_stddv[i] < w[i] < w_mean[i] + r*w_stddv[i]
    assert tau_mean - r*tau_stddv < tau < tau_mean + r*tau_stddv
