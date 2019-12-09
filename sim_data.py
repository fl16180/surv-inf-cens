import numpy as np
from scipy.stats import multivariate_normal as mvn


def gen_lognormal(N, mu, sigma):
    Z = np.random.normal(mu, sigma, N)
    return np.exp(Z)


def gen_uniform(N, low, high):
    return np.random.uniform(low, high, size=N)


def gen_joint_lognormal(N, mu_s, mu_c, var_s, var_c, cov):
    mu = [mu_s, mu_c]
    S = [[var_s, cov], [cov, var_c]]

    R = mvn(mean=mu, cov=S)
    raw = np.array([R.rvs() for _ in range(N)])
    raw = np.exp(raw)
    S, C = raw[:, 0], raw[:, 1]
    return S, C


def survival_table(surv, cens=None):
    """ generates numpy array with columns 'Event' and 'Time'

    Input:
        surv: array of survival times
        cens: array of censoring times
    """
    if cens is None:
        cens = np.full_like(surv, np.inf)

    raw = np.array((surv, cens)).T
    T = np.min(raw, axis=1)
    E = raw[:, 0] < raw[:, 1]

    col_event = 'Event'
    col_time = 'Time'
    Y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)],
                 shape=T.shape[0])
    Y[col_event] = E
    Y[col_time] = T
    return Y



