import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.datasets import make_regression
from np.random import lognormal, weibull


# ----- parameter estimation data generation ----- #
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


# ----- regression estimation data generation ----- #
def sample_lognormal(mu, sigma):
    return np.exp(np.random.normal(mu, sigma, 1))


class SynthCovariateData:
    def __init__(self, N, n_features, n_informative,
                 surv_dist='lognormal', cens_dist='lognormal'):
        self.N = N
        self.n_features = n_features
        self.n_informative = n_informative
        self.surv_dist = locals()[f'sample_{surv_dist}']
        self.cens_dist = locals()[f'sample_{cens_dist}']

    def make_linear(self, tau, bias_Y, bias_C, sigma_Y, sigma_C, rho):

        X, Y = make_regression(n_samples=self.N, n_features=self.n_features,
                               n_informative=self.n_informative, noise=0)
        Y = np.divide(Y - np.mean(Y), np.std(Y))
        C = self.gen_corr_C(Y, rho)

        Y += bias_Y + tau
        C += bias_C

        Y_samp = [self.surv_dist(N=1, x, sigma_Y) for x in Y]
        C_samp = [self.cens_dist(N=1, x, sigma_C) for x in C]

        Y_obs = survival_table(Y_samp, C_samp)
        return X, Y_obs, Y_samp, C_samp

    def gen_C_unobserved(self, Y, rho):
        U = np.random.normal(np.mean(Y), np.std(Y), len(Y))
        C = rho * Y + np.sqrt((1 - rho ** 2)) * U
        C = np.divide(C - np.mean(C), np.std(C))
        return C

    def gen_C_observed(self, Y, rho, X):
        beta = np.random.uniform(-2, 2, size=X.shape[1])
        Xbeta = X @ beta[:, None]
        C = rho * Y + np.sqrt((1 - rho ** 2)) * Xbeta
        C = np.divide(C - np.mean(C), np.std(C))
        return C


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



