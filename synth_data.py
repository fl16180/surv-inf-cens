import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.datasets import make_regression
from numpy.random import lognormal, weibull
from scipy.stats import pearsonr


def corr(x, y):
    return pearsonr(x, y)[0]

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


# ----- regression estimation data generation ----- #
def sample_lognormal(mu, sigma):
    return np.exp(np.random.normal(mu, sigma))


def sample_weibull(scale, shape):
    return scale * np.random.weibull(shape)


def sample_exponential(mu, sigma):
    ln_mean = np.exp(mu + sigma ** 2 / 2)
    return np.random.exponential(ln_mean)


class SynthCovariateData:
    def __init__(self, N, n_features, n_informative, observe_confounding=False,
                 surv_dist='lognormal', cens_dist='lognormal'):
        self.N = N
        self.n_features = n_features
        self.n_informative = n_informative
        self.U_obs = observe_confounding
        self.surv_dist = globals()[f'sample_{surv_dist}']
        self.cens_dist = globals()[f'sample_{cens_dist}']

    def make_linear(self, tau, bias_Y, bias_C, sigma_Y, sigma_C, rho):

        # generate Y~X as linear model
        X, Y = make_regression(n_samples=self.N, n_features=self.n_features,
                               n_informative=self.n_informative, noise=0)
        Y = np.divide(Y - np.mean(Y), np.std(Y))

        # generate C either as C~Y+X or C~Y+U
        if self.U_obs:
            C = self.gen_C_observed(Y, rho, X)
        else:
            C = self.gen_C_unobserved(Y, rho)

        # print(corr(X[:, 0], Y))
        # print(corr(X[:, 0], C))
        # print(corr(X[:, 1], Y))
        # print(corr(X[:, 1], C))
        # print(corr(Y, C))

        # add offsets and treatment effect
        Y += bias_Y + tau
        C += bias_C

        # add noise following specified distributions
        Y_true = [self.surv_dist(x, sigma_Y) for x in Y]
        C_true = [self.cens_dist(x, sigma_C) for x in C]

        Y_obs = survival_table(Y_true, C_true)
        return X, Y_obs, Y_true, C_true

    def gen_C_unobserved(self, Y, rho):
        U = np.random.normal(np.mean(Y), np.std(Y), len(Y))
        C = rho * Y + np.sqrt((1 - rho ** 2)) * U
        C = np.divide(C - np.mean(C), np.std(C))
        return C

    def gen_C_observed(self, Y, rho, X):
        beta = np.random.uniform(-2, 2, size=X.shape[1])
        Xbeta = X @ beta[:, None]

        U = np.random.normal(np.mean(Y), np.std(Y), len(Y))
        C = rho * Xbeta.flatten() + np.sqrt((1 - rho ** 2)) * U
        C = np.divide(C - np.mean(C), np.std(C))
        return C
