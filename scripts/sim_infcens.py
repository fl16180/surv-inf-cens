from ngboost.distns import MultivariateNormal
from ngboost.distns import LogNormal
from ngboost.scores import MLE
import numpy as np
from scipy.stats import multivariate_normal as dist
import matplotlib.pyplot as plt
import pandas as pd


def Y_join(T, E):
    col_event = 'Event'
    col_time = 'Time'
    y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)],
                 shape=T.shape[0])
    y[col_event] = E
    y[col_time] = np.exp(T)
    return y


def generate_mv_surv(N, mu0, mu1, S00, S11, S01):
    mu = [mu0, mu1]
    S = [[S00, S01], [S10, S11]]

    R = dist(mean=mu, cov=S)
    raw = np.array([R.rvs() for _ in range(N)])
    T = np.min(raw, axis=1)
    E = raw[:, 0] < raw[:, 1]

    print('Prevalence:', np.mean(E))
    print('mu_E:', mu0)
    print('Marginal Mean of E:', np.mean( T[np.where(E == 1)] ))
    print('mu_C:', mu1)
    print('Marginal Mean of C:', np.mean( T[np.where(E == 0)] ))

    Y = Y_join(T, E)
    return Y


def estimate_infcens(Y):
    res = {}
    params = np.array([[0, 0, 1, 0, 1]] * N).T
    for _ in range(100000):
        D = MultivariateNormal(params)
        S = MLE()
        grad = np.mean(S.grad(D, Y, natural=True).T, axis=1, keepdims=True)
        params = params - 1 * grad
        if np.linalg.norm(grad) < 1e-4:
            break

    print('Jointly Estimated E:', params[0, 0])
    res['joint'] = params[0, 0]

    params = np.array([[0, 0]] * N).T
    for _ in range(100000):
        D = LogNormal(params)
        S = MLE()
        grad = np.mean(S.grad(D, Y, natural=True).T, axis=1, keepdims=True)
        params = params - 0.005 * grad
        if np.linalg.norm(grad) < 1e-4:
            break

    print('Estimate E (assume non-inf):', params[0, 0])
    res['lognorm'] = params[0, 0]
    return res


if __name__ == '__main__':

    logfile = './log.csv'
    try:
        results = pd.read_csv(logfile)
    except FileNotFoundError:
        results = pd.DataFrame(columns=['data', 'N', 'mu_E', 'mu_C', 'inf_par',
                                        'joint', 'lognorm'])

    # covs = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    # covs = [0, 0, 0, 0, 0, .2, .2, .2, .2, .2]
    covs = [-0.1] * 5 + [-0.3] * 5 + [0.1] * 5 + [0.3] * 5
    joint = np.zeros(len(covs))
    lognorm = np.zeros(len(covs))

    for i, cov in enumerate(covs):

        N = 1000
        mu0 = 2 # mean of E
        mu1 = 1 # mean of C

        S00 = 1
        S11 = 0.8
        S01 = S10 = cov

        Y = generate_mv_surv(N, mu0, mu1, S00, S11, S01)
        res = estimate_infcens(Y)

        out = {'data': 'mv', 'N': N, 'mu_E': mu0, 'mu_C': mu1,
               'inf_par': str([S00, S11, S01]),
               'joint': res['joint'], 'lognorm': res['lognorm']}
        results = pd.concat([results, pd.DataFrame(out, index=[0])], axis=0, sort=False)
        results.reset_index(drop=True).to_csv(logfile, index=False)

    #     joint[i] = res['joint']
    #     lognorm[i] = res['lognorm']

    # plt.plot(covs, joint, label='joint')
    # plt.plot(covs, single, label='lognormal')
    # plt.ylabel('est. E')
    # plt.xlabel('cov(E, C)')
    # plt.legend()
    # plt.show()



    mu = [mu0, mu1]
    S = [[S00, S01], [S10, S11]]

    R = dist(mean=mu, cov=S)
    raw = np.array([R.rvs() for _ in range(N)])
    T = np.min(raw, axis=1)
    E = raw[:, 0] < raw[:, 1]

    print('Prevalence:', np.mean(E))
    print('mu_E:', mu0)
    print('Marginal Mean of E:', np.mean( T[np.where(E == 1)] ))
    print('mu_C:', mu1)
    print('Marginal Mean of C:', np.mean( T[np.where(E == 0)] ))
