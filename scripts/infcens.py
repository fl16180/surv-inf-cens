from ngboost.distns import MultivariateNormal
from ngboost.distns import LogNormal
from ngboost.scores import MLE
import numpy as np
from scipy.stats import multivariate_normal as dist

N = 1000

mu0 = 2 # mean of E
mu1 = 1 # mean of C

S00 = 1
S11 = 0.8
S01 = S10 = 0.0
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


def Y_join(T, E):
    col_event = 'Event'
    col_time = 'Time'
    y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)],
                 shape=T.shape[0])
    y[col_event] = E
    y[col_time] = np.exp(T)
    return y

Y = Y_join(T, E)

params = np.array([[0, 0, 1, 0, 1]] * N).T
for _ in range(100000):
    D = MultivariateNormal(params)
    S = MLE()
    grad = np.mean(S.grad(D, Y, natural=True).T, axis=1, keepdims=True)
    params = params - 1 * grad
    if np.linalg.norm(grad) < 1e-4:
        break
    
print('Jointly Estimated E:', params[0, 0])

params = np.array([[0, 0]] * N).T
for _ in range(100000):
    D = LogNormal(params)
    S = MLE()
    grad = np.mean(S.grad(D, Y, natural=True).T, axis=1, keepdims=True)
    params = params - 0.1 * grad
    if np.linalg.norm(grad) < 1e-4:
        break
    
print('Estimate E (assume non-inf):', params[0, 0])

