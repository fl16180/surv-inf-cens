from ngboost.distns import MultivariateNormal
from ngboost.distns import LogNormal
from lognormal import LogNormal
from ngboost.scores import MLE
import numpy as np
from scipy.stats import multivariate_normal as dist


def lognormal_mle(Y, max_iter=1e4, lr=0.05, eps=1e-4, verbose=False):
    N = Y.shape[0]
    params = np.array([[0, 0]] * N).T
    for i in range(int(max_iter)):
        if i % 500 == 1 and verbose:
            print('Param: ', params[:, :2])
            print('Grad: ', grad)
        D = LogNormal(params)
        S = MLE()

        grad = np.mean(S.grad(D, Y, natural=True).T, axis=1, keepdims=True)
        params = params - lr * grad
        if np.linalg.norm(grad) < eps:
            break

    mu = params[0, 0]
    sigma = params[1, 0]
    return mu, sigma


def mvnorm_mle(Y, max_iter=1e4, lr=0.5, eps=1e-4):
    N = Y.shape[0]
    params = np.array([[0, 0, 1, 0, 1]] * N).T
    for _ in range(max_iter):
        D = MultivariateNormal(params)
        S = MLE()
        grad = np.mean(S.grad(D, Y, natural=True).T, axis=1, keepdims=True)
        params = params - lr * grad
        if np.linalg.norm(grad) < eps:
            break

