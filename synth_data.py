from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np


# simple
X, y, coef = make_regression(n_samples=500, n_features=5, n_informative=3,
                             noise=5, coef=True)



plt.scatter(X[:, 0], y)
plt.scatter(X[:, 1], y)
plt.scatter(X[:, 2], y)
plt.show()





# def generate from distribution

# run test

# compute power after iterations