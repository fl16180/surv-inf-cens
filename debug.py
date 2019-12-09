import numpy as np
from estimators import *
from sim_data import *

import matplotlib.pyplot as plt


S = gen_lognormal(1000, 0.5, 0.25)
Y = survival_table(S, cens=None)
lognormal_mle(Y)


S = gen_lognormal(1000, 0.5, 0.25)
C = np.array([0.1] * 900 + [10000] * 100)
Y = survival_table(S, cens=C)
lognormal_mle(Y)



S, C = gen_joint_lognormal(1000, mu_s=2, mu_c=1, var_s=1, var_c=0.8, cov=0.4)
Y = survival_table(S, C)
show_joint(S,C)


lognormal_mle(Y)

global cout = False

def show_joint(S, C):
    plt.scatter(S, C)
    plt.plot([0, 60], [0, 60])
    plt.show()