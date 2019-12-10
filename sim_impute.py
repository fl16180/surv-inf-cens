import numpy as np
import pandas as pd
from synth_data import SynthCovariateData


N = 5000
n_features = 5
n_informative = 3
surv_dist = 'lognormal'
cens_dist = 'lognormal'

tau = 2


def impute_data():

    synth = SynthCovariateData(N, n_features, n_informative, surv_dist, cens_dist)
    treat = synth.make_linear(tau, bias_Y=0.5, bias_C=0.5, sigma_Y=1.0, sigma_C=0.8)
    control = synth.make_linear(0, bias_Y=0.5, bias_C=0.5, sigma_Y=1.0, sigma_C=0.8)


