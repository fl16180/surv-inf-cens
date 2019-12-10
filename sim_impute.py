import numpy as np
import pandas as pd
from synth_data import SynthCovariateData
from lifelines import KaplanMeierFitter
from ngboost.distns import LogNormal, Exponential, MultivariateNormal
from ngboost.ngboost import NGBoost
from lifelines.statistics import logrank_test

N = 5000
n_features = 5
n_informative = 3
surv_dist = 'lognormal'
cens_dist = 'lognormal'

tau = 2

def logrank(Y1, Y2):
    results = logrank_test(durations_A=Y1, durations_B=Y2)
                       #event_observed_A=E[~dem], event_observed_B=E[dem], alpha=.99)

    return results.p_value

def ngb_impute(distribution, X, Y):
    ngb = NGBoost(Dist=distribution,
          n_estimators=200,
          learning_rate=.05,
          natural_gradient=True,
          verbose=True,
          minibatch_frac=1.0,
          Base=base_name_to_learner["tree"],
          Score=MLE)
    
    train = ngb.fit(X, Y)
    Y_imputed = np.copy(Y)
    Y_imputed['Time'][Y['Event']] = np.exp(train.predict(X[Y['Event']]))
    return Y_imputed

def impute_data(distribution):

    synth = SynthCovariateData(N, n_features, n_informative, surv_dist, cens_dist)
    treat_X, treat_obsY, treat_Y, treat_C  = synth.make_linear(tau, bias_Y=0.5, bias_C=0.5, sigma_Y=1.0, sigma_C=0.8)
    control_X, control_obsY, control_Y, control_C = synth.make_linear(0, bias_Y=0.5, bias_C=0.5, sigma_Y=1.0, sigma_C=0.8)
    
    treat_Y_imputed = ngb_impute(distribution, treat_X, treat_obsY)
    control_Y_imputed = ngb_impute(distribution, control_X, control_obsY)
    
    p_val = logrank(treat_Y_imputed['Time'], control_Y_imputed['Time'])
    

def KM(Y, observed_censoring=True):
    kmf = KaplanMeierFitter()

    if observed_censoring:
        kmf.fit(Y['Time'], event_observed=Y['Event'])
    else:
        kmf.fit(Y)
        
    return kmf
