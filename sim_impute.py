import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synth_data import SynthCovariateData
from lifelines import KaplanMeierFitter
from ngboost.distns import LogNormal, Exponential, MultivariateNormal
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.scores import MLE, CRPS
from lifelines.statistics import logrank_test
import argparse

N = 1000
n_features = 5
n_informative = 3
surv_dist = 'lognormal'
cens_dist = 'lognormal'

def logrank(Y1, Y2):
    results = logrank_test(durations_A=Y1, durations_B=Y2)

    return results.p_value

def ngb_impute(distribution, X, Y):
    base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
    }
    
    ngb = NGBoost(Dist=distribution,
          n_estimators=200,
          learning_rate=.05,
          natural_gradient=True,
          verbose=False,
          minibatch_frac=1.0,
          Base=base_name_to_learner["tree"],
          Score=MLE)
    
    train = ngb.fit(X, Y)
    Y_imputed = np.copy(Y)
    Y_imputed['Time'][Y['Event']] = np.exp(train.predict(X[Y['Event']]))[:len(Y_imputed['Time'][Y['Event']])]
    return Y_imputed

def impute_data(distribution, tau=0, rho=0.0, obs_conf=False):

    synth = SynthCovariateData(N, n_features, n_informative, 
                               observe_confounding=obs_conf, 
                               surv_dist='lognormal', cens_dist='lognormal')
    treat_X, treat_obsY, treat_Y, treat_C  = synth.make_linear(tau=tau, 
                                                               bias_Y=0.5, 
                                                               bias_C=0.5, 
                                                               sigma_Y=0.1, 
                                                               sigma_C=0.1, 
                                                               rho=rho)
    control_X, control_obsY, control_Y, control_C = synth.make_linear(tau=0, 
                                                                      bias_Y=0.5, 
                                                                      bias_C=0.5, 
                                                                      sigma_Y=0.1, 
                                                                      sigma_C=0.1, 
                                                                      rho=rho)
    
    treat_Y_imputed = ngb_impute(distribution, treat_X, treat_obsY)
    control_Y_imputed = ngb_impute(distribution, control_X, control_obsY)
    
    p_val = logrank(treat_Y_imputed['Time'], control_Y_imputed['Time'])
    
    print(p_val)
    return p_val
    

def KM(Y, observed_censoring=True):
    kmf = KaplanMeierFitter()

    if observed_censoring:
        kmf.fit(Y['Time'], event_observed=Y['Event'])
    else:
        kmf.fit(Y)
        
    kmf.survival_function_.plot()
    plt.title('Survival function of synthetic censored data');
        
    return kmf

def run_sim(N, distribution, tau, rho, obs_conf):
    p_values = np.zeros(N)
    for i in range(N):
        p_values[i] = impute_data(distribution, tau=tau, rho=rho, obs_conf=obs_conf)
        
    return p_values
