import argparse
import numpy as np
import pandas as pd
from synth_data import SynthCovariateData
from lifelines import KaplanMeierFitter
from ngboost.ngboost import NGBoost
from ngboost.distns import LogNormal, MultivariateNormal
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.scores import MLE, CRPS
from lifelines.statistics import logrank_test
from scipy.stats import lognorm


str_to_estimator = {'lognorm': LogNormal, 'mvnorm': MultivariateNormal}

N = 1000
N_FEAT = 5
N_INFO = 3
BIAS_Y=0.5
BIAS_C=0.5
SIGMA_Y=0.1
SIGMA_C=0.1
LEARNER = 'tree'


# ----- survival tools ----- #
def logrank(Y1, Y2):
    results = logrank_test(durations_A=Y1, durations_B=Y2)
    return results.p_value


def KM(Y):
    kmf = KaplanMeierFitter()

    kmf.fit(Y['Time'], event_observed=Y['Event'])

    kmf.survival_function_.plot()
    plt.title('Survival function of synthetic censored data');
    return kmf


def cond_expectation(estimator, mus, sigmas, min_vals):
    # currently hardcoded for lognormal
    def cond_expect_single(mu, sigma, min_val):
        v = lognorm(s=sigma, scale=np.exp(mu))
        E1XgtY = v.expect(lambda x: x, lb=min_val, ub=np.inf)
        PXgtY = v.expect(lambda x: 1, lb=min_val, ub=np.inf)
        return E1XgtY / PXgtY
    return np.array([cond_expect_single(*x) for x in zip(mus, sigmas, min_vals)])


# ----- simulation functions ----- #
def ngb_impute(estimator, X, Y):
    base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
    }

    ngb = NGBoost(Dist=estimator,
          n_estimators=200,
          learning_rate=.05,
          natural_gradient=True,
          verbose=False,
          minibatch_frac=1.0,
          Base=base_name_to_learner[LEARNER],
          Score=MLE)

    train = ngb.fit(X, Y)
    Y_imputed = np.copy(Y)

    cens_mask = (Y['Event'] == 0)
    min_vals = Y['Time'][cens_mask]
    pred_dists = train.pred_dist(X[cens_mask])

    # mus = pred_dists.loc
    # sigmas = pred_dists.scale
    # preds = cond_expectation(estimator, mus, sigmas, min_vals)

    # print(np.sum(cens_mask))
    # print(min_vals.shape, preds.shape)
    # print(min_vals)
    # print(preds)

    # print(min_vals[:10])
    # print(np.exp(pred_dists.loc)[:10])
    # print(pred_dists.mean()[:10])

    Y_imputed['Time'][cens_mask] = np.exp(pred_dists.loc)
    return Y_imputed


def compute_survival_pvals(estimator, distn, tau, rho, obs_conf=False):
    synth = SynthCovariateData(N, n_features=N_FEAT, n_informative=N_INFO,
                               observe_confounding=obs_conf,
                               surv_dist=distn, cens_dist=distn)

    treat_X, treat_obsY, Y_true, _  = synth.make_linear(tau=tau,
                                                   bias_Y=BIAS_Y,
                                                   bias_C=BIAS_C,
                                                   sigma_Y=SIGMA_Y,
                                                   sigma_C=SIGMA_C,
                                                   rho=rho)
    control_X, control_obsY, _, _ = synth.make_linear(tau=0,
                                                      bias_Y=BIAS_Y,
                                                      bias_C=BIAS_C,
                                                      sigma_Y=SIGMA_Y,
                                                      sigma_C=SIGMA_C,
                                                      rho=rho)

    treat_Y_imputed = ngb_impute(estimator, treat_X, treat_obsY)
    control_Y_imputed = ngb_impute(estimator, control_X, control_obsY)

    p_val = logrank(treat_Y_imputed['Time'], control_Y_imputed['Time'])

    print(p_val)
    return p_val


def run_sim(N_iter, estimator, distn, tau, rho, obs_conf, seed):
    np.random.seed(seed)

    p_values = np.zeros(N_iter)
    for i in range(N_iter):
        p_values[i] = compute_survival_pvals(estimator,
                                             distn,
                                             tau=tau,
                                             rho=rho,
                                             obs_conf=obs_conf)
    return p_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--rho', type=float, default=0.0)
    parser.add_argument('--est', type=str, default='lognorm', choices=['lognorm', 'mvnorm'])
    parser.add_argument('--dist', type=str, default='lognormal')
    parser.add_argument('--obs_conf', type=bool, default=False)
    parser.add_argument('--num_sim', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()

    pvals = run_sim(args.num_sim, str_to_estimator[args.est], args.dist,
                    args.tau, args.rho, args.obs_conf, args.seed)

    infos = {'tau': args.tau, 'rho': args.rho, 'est': args.est,
             'dist': args.dist,
             'obs_conf': args.obs_conf, 'pvals': list(pvals)}

    name = f'run_{args.tau}_{args.rho}_{args.est}_{args.dist}_{args.obs_conf}'

    results = pd.DataFrame(infos)
    results.to_csv(f'./results/{name}.csv', index=False)
