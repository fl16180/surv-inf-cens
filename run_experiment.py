import argparse
from sim_impute import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--tau', type=float, default=0.0)
    args = parser.parse_args()

    args.tau
=======
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--num_sim', type=int, default=10)
    parser.add_argument('--distn', type=str, default="LogNormal")
    parser.add_argument('--rho', type=float, default=0.0)
    parser.add_argument('--obs_conf', type=bool, default=False)
    args = parser.parse_args()
    
    p = run_sim(args.num_sim, eval(args.distn), args.tau, args.rho, args.obs_conf)
    
    np.savetxt('{}.out'.format(args.distn), p, delimiter=',')
    
>>>>>>> run experiment and sim impute prep
