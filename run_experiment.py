import os
import argparse
import random
import string
from subprocess import check_call


RHOS = [-0.4, -0.2, 0, 0.2, 0.4]
TAUS = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
ESTS = ['lognorm', 'mvnorm']
DISTS = ['lognormal']
CONFOUNDING = [False]
N_SIM = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
                    '--spawn_type',
                    type=str,
                    choices=('stdout', 'shell', 'sbatch'),
                    default='stdout',
                    help='Whether to print commands to stdout (stdout), spawn\
                          them on the current machine (shell), or spawn\
                          them using sbatch (sbatch).')
    args = parser.parse_args()
    spawn_type = args.spawn_type

    commands = []
    for tau in TAUS:
        for rho in RHOS:
            for est in ESTS:
                for dist in DISTS:
                    for obs_conf in CONFOUNDING:
                        command = 'python sim_impute.py'
                        command += f' --tau {tau}'
                        command += f' --rho {rho}'
                        command += f' --est {est}'
                        command += f' --dist {dist}'
                        command += f' --obs_conf {obs_conf}'
                        command += f' --num_sim {N_SIM}'

                        name = f'run_{tau}_{rho}_{est}_{dist}_{obs_conf}'
                        commands.append((name, command))


    if args.spawn_type == 'stdout':
        # Print commands to stdout.
        for _, command in commands:
            print(command)

    elif args.spawn_type == 'shell':
        # Run commands sequentially on this machine.
        commands_str = ";".join([com for _, com in commands])
        print(commands_str)
        check_call(commands_str, shell=True)

    else:
        # Output the commands to a sbatch script then call sbatch.
        with open("./scripts/sbatch_template.sh") as f:
            template = f.read()

        for name, command in commands:
            sbatch = template.replace("COMMAND", command)
            sbatch = sbatch.replace("NAME", name)

            with open("spawn_tmp.sh", 'w') as f:
                f.write(sbatch)

            print(command)
            check_call("sbatch spawn_tmp.sh", shell=True)

        os.remove("spawn_tmp.sh")
