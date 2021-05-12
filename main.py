#!/usr/bin/env python3

import random
from time import time
import argparse

from algorithm import ApAlg, ApAlgS, ApAlgH
from distribution import Dist
from list_scheduler import LPT, LRR, HRR, RAND
from simulator import SimulationRunner


def run(args):
    print(f'seed: {args.seed}')
    random.seed(args.seed)
    algorithms = [
        (ApAlg, 'ApAlg'),
        (ApAlgS, 'ApAlg-S'),
        (ApAlgH, 'ApAlg-H'),
        (LPT, 'LPT'),
        (LRR, 'LRR'),
        (HRR, 'HRR'),
        (RAND, 'RAND')
    ]

    simulation_input = Dist('logs_zapat')

    params = [(n, m) for n in args.jobs for m in args.machines]

    SimulationRunner(algorithms=algorithms, simulation_input=simulation_input, params=params, reps=args.reps,
                     output_prefix=args.output_prefix).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    default_seed = int(time())
    parser.add_argument('--seed', type=int, help='Seed used to generate test instances. Defaults to current time.',
                        default=default_seed)
    parser.add_argument('--reps', type=int,
                        help='Number of instances per <number of machines, number of jobs> pair every algorithm runs on.',
                        default=30)
    parser.add_argument('--machines', type=int, nargs='+',
                        help='Space separated list of machines number to test each algorithm on.',
                        default=[10, 20, 50, 100])
    parser.add_argument('--jobs', type=int, nargs='+',
                        help='Space separated list of jobs number to test each algorithm on.',
                        default=[500, 1000, 5000, 10000])
    parser.add_argument('--output-prefix', type=str,
                        help='Prefix of the output files. Script will generate the following files:'
                             '<output_prefix>_{approx,runtime}.{tex,pdf}', default='results')

    args = parser.parse_args()

    run(args)
