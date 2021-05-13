#!/usr/bin/env python3

import random
from time import time
import argparse

from algorithm import ApAlg, ApAlgS, ApAlgH
from distribution import Dist
from list_scheduler import LPT, LRR, HRR, RAND
from simulator import SimulationRunner

algorithms_dict = dict([
        ('ApAlg', ApAlg),
        ('ApAlg-S', ApAlgS),
        ('ApAlg-H', ApAlgH),
        ('LPT', LPT),
        ('LRR', LRR),
        ('HRR', HRR),
        ('RAND', RAND)
])


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f'Positive value expected, got {value}')
    return ivalue


def algorithm_name(value):
    if value not in algorithms_dict.keys():
        print(f'Invalid algorithm name: {value}. Must be one of {list(algorithms_dict.keys())}')
    return value


def run(args):
    print(f'seed: {args.seed}')
    random.seed(args.seed)

    algorithms = list(map(lambda x: (algorithms_dict[x], x), args.algorithms))

    simulation_input = Dist('logs_zapat')

    params = [(n, m) for n in args.jobs for m in args.machines]

    SimulationRunner(algorithms=algorithms, simulation_input=simulation_input, params=params, reps=args.reps,
                     output_dir=args.output_dir).run(args.check_assertions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for comparison between ApAlg variations and '
                                                 'benchmark consisting of list schedulers. '
                                                 'Produces boxplot chart with makespans normalized by known '
                                                 'lower bounds for optimal schedule length. ')

    default_seed = int(time())
    parser.add_argument('--seed', type=positive_int, help='Seed used to generate test instances. '
                                                          'Defaults to a value derived from the current time.',
                        default=default_seed)
    parser.add_argument('--reps', type=positive_int,
                        help='Number of instances per <number of machines, number of jobs> '
                             'pair every algorithm runs on. Defaults to %(default)s.',
                        default=30)
    parser.add_argument('--machines', type=positive_int, nargs='+',
                        help='Space separated list of machines number to test each algorithm on. '
                             'Defaults to %(default)s.',
                        default=[10, 20, 50, 100])
    parser.add_argument('--jobs', type=positive_int, nargs='+',
                        help='Space separated list of jobs number to test each algorithm on. Defaults to %(default)s.',
                        default=[500, 1000, 5000, 10000])
    parser.add_argument('--output-dir', type=str,
                        help='Directory to put the output files. Script will generate the following files: '
                             '<output-dir>/{approx,runtime})_{{jobs}_{machines}}.{tex,pdf}. Defaults to "%(default)s".',
                        default='results')
    parser.add_argument('--check-assertions', action='store_true', help='Check assertions in ApAlg variations.')
    parser.add_argument('--algorithms', type=algorithm_name, nargs='+',
                        help='Choose which algorithms to compare in a space separated list. '
                             'Possible values are: %(default)s. '
                             'By default all algorithms are compared.', default=list(algorithms_dict.keys()))

    args = parser.parse_args()

    run(args)
