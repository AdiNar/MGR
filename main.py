#!/usr/bin/env python3

import random
from time import time
import argparse

from algorithm import ApAlg, ApAlgS, ApAlgH
from distribution import Dist
from list_scheduler import LPT, LRR, HRR, RAND
from simulator import SimulationRunner


def run(seed):
    print(f'seed: {seed}')
    random.seed(seed)
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

    ns = [500, 1000, 5000, 10000]
    ms = [10, 20, 50, 100]
    params = [(n, m) for n in ns for m in ms]

    SimulationRunner(algorithms=algorithms, simulation_input=simulation_input, params=params, reps=30).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    default_seed = int(time())
    parser.add_argument('--seed', type=int, help='Seed used to generate test instances. Defaults to current time.', default=default_seed)
    args = parser.parse_args()

    run(args.seed)
