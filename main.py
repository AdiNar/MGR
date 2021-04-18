import random
from time import time

from algorithm import ApAlg, ApAlgS, ApAlgH
from distribution import Dist
from list_scheduler import LPT, LRR, HRR, RAND
from simulator import SimulationRunner


def run():
    seed = int(time())
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

    input = Dist('logs_zapat')

    ns = [500, 1000, 5000, 10000]
    ms = [10, 20, 50, 100]
    params = [(n, m) for n in ns for m in ms]

    SimulationRunner(algorithms=algorithms, simulation_input=input, params=params, reps=1,
               display=False).run()


if __name__ == '__main__':
    run()