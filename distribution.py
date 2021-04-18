import random


class SimulationInput:
    def prepare(self, n, m=0):
        pass

    def get_next(self):
        pass

    def __call__(self):
        nxt = 0
        while not nxt:
            nxt = self.get_next()
        return nxt


class Dist(SimulationInput):
    def __init__(self, filename):
        self.pairs = self.parse_pairs(filename)

        outliers = len(self.pairs) // 100
        to_delete_p = self.get_processing_time_outliers(outliers)
        to_delete_r = self.get_resource_outliers(outliers)

        to_delete = to_delete_r.union(to_delete_p)
        self.pairs = self.remove_outliers(to_delete)

        self.normalize_resource()

    def parse_pairs(self, filename):
        raw_lines = open(filename).read().split('\n')
        raw_pairs = [line.split(' ') for line in raw_lines]
        raw_pairs = [p for p in raw_pairs if len(p) == 2]

        p_list = [int(rp[0]) for rp in raw_pairs]
        r_list = [int(rp[1]) for rp in raw_pairs]
        return [(p, r) for p, r in zip(p_list, r_list) if p]

    def remove_outliers(self, to_delete):
        return [(p, r) for p, r in self.pairs if (p, r) not in to_delete]

    def get_processing_time_outliers(self, outliers):
        return set(sorted(self.pairs)[-outliers:])

    def get_resource_outliers(self, outliers):
        resource_first_pairs = [list(reversed(x)) for x in self.pairs]
        return set((x, y) for y, x in sorted(resource_first_pairs)[-outliers:])

    def normalize_resource(self):
        mem = max(r for p, r in self.pairs)
        self.pairs = [(p, r / mem) for p, r in self.pairs]

    def get_next(self):
        return random.sample(self.pairs, 1)[0]
