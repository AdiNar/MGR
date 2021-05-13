import heapq
from decimal import Decimal
from enum import Enum, auto
from typing import List, Tuple, Iterable
from collections import defaultdict
from time import time

import numpy as np

from intervaltree import IntervalTree

from distribution import SimulationInput
from utils import print_latex


class JobState(Enum):
    NOT_SCHEDULED = auto()
    SCHEDULED = auto()
    DIRTY = auto()


class Job:
    def __init__(self, length, resource=0.0):
        self.p = length
        self.r = resource
        self.state = JobState.NOT_SCHEDULED
        self.scheduled_at = None
        self.machine = None
        self.schedule_obj = None

        assert length > 0
        assert 0 <= resource <= 1

    def schedule(self, t, m, schedule_obj):
        self.scheduled_at = t
        self.machine = m
        self.state = JobState.SCHEDULED
        self.schedule_obj = schedule_obj

    def unschedule(self):
        if self.schedule_obj:
            self.schedule_obj.unschedule(self)
        self.state = JobState.NOT_SCHEDULED
        self.scheduled_at = None

    def mark_dirty(self):
        self.unschedule()
        self.state = JobState.DIRTY

    def is_scheduled(self):
        return self.state is JobState.SCHEDULED

    def is_not_scheduled(self):
        return self.state is JobState.NOT_SCHEDULED

    def is_dirty(self):
        return self.state is JobState.DIRTY

    @property
    def S(self):
        return self.scheduled_at

    @property
    def C(self):
        return self.scheduled_at + self.p

    def __str__(self):
        time_range = f'scheduled at ({self.S:.2f} - {self.C:.2f})' if self.scheduled_at else ''
        
        return f'({self.p:.2f}, {self.r:.2f} {time_range})'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if other is None:
            return False
        return id(self) < id(other)

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


class JobSet:
    def __init__(self, jobs: Iterable[Job]):
        self.jobs = list(jobs)
        
    def by_resource_ascending(self):
        return list(sorted(self.jobs, key=lambda j: j.r))

    def by_resource_descending(self):
        return list(reversed(self.by_resource_ascending()))

    def by_length_descending(self):
        return sorted(self.jobs, key=lambda j: -j.p)

    def is_light_set(self, threshold: float, machines_count: int):
        jobs_resources = map(lambda j: j.r, self.jobs)
        jobs_by_resource_descending = list(reversed(sorted(jobs_resources)))
        return sum(jobs_by_resource_descending[:machines_count]) <= threshold

    def unschedule(self):
        for j in self.jobs:
            j.unschedule()

    def mark_dirty(self):
        for j in self.jobs:
            j.mark_dirty()

    def filter(self, cond):
        return JobSet(filter(cond, self.jobs))

    def dirty_jobs(self):
        return self.filter(lambda j: j.is_dirty())

    def not_scheduled_jobs(self):
        return self.filter(lambda j: j.is_not_scheduled())

    def vol_bound(self, m):
        return sum(j.p for j in self.jobs) / m

    def res_bound(self):
        return sum(j.r * j.p for j in self.jobs)

    def job_length_bound(self):
        return max(j.p for j in self.jobs)

    def __str__(self):
        return str(self.jobs)

    def __add__(self, other: 'JobSet'):
        return JobSet(self.jobs + other.jobs)

    def __bool__(self):
        return bool(self.jobs)


class Instance:
    def __init__(self, jobs: JobSet, machines_count: int):
        self.jobs = jobs
        self.machines_count = machines_count

    def reset(self):
        self.jobs.unschedule()

class Machine:
    def __init__(self, i):
        self.i = i
        self.jobs: List[Job] = []

    def put(self, j: Job, t=None):
        if t is None:
            t = self.C_max

        if self.C_max > t:
            raise RuntimeError(f'Machine {self.i} is already occupied at time {t}. '
                               f'Please schedule jobs in time order.')

        self.jobs.append(j)

    @property
    def C_max(self):
        if self.jobs:
            return self.jobs[-1].C
        return 0

    def __str__(self):
        return f'Machine {self.i} {self.jobs}'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other: 'Machine'):
        return


class Schedule:
    TimeType = float

    def __init__(self, machines_count):
        self.machines_count = machines_count

    def schedule(self, j: Job, t: 'Schedule.TimeType'):
        raise NotImplementedError

    def C_max(self) -> 'Schedule.TimeType':
        raise NotImplementedError

    def get_machines(self):
        raise NotImplementedError


class OrderedSet:
    """Set that pops the smallest element"""
    def __init__(self, values):
        self.lst = list(values)
        heapq.heapify(self.lst)

    def add(self, v):
        heapq.heappush(self.lst, v)

    def pop(self):
        v = heapq.heappop(self.lst)

        while self.lst and self.lst[0] == v:
            heapq.heappop(self.lst)

        return v


class IntervalTreeSchedule(Schedule):
    def __init__(self, machines_count):
        super().__init__(machines_count)
        self.tree = IntervalTree()
        self.action_points = set()
        self.resource_cautious_schedule = ResourceCautiousSchedule(self)

    def schedule(self, j: Job, t: 'Schedule.TimeType', m: Machine = None, check_for_machine = True):
        if check_for_machine and len(self.tree.at(t)) == self.machines_count:
            return False

        self.action_points.add(t + j.p)
        self.tree.addi(t, t + j.p, data=j)
        j.schedule(t, m, self)

        return True

    def LPT(self, jobs: JobSet, start_from: float = 0):
        free_machine_at = [start_from for _ in range(self.machines_count)]
        for j in jobs.by_length_descending():
            t = heapq.heappop(free_machine_at)
            self.schedule(j, t)
            heapq.heappush(free_machine_at, t + j.p)

    def resources_consumption_at(self, t: 'Schedule.TimeType') -> float:
        return sum(i.data.r for i in self.tree.at(t))

    def C_max(self) -> 'Schedule.TimeType':
        return self.tree.end()

    def get_machines(self):
        machines = [Machine(i) for i in range(self.machines_count)]

        # OrderedSet is used for purely cosmetic reasons
        free_machines = OrderedSet(range(self.machines_count))
        finished_jobs = []

        def acknowledge_idle_machines(t, resources):
            """As we move to the next moment we should reclaim all machines that become idle before t and
            resources used by jobs scheduled on them"""
            while finished_jobs and finished_jobs[0][0] <= t:
                _, old_job, old_machine = heapq.heappop(finished_jobs)
                free_machines.add(old_machine)
                resources -= Decimal(old_job.r)

            return resources

        resources = Decimal(0)
        # Sorting order: i.begin is for correctness, -i.data.r for cosmetics
        for start, end, job in sorted(self.tree, key=lambda i: (i.begin, -i.data.r)):
            resources = acknowledge_idle_machines(start, resources)

            assert free_machines
            machine_id = free_machines.pop()
            machines[machine_id].put(job, start)

            heapq.heappush(finished_jobs, (end, job, machine_id))

            resources += Decimal(job.r)
            assert resources <= 1 + 1e-9

        return machines

    def jobs_running_after(self, t: 'Schedule.TimeType') -> JobSet:
        return JobSet([i.data for i in self.tree if i.end > t])

    def jobs_after(self, t: 'Schedule.TimeType') -> JobSet:
        return JobSet([i.data for i in self.tree if i.begin >= t])

    def schedule_jobs_on_one_machine_by_r_j(self, jobs: JobSet):
        t = 0
        for j in jobs.by_resource_descending():
            self.schedule(j, t)
            t += j.p

    def medium_guarantee(self):
        last_value = 0

        for ap in reversed(sorted(self.action_points)):
            if self.resources_consumption_at(ap) >= 2/3:
                return last_value
            last_value = ap
        return last_value

    def schedule_jobs_with_infill_guarantee(self, threshold: float, jobs: JobSet, start_at=0):
        heap = self.get_action_points_heap(start_at)
        t = start_at

        for j in jobs.by_resource_descending():
            t = self.first_moment_with_resource_consumption_less_than(threshold, heap)
            if not self.schedule(j, t):
                return t
            heapq.heappush(heap, t + j.p)
        return self.first_moment_with_resource_consumption_less_than(threshold, heap)

    def schedule_jobs_with_infill_guarantee_heuristic(self, threshold: float, jobs: JobSet, start_at=0):
        heap = self.get_action_points_heap(start_at)
        t = start_at

        jobs_ordered = jobs.by_resource_ascending()

        while jobs_ordered:
            t = heapq.heappop(heap)
            rr = Decimal(self.resources_consumption_at(t))
            j = jobs_ordered[-1]
            jobs_at = len(self.tree.at(t))

            while j and rr + Decimal(j.r) <= 1 and jobs_at < self.machines_count:
                self.schedule(j, t)
                jobs_at += 1
                rr += Decimal(j.r)
                jobs_ordered.pop()
                heapq.heappush(heap, t + j.p)
                if jobs_ordered:
                    j = jobs_ordered[-1]
                else:
                    j = None

            if rr < threshold:
                break

        while self.resources_consumption_at(t) >= threshold:
            t = heapq.heappop(heap)

        return t

    def get_action_points_heap(self, start_at=0):
        heap = list(self.action_points)
        heapq.heapify(heap)
        while heap and heap[0] < start_at:
            heapq.heappop(heap)
        if not heap:
            heap = [start_at]
        return heap

    def first_moment_with_resource_consumption_less_than(self, threshold, heap=None) -> 'Schedule.TimeType':
        if heap is None:
            heap = self.get_action_points_heap()

        while heap:
            t = heapq.heappop(heap)
            if self.resources_consumption_at(t) < threshold:
                heapq.heappush(heap, t)
                return t

        raise RuntimeError("Should never reach that moment")

    def get_machine_by_number(self, machine_nr):
        return machine_nr

    def filter(self, fun):
        return JobSet([j for j in self.tree if fun(j)])

    def jobs_running_at(self, t):
        return JobSet(i.data for i in self.tree.at(t))

    def unschedule(self, j: Job):
        self.tree.removei(j.S, j.C, j)

    def generate_resource_consumption_array(self):
        self.resource_cautious_schedule.generate_resource_consumption_array()

    def fit_in_first_place(self, j):
        self.resource_cautious_schedule.fit_in_first_place(j)


class Scheduler:
    def __init__(self, instance, schedule=None):
        self.instance = instance
        if not schedule:
            schedule = IntervalTreeSchedule(self.instance.machines_count)
        self.schedule = schedule

    def _run(self, start_at=0):
        pass

    def run(self, start_at=0) -> Schedule:
        self._run(start_at=start_at)

        assert not self.instance.jobs.not_scheduled_jobs()
        return self.schedule


class LinkedList:
    def __init__(self, t, val, refs):
        assert refs >= 0
        self.t = t
        self.val = val
        self.refs = refs
        self.nxt = None
        self.prev = None

    def incr(self):
        self.refs += 1

    def decr(self):
        assert self.refs > 0
        self.refs -= 1

    def insert_after(self, el: 'LinkedList'):
        el.nxt = self.nxt
        el.prev = self

        if self.nxt:
            self.nxt.prev = el
        self.nxt = el

    def insert_before(self, el: 'LinkedList'):
        el.nxt = self
        el.prev = self.prev
        self.prev.nxt = el  # we won't try to insert before 0
        self.prev = el

    def find(self, t):
        el = self
        while el.t < t:
            el = el.nxt
        return el

    def __iter__(self):
        return self

    def __next__(self):
        if self.nxt is None:
            raise StopIteration
        return self.nxt

    def __str__(self):
        return f'{self.t:.2f}: {self.val:.2f}'


class ResourceCautiousSchedule:
    def __init__(self, schedule):
        self.schedule = schedule
        self.machines_count = schedule.machines_count
        self.action_points = set()
        self.action_points_list_map = {}

    def generate_resource_consumption_array(self):
        self.init_action_points()

        tmp_array = len(self.action_points) * [Decimal(0)]
        mach_count = len(tmp_array) * [0]
        sorted_action_points = sorted(self.action_points)

        action_point_id = {ap : i for i, ap in enumerate(sorted_action_points)}

        for s, e, j in self.schedule.tree:
            tmp_array[action_point_id[s]] += Decimal(j.r)
            mach_count[action_point_id[s]] += 1
            tmp_array[action_point_id[e]] -= Decimal(j.r)
            mach_count[action_point_id[e]] -= 1

        list_head = None
        last_val = Decimal(0)
        machines = 0
        for i, (p, mc) in enumerate(zip(tmp_array, mach_count)):
            last_val += p
            machines += mc
            assert last_val <= 1
            ll = LinkedList(sorted_action_points[i], last_val, machines)
            self.action_points_list_map[i] = ll

            if list_head:
                list_head.insert_after(ll)
            list_head = ll

    def init_action_points(self):
        for s, e, j in self.schedule.tree:
            self.action_points.add(s)
            self.action_points.add(e)

    def fit_in_first_place(self, j: Job):
        head = self.action_points_list_map[0]

        self.unschedule_job_from_list(head, j)

        head, tail = self._find_first_fit(head, j)

        self.schedule.schedule(j, head.t)

        while head != tail:
            head.val += Decimal(j.r)
            head.incr()
            if head.refs > self.machines_count:
                raise RuntimeError
            head = head.nxt

        if tail.t != j.C:
            assert tail.t > j.C
            tail.insert_before(LinkedList(j.C, tail.prev.val, tail.prev.refs))

        return head

    def _find_first_fit(self, head, j):
        tail = head
        while tail and head.t + j.p > tail.t:
            if tail.val > 1 - j.r or tail.refs == self.machines_count:
                head = tail.nxt
                tail = head
                continue

            tail = tail.nxt

        assert tail
        return head, tail

    def unschedule_job_from_list(self, head, j):
        job_head = head.find(j.scheduled_at)
        while job_head.t != j.C:
            job_head.val -= Decimal(j.r)
            job_head.decr()
            job_head = job_head.nxt
        j.unschedule()


def get_instance(simulation_input: SimulationInput, m, n):
    generate = True
    while generate:
        simulation_input.prepare(n, m)
        jobs = JobSet([Job(*simulation_input()) for _ in range(n)])
        instance = Instance(jobs, m)
        bounds = [(instance.jobs.vol_bound(m), 'VOL OPT'), (instance.jobs.res_bound(), 'RES OPT'),
                  (instance.jobs.job_length_bound(), 'MAX OPT')]
        generate = bounds[2] > max(bounds[0], bounds[1])

    ref = max(bounds)
    return [ref[0]], [ref[1]], ref[0], instance


class BoxBuilder:
    def __init__(self):
        self.row_length = None
        self.content = ''
        self.row_content = []
        self.row_title = ''
        self.titles = []
        self.results = defaultdict(list)

    def start(self):
        pass

    def start_row(self, title, row_length):
        self.row_content = []
        self.row_title = title
        self.row_length = row_length
        self.content = ''''''

    def end_row(self):
        for box in self.row_content:
            self.content += box
        pass

    def end(self):
        return self.content

    def add_title(self, title):
        self.titles.append(title)

    def add_result(self, title, times):
        if title not in self.titles:
            self.titles.append(title)
        self.results[title].append(float(times))

    def add_boxplot(self):
        boxes = []
        times_list = []

        for title in self.titles:
            times_list.append(self.results[title])

        for times in times_list:
            if not times:
                continue
            times = np.array(times)
            min = np.min(times)
            p25 = np.percentile(times, 25)
            mean = np.mean(times)
            p75 = np.percentile(times, 75)
            max = np.max(times)

            boxes.append(f"""\\addplot+[
            boxplot prepared={{
              median={mean},
              upper quartile={p75},
              lower quartile={p25},
              upper whisker={max},
              lower whisker={min}
            }},
            ] coordinates {{}};""")

        self.row_content.append(f"""
    \\begin{{tikzpicture}}
      \\begin{{axis}}
        [
        boxplot/draw direction = y,
        xticklabel style = {{align=center, font=\small, rotate=60}},
        xtick={{{', '.join([str(i + 1) for i, _ in enumerate(self.titles)])}}},
        xticklabels={{{','.join(self.titles)}}},
        ]
        {' '.join(boxes)}
      \\end{{axis}}
    \\end{{tikzpicture}}""")

        self.results.clear()
        self.title = []


class SimulationRunner:
    def __init__(self, algorithms: List[Tuple[Schedule, str]], simulation_input: SimulationInput,
                 params: List[Tuple[int, int]],
                 reps: int, output_prefix: str):
        self.algorithms = algorithms
        self.simulation_input = simulation_input
        self.params = params
        self.reps = reps
        self.approx_boxplot = BoxBuilder()
        self.time_boxplot = BoxBuilder()
        self.output_prefix = output_prefix

    def handle_input(self, args):
        alg, name, inp = args
        bounds, bounds_titles, ref, instance = inp

        t = time()
        schedule = alg(instance).run()

        schedule.get_machines()

        runtime = time() - t

        approx = schedule.C_max() / ref
        if approx < 1:
            raise RuntimeError(f"{name} gave {approx} approx")
        self.approx_boxplot.add_result(name, approx)
        self.time_boxplot.add_result(name, runtime)
        instance.reset()

    def run(self):
        total = self.reps * len(self.params) * len(self.algorithms)
        cur = 1

        schedules_graphic = []
        self.time_boxplot.start_row('Runtime [s]', 2)

        for n, m in self.params:
            self.approx_boxplot.start_row(f'({n}, {m})', 2)
            inputs = [get_instance(self.simulation_input, m=m, n=n) for _ in range(self.reps)]
            for alg, name in self.algorithms:
                for inp in inputs:
                    print(f'Proceeding {name} with params n={n}, m={m} {cur}/{total}')
                    cur += 1
                    self.handle_input((alg, name, inp))

            for bounds, bounds_titles, ref, instance in inputs:
                for bound, bound_title in zip(bounds, bounds_titles):
                    self.approx_boxplot.add_result(bound_title, bound / ref)
            self.approx_boxplot.add_boxplot()
            self.time_boxplot.add_boxplot()

            self.approx_boxplot.end_row()
            approx_latex = self.approx_boxplot.end()
            self.approx_boxplot.start_row('', 2)

        self.time_boxplot.end_row()
        time_latex = self.time_boxplot.end()
        for name, slide in schedules_graphic:
            time_latex += f'\n\n{name}'

        print('Generating latex files...')
        print_latex(approx_latex, filename=f'{self.output_prefix}_approx')
        print_latex(time_latex, filename=f'{self.output_prefix}_runtime')
