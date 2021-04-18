import heapq
from collections import Iterable
from decimal import Decimal
from enum import Enum, auto
from typing import List

import IntervalTree as IntervalTree


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

        assert length > 0
        assert 0 <= resource <= 1

    def schedule(self, t):
        self.scheduled_at = t
        self.state = JobState.SCHEDULED

    def unschedule(self):
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

    def schedule(self, j: Job, t: 'Schedule.TimeType', m: Machine = None, check_for_machine = True):
        if check_for_machine and len(self.tree.at(t)) == self.machines_count:
            return False

        self.action_points.add(t + j.p)
        self.tree.addi(t, t + j.p, data=j)
        j.schedule(t)

        return True

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
        heap = list(self.action_points)
        heapq.heapify(heap)
        t = start_at

        while heap and heap[0] < t:
            heapq.heappop(heap)

        if not heap:
            heap = [t]

        for j in jobs.by_resource_descending():
            t = self.first_moment_with_resource_consumption_less_than(threshold, heap)
            if not self.schedule(j, t):
                return t
            heapq.heappush(heap, t + j.p)
        return self.first_moment_with_resource_consumption_less_than(threshold, heap)

    def schedule_jobs_with_infill_guarantee_heuristic(self, threshold: float, jobs: JobSet, start_at=0):
        heap = list(self.action_points)
        heapq.heapify(heap)
        t = start_at

        while heap and heap[0] < t:
            heapq.heappop(heap)

        if not heap:
            heap = [t]

        jobs_ordered = jobs.by_resource_ascending()

        while jobs_ordered:
            t = heapq.heappop(heap)
            rr = self.resources_consumption_at(t)
            j = jobs_ordered[-1]
            jobs_at = len(self.tree.at(t))

            while j and rr + j.r <= 1 and jobs_at < self.machines_count:
                self.schedule(j, t)
                jobs_at += 1
                rr += j.r
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

    def first_moment_with_resource_consumption_less_than(self, threshold, heap=None) -> 'Schedule.TimeType':
        if heap is None:
            heap = list(self.action_points)
            heapq.heapify(heap)

        while heap:
            t = heapq.heappop(heap)
            if self.resources_consumption_at(t) < threshold:
                heapq.heappush(heap, t)
                return t

        raise RuntimeError("Should never reach that moment")

    def get_machine_by_number(self, machine_nr):
        return machine_nr  # Machines

    def filter(self, fun):
        return JobSet([j for j in self.tree if fun(j)])

    def jobs_running_at(self, t):
        return JobSet(i.data for i in self.tree.at(t))

    def unschedule(self, j: Job):
        self.tree.removei(j.S, j.C, j)


class Scheduler:
    def __init__(self, instance, schedule):
        self.instance = instance
        self.schedule = schedule

    def _run(self, start_at=0):
        pass

    def run(self, start_at=0) -> Schedule:
        self._run(start_at=start_at)

        assert not self.instance.jobs.not_scheduled()
        return self.schedule