from collections import Iterable
from enum import Enum, auto
from typing import List


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
