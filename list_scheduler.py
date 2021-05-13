import heapq
import random
from decimal import Decimal
from typing import List

from simulator import Job, Scheduler


class ListScheduler(Scheduler):
    def _list_scheduler_run(self, m: int, jobs: List[Job], start_at=0):
        schedule = self.schedule
        action_points = [(start_at, 0, 0)]  # (time, machine, resources)

        # 0 is not in the free_machines set.
        # Think of it as if there was a virtual job scheduled on machine 0
        # that ended at <start_at> and we're starting now.
        free_machines = set(range(1, m))
        scheduled_jobs = set()

        # First we handle already scheduled jobs, so that they are properly acknowledged by the algorithm
        org_jobs = schedule.jobs_running_after(start_at).jobs
        running_jobs = list()

        for j in sorted(org_jobs, key=lambda x: x.S):
            if j.S < start_at:
                running_jobs.append(Job(j.S - start_at + j.p, j.r))
            else:
                running_jobs.append(Job(j.p, j.r))

        jobs = list(running_jobs) + jobs

        rc = Decimal(0)
        while jobs:
            t, machine, jr = action_points[0]
            heapq.heappop(action_points)
            rc -= Decimal(jr)

            while action_points and action_points[0][0] <= t:
                _, m_1, jr_1 = heapq.heappop(action_points)
                free_machines.add(m_1)
                rc -= Decimal(jr_1)

            for j in jobs:
                if free_machines or machine is not None:
                    if len(free_machines) == m or (len(free_machines) == m - 1 and machine is not None):
                        # Due to floating ops it may be sth above 0 even if all machines are free,
                        # which breaks the algorithm.
                        rc = Decimal(0)

                    if rc + Decimal(j.r) <= 1:
                        if machine is None:
                            machine = free_machines.pop()

                        rc += Decimal(j.r)
                        if j not in running_jobs:
                            schedule.schedule(j, t, schedule.get_machine_by_number(machine), check_for_machine=False)
                        heapq.heappush(action_points, (t + j.p, machine, j.r))
                        scheduled_jobs.add(j)

                        if free_machines:
                            machine = free_machines.pop()
                        else:
                            machine = None
                else:
                    break

            if machine is not None:
                free_machines.add(machine)

            jobs = [j for j in jobs if j not in scheduled_jobs]

        return schedule

    def _run(self, start_at=0):
        return self._list_scheduler_run(self.instance.machines_count,
                                        self.instance.jobs.jobs,
                                        start_at=start_at)


class LPT(ListScheduler):
    def _run(self, start_at=0):
        return self._list_scheduler_run(self.instance.machines_count,
                                        self.instance.jobs.by_length_descending(),
                                        start_at=start_at)


class HRR(ListScheduler):
    def _run(self, start_at=0):
        return self._list_scheduler_run(self.instance.machines_count,
                                        self.instance.jobs.by_resource_descending(),
                                        start_at=start_at)


class LRR(ListScheduler):
    def _run(self, start_at=0):
        return self._list_scheduler_run(self.instance.machines_count,
                                        self.instance.jobs.by_resource_ascending(),
                                        start_at=start_at)


class RAND(ListScheduler):
    def _run(self, start_at=0):
        jobs = list(self.instance.jobs.jobs)[:]
        random.shuffle(jobs)
        return self._list_scheduler_run(self.instance.machines_count, jobs, start_at=start_at)
