from list_scheduler import HRR
from simulator import Instance, JobSet, Scheduler


class ApAlg(Scheduler):
    def run(self, check_assertions=False):
        schedule, instance = self.schedule, self.instance
        jobs = instance.jobs
        m = instance.machines_count

        J_heavy = jobs.filter(lambda j: j.r > 1 / 2)
        J_light = jobs.filter(lambda j: j.r <= 1 / 3)

        if J_heavy:
            schedule.schedule_jobs_on_one_machine_by_r_j(J_heavy)
            C_heavy = schedule.C_max()

            t_first_guarantee = schedule.schedule_jobs_with_infill_guarantee(2 / 3, J_light, start_at=0)
            t_first_cut = min(t_first_guarantee, C_heavy)

            J_dirty_light = jobs.filter(lambda j: j.is_scheduled() and j.S < t_first_cut < j.C and j.r <= 1 / 3)
            J_dirty_light.mark_dirty()
            jobs.filter(lambda j: j.is_scheduled() and j.S >= t_first_cut and j.r <= 1 / 3).unschedule()

            if check_assertions:
                if C_heavy == t_first_cut:
                    assert J_dirty_light.is_light_set(1 / 3, m)
                else:
                    assert J_dirty_light.is_light_set(1 / 6, m)
                    assert jobs.filter(lambda j: j.is_not_scheduled() and j.r <= 1 / 3).is_light_set(1 / 3, m)
        else:
            J_dirty_light = JobSet([])
            C_heavy = 0
            t_first_cut = 0

        HRR(Instance(jobs.filter(lambda j: 1 / 3 < j.r <= 1 / 2), 2), schedule).run(start_at=t_first_cut)
        t_medium_guarantee = max(schedule.medium_guarantee(), t_first_cut)

        if C_heavy == t_first_cut:
            t_second_cut = \
                schedule.schedule_jobs_with_infill_guarantee(2 / 3,
                                                             jobs.filter(
                                                                 lambda j: j.is_not_scheduled() and j.r <= 1 / 3),
                                                             start_at=t_medium_guarantee)
        else:
            t_second_cut = max(C_heavy, t_medium_guarantee)

        C_alg = schedule.C_max()

        if C_heavy == C_alg:
            if check_assertions:
                assert jobs.not_scheduled_jobs().is_light_set(2 / 3, m)
            J_lpt = J_dirty_light + jobs.not_scheduled_jobs()
        else:
            J_dirty = jobs.filter(lambda j: j.is_scheduled() and j.S <= t_second_cut < j.C)
            J_dirty.mark_dirty()

            if check_assertions:
                if C_heavy == t_first_cut:
                    assert (J_dirty + jobs.not_scheduled_jobs()).is_light_set(2 / 3, m)
                else:
                    assert J_dirty.is_light_set(1 / 2, m)
                    assert jobs.not_scheduled_jobs().is_light_set(1 / 3, m)
            J_lpt = J_dirty + J_dirty_light + jobs.not_scheduled_jobs()

        if check_assertions:
            assert J_lpt.is_light_set(1, m)

        t_lpt = schedule.C_max()
        schedule.LPT(J_lpt, t_lpt)

        for j in jobs.jobs:
            assert j.is_scheduled()

        return schedule


class ApAlgS(ApAlg):
    def run(self, check_assertions=False):
        schedule = super().run(check_assertions)
        schedule.get_resource_consumption_array()
        jobs = sorted(self.instance.jobs.jobs, key=lambda x: x.S)

        for j in jobs:
            schedule.fit_in_first_place(j)

        return schedule


class ApAlgH(Scheduler):
    def run(self, check_assertions=False):
        schedule, instance = self.schedule, self.instance
        jobs = instance.jobs
        m = instance.machines_count

        J_heavy = jobs.filter(lambda j: j.r > 1 / 2)
        J_light = jobs.filter(lambda j: j.r <= 1 / 3)

        if J_heavy:
            schedule.schedule_jobs_on_one_machine_by_r_j(J_heavy)
            C_heavy = schedule.C_max()

            t_first_guarantee = self.schedule_jobs_with_infill_guarantee(2 / 3, J_light, start_at=0)
            t_first_cut = min(t_first_guarantee, C_heavy)

            self.apply_heuristics(t_first_cut, C_heavy)

            J_dirty_light = jobs.filter(lambda j: j.is_scheduled() and j.S < t_first_cut < j.C and j.r <= 1 / 3)
            J_dirty_light.mark_dirty()
            jobs.filter(lambda j: j.is_scheduled() and j.S >= t_first_cut and j.r <= 1 / 3).unschedule()

            if check_assertions:
                if C_heavy == t_first_cut:
                    assert J_dirty_light.is_light_set(1 / 3, m)
                else:
                    assert J_dirty_light.is_light_set(1 / 6, m)
                    assert jobs.filter(lambda j: j.is_not_scheduled() and j.r <= 1 / 3).is_light_set(1 / 3, m)
        else:
            J_dirty_light = JobSet([])
            C_heavy = 0
            t_first_cut = 0

        HRR(Instance(jobs.filter(lambda j: 1 / 3 < j.r <= 1 / 2), 2), schedule).run(start_at=t_first_cut)
        t_medium_guarantee = max(schedule.medium_guarantee(), t_first_cut)

        if C_heavy == t_first_cut:
            t_second_cut = \
                self.schedule_jobs_with_infill_guarantee(2 / 3,
                                                         jobs.filter(
                                                             lambda j: j.is_not_scheduled() and j.r <= 1 / 3),
                                                         start_at=max(self.get_last_gap_end(), t_first_cut))
        else:
            t_second_cut = max(C_heavy, t_medium_guarantee)

        C_alg = schedule.C_max()

        if C_heavy == C_alg:
            if check_assertions:
                assert jobs.not_scheduled().is_light_set(2 / 3, m)
            J_lpt = J_dirty_light + jobs.not_scheduled_jobs()
        else:
            J_dirty = jobs.filter(lambda j: j.is_scheduled() and j.S <= t_second_cut < j.C)
            J_dirty.mark_dirty()

            if check_assertions:
                if C_heavy == t_first_cut:
                    assert (J_dirty + jobs.not_scheduled_jobs()).is_light_set(2 / 3, m)
                else:
                    assert J_dirty.is_light_set(1 / 2, m)
                    assert jobs.not_scheduled_jobs().is_light_set(1 / 3, m)
            J_lpt = J_dirty + J_dirty_light + jobs.not_scheduled_jobs()

        if check_assertions:
            assert J_lpt.is_light_set(1, m)

        t_lpt = schedule.C_max()
        schedule.LPT(J_lpt, t_lpt)

        for j in jobs.jobs:
            assert j.is_scheduled()

        schedule.graphic_data = [(C_heavy, 'C_heavy'), (t_first_cut, 't_first_cut'),
                                 (t_medium_guarantee, 't_medium_guarantee'), (t_second_cut, 't_second_cut'),
                                 (t_lpt, 't_lpt')]

        return schedule

    def schedule_jobs_with_infill_guarantee(self, threshold, jobs, start_at=0):
        return self.schedule.schedule_jobs_with_infill_guarantee_heuristic(threshold, jobs, start_at)

    def get_last_gap_end(self):
        epsilon = 0
        tree = self.schedule.tree

        unique_action_points = sorted([j.data.scheduled_at for j in tree] + [j.data.C for j in tree])

        for first, second in zip(unique_action_points, unique_action_points[1:]):
            if not epsilon or epsilon > second - first:
                epsilon = second - first

        start_points = [j.data.scheduled_at for j in tree]

        for ap in reversed(sorted(start_points)):
            if len(tree.at(ap)) == 2 and len(tree.at(ap - epsilon)) == 1:
                return ap

        return 0

    def apply_heuristics(self, t_cut, C_heavy):

        if t_cut == C_heavy:
            res = self.schedule.resources_consumption_at(t_cut)
            jobs = self.schedule.jobs_running_at(t_cut).by_resource_ascending()

            while res > 1 / 3:
                res -= jobs[-1].r
                jobs.pop().unschedule()
