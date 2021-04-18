from simulator import Instance, JobSet, Scheduler


class ApAlg(Scheduler):
    def run(self, check_assertions=False):
        schedule, instance = self.schedule, self.instance
        jobs = instance.jobs
        m = instance.machines_count

        J_heavy = jobs.filter(lambda j: j.r > 1 / 2)
        J_light = jobs.filter(lambda j: j.r <= 1 / 3)

        if jobs.heavy_jobs():
            schedule.schedule_jobs_on_one_machine_by_r_j(J_heavy)
            C_heavy = schedule.ends_at()

            t_first_guarantee = schedule.schedule_jobs_with_infill_guarantee(2 / 3, J_light, start_at=0)
            t_first_cut = min(t_first_guarantee, C_heavy)

            J_dirty_light = jobs.filter(lambda j: j.is_scheduled() and j.s < t_first_cut < j.c and j.r <= 1 / 3)
            J_dirty_light.mark_dirty()
            jobs.filter(lambda j: j.is_scheduled() and j.s >= t_first_cut and j.r <= 1 / 3).unschedule()

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

        BRR(Instance(jobs.filter(lambda j: 1 / 3 < j.r <= 1 / 2), 2), schedule).run(start_at=t_first_cut)
        t_medium_guarantee = max(schedule.medium_guarantee(), t_first_cut)

        if C_heavy == t_first_cut:
            t_second_cut = \
                schedule.schedule_jobs_with_infill_guarantee(2 / 3,
                                                             jobs.filter(
                                                                 lambda j: j.is_not_scheduled() and j.r <= 1 / 3),
                                                             start_at=t_medium_guarantee)
        else:
            t_second_cut = max(C_heavy, t_medium_guarantee)

        C_alg = schedule.ends_at()

        if C_heavy == C_alg:
            if check_assertions:
                assert jobs.not_scheduled().is_light_set(2 / 3, m)
            J_lpt = J_dirty_light + jobs.not_scheduled()
        else:
            J_dirty = jobs.filter(lambda j: j.is_scheduled() and j.s <= t_second_cut < j.c)
            J_dirty.mark_dirty()

            if check_assertions:
                if C_heavy == t_first_cut:
                    assert (J_dirty + jobs.not_scheduled()).is_light_set(2 / 3, m)
                else:
                    assert J_dirty.is_light_set(1 / 2, m)
                    assert jobs.not_scheduled().is_light_set(1 / 3, m)
            J_lpt = J_dirty + J_dirty_light + jobs.not_scheduled()

        if check_assertions:
            assert J_lpt.is_light_set(1, m)

        t_lpt = schedule.ends_at()
        schedule.LPT(J_lpt, t_lpt)

        for j in jobs.jobs:
            assert j.is_scheduled()

        return schedule
