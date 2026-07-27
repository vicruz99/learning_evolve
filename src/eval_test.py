import asyncio, tempfile
from envs import EnvConfig, get_problem
from sandbox import init_ray

class _Dummy:  # Environment needs a sampler; grading never uses it
    def update_states(self,*a,**k): pass
    def record_failed_rollout(self,*a,**k): pass

async def grade(problem, code, eval_timeout=530):
    init_ray(get_problem(problem).num_cpus_per_task)
    spec = get_problem(problem)
    initial = spec.env_type.create_initial_state(spec.problem_type)
    with tempfile.TemporaryDirectory() as td:
        cfg = EnvConfig(problem_type=spec.problem_type, log_path=td,
                        num_cpus_per_task=spec.num_cpus_per_task,
                        eval_timeout=eval_timeout, timeout=eval_timeout+60)
        env = spec.env_type(initial_state=initial, sampler=_Dummy(), config=cfg)
        outs = await env.check_answer(code, step=0)   # code = a ```python ...``` string
        print(outs.correctness, outs.raw_score, outs.failure_type, outs.msg)#[:200])

asyncio.run(grade("circle_packing_26", open("/home/guests2/vic/work/projects/phd/learning_evolve/src/runs/cp_26_no_icl_g5x12_gen30v2/generations/gen_0000/parent_00/child_06.txt").read()))