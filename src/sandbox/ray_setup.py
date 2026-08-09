"""Ray bootstrap for the sandboxed evaluators.

The sandbox executes each candidate solution in a Ray remote task that pins CPUs via a detached
``cpu_scheduler`` actor (see :mod:`sandbox.sandbox_reward_evaluator`). Call :func:`init_ray` once,
before constructing any evaluator, to make sure Ray is up and the scheduler actor exists.

Extracted from TTT-Discover's ``discovery.init_ray`` (the ``AhcEnv`` special-case is dropped —
irrelevant to the math benchmarks).
"""
from __future__ import annotations

import logging

from sandbox.cpu_scheduler import CpuScheduler

logger = logging.getLogger("icl.sandbox")

RAY_NAMESPACE = "icl"


def init_ray(num_cpus_per_task: int, num_persistent_workers: int = 0, *,
             ray_num_cpus: int | None = None) -> None:
    """Initialize Ray and ensure the detached ``cpu_scheduler`` actor is running.

    Args:
        num_cpus_per_task: CPUs handed to each evaluation task (matches the evaluator's
            ``num_cpus_per_task``). Must be >= 1.
        num_persistent_workers: CPUs to reserve away from the schedulable pool.
        ray_num_cpus: cores for the cluster this call may have to START. It cannot apply to a shared
            head — ``--num-cpus`` is fixed at ``ray start`` and a connecting client has no say — so
            it is deliberately ignored (loudly) on the attach path rather than silently accepted.
    """
    import ray

    if not ray.is_initialized():
        # namespace= silences the "detached actor in an anonymous namespace" warning; the other
        # flags drop the dashboard URL line and per-worker log spam from the console.
        common = dict(
            namespace=RAY_NAMESPACE,
            include_dashboard=False,
            log_to_driver=False,
            logging_level=logging.WARNING,
        )
        # Prefer a shared head started out-of-band (`ray start --head`): every concurrent experiment
        # then connects to ONE 96-CPU pool + one cpu_scheduler, so Ray caps *total* grading across all
        # runs (no core oversubscription) and simultaneous launches don't each boot their own head
        # (which deadlocks). address="auto" reads the local head's address file — no RAY_ADDRESS env
        # needed. If no shared head is running, fall back to a private per-run cluster (original path).
        try:
            ray.init(address="auto", **common)
            logger.info("Connected to shared Ray head (address=auto).")
            if ray_num_cpus is not None:
                logger.warning(
                    f"--ray-num-cpus={ray_num_cpus} IGNORED: this run attached to a head that is "
                    "already running, and its --num-cpus was fixed when it started. Restart the "
                    f"head with `ray stop && ray start --head --num-cpus={ray_num_cpus}`, or let "
                    "run_sweep.py own it (--ray-head auto --ray-num-cpus N).")
        except ConnectionError:
            ray.init(**common, **({} if ray_num_cpus is None else {"num_cpus": ray_num_cpus}))
            logger.info("No shared Ray head found; started a private Ray cluster for this run"
                        + (f" with num_cpus={ray_num_cpus}." if ray_num_cpus is not None
                           else " sized to the whole box."))

    # get_if_exists makes this atomic: returns the existing detached actor or creates it, with no
    # get-then-create race. Without it, several runs starting against a shared head at the same instant
    # can all see "no scheduler" and all try to create it -> all but one crash with "actor already exists".
    CpuScheduler.options(
        name="cpu_scheduler",
        lifetime="detached",
        get_if_exists=True,
    ).remote(
        num_cpus_per_task=num_cpus_per_task,
        num_persistent_workers=num_persistent_workers,
    )
