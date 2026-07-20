"""Slim, tinker-free ``Environment`` base for the ICL harness.

Extracted from TTT-Discover's ``dataset_builder.Environment`` (`191-441`), keeping only the
problem-definition surface (``create_initial_state`` / ``get_question`` / ``is_maximize``) and the
grading path (``check_answer`` -> ``_safe_grade`` -> ``_run_verification`` ->
``reward_function.get_reward``). Dropped: the ``ProblemEnv`` base, the ``renderer`` / ``convo_prefix``
constructor args, and the token-based ``step()`` (its logic lives in :meth:`Environment.rollout_step`,
which returns plain data instead of a tinker ``StepResult``).
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from puct.state import State
from envs.codeblock import last_codeblock_postprocess

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Everything the grading path needs from a run's config.

    Mirrors the subset of TTT-Discover's ``DatasetConfig`` that ``Environment`` reads.
    """
    problem_type: str
    log_path: str
    num_cpus_per_task: int = 1
    eval_timeout: int = 1000          # sandbox (per-candidate) execution timeout, seconds
    timeout: float = 8000.0           # async grading wall-clock timeout, seconds


@dataclass
class VerifyResult:
    reward: float
    msg: str
    correctness: float
    raw_score: float
    result_construction: Any
    stdout: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    """Outcome of grading one completion (the tinker-free analogue of ``StepResult``)."""
    reward: float
    correctness: float
    raw_score: float
    msg: str
    parsed_code: str
    correct_format: bool
    next_state: State | None          # the buffered child if the candidate was valid, else None


# Shared ThreadPoolExecutor for all environments (grading runs off the event loop).
SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


class Environment(ABC):

    reward_function: type          # concrete SandboxRewardEvaluator subclass
    state_type: type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        """Create an initial state for rollouts. Override in subclasses that need a different seed."""
        return State(timestep=-1, construction=None, code="", value=0.0)

    def __init__(self, initial_state: State, sampler, config: EnvConfig):
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")

        self.config = config
        self.timeout = config.timeout
        self.num_cpus_per_task = config.num_cpus_per_task
        self.eval_timeout = config.eval_timeout
        self.log_path = config.log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state
        self.problem_type = config.problem_type

    @abstractmethod
    def get_question(self) -> str:
        """Build the base prompt from the current ``initial_state``."""

    def is_maximize(self) -> bool:
        return True

    def _create_next_state(self, step_idx: int, parsed_code: str, outs: VerifyResult) -> State:
        return self.state_type(
            timestep=step_idx,
            construction=outs.result_construction,
            code=parsed_code,
            value=outs.raw_score if self.is_maximize() else -outs.raw_score,  # higher = better
            observation=outs.stdout,
        )

    def _get_code_languages(self) -> list[str]:
        return ["python"]

    def _should_keep_code_separators(self) -> bool:
        return True

    def check_format(self, parsed_code: str) -> bool:
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return False
        return True

    async def check_answer(self, parsed_code: str, step: int) -> VerifyResult:
        """Grade a parsed code string asynchronously, with a timeout."""
        if not self.check_format(parsed_code):
            return VerifyResult(
                reward=0.0,
                msg="Invalid code",
                correctness=0.0,
                raw_score=0.0,
                result_construction=None,
                stdout="",
            )
        return await self._safe_grade(parsed_code, step)

    def _run_verification(self, generation: str, problem_type: str, log_path: str, state: State) -> VerifyResult:
        task = self.reward_function(
            problem_type=problem_type,
            log_dir=log_path,
            eval_timeout=self.eval_timeout,
            num_cpus_per_task=self.num_cpus_per_task,
        )
        out = task.get_reward(generation, state=state)
        return VerifyResult(
            reward=out["reward"],
            msg=out["msg"],
            correctness=out["correctness"],
            raw_score=out["raw_score"],
            result_construction=out.get("result_construction", None),
            stdout=out.get("stdout", ""),
            metrics=out.get("metrics", {}),
        )

    async def _safe_grade(self, given_answer: str, step: int) -> VerifyResult:
        """Run ``_run_verification`` in a background thread with an asyncio timeout."""
        loop = asyncio.get_running_loop()
        start_time = time.time()
        try:
            out = await asyncio.wait_for(
                loop.run_in_executor(
                    SAFE_GRADE_EXECUTOR,
                    partial(
                        self._run_verification,
                        given_answer,
                        self.problem_type,
                        self.log_path,
                        self.state,
                    ),
                ),
                timeout=self.timeout,
            )
            return out
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"Timeout grading: took {elapsed:.1f}s, limit was {self.timeout:.1f}s")
            return VerifyResult(
                reward=0.0, msg="Timeout grading", correctness=0.0, raw_score=0.0,
                result_construction=None, stdout="",
            )
        except Exception as e:
            import traceback
            error_msg = f"Error grading: {e}\n{traceback.format_exc()}"
            logger.warning(f"Exception while grading: {e}")
            return VerifyResult(
                reward=0.0, msg=f"Error grading: {error_msg}", correctness=0.0, raw_score=0.0,
                result_construction=None, stdout="",
            )

    async def rollout_step(self, completion_text: str, step_idx: int) -> RolloutResult:
        """Parse a raw completion, grade it, and feed the result back into the buffer.

        Faithful port of TTT-Discover's ``Environment.step`` minus the tinker ``StepResult``:
        a valid child is added to the sampler via ``update_states``; an invalid/failed one is
        recorded via ``record_failed_rollout`` (so PUCT visit counts advance either way).
        """
        parsed_code = last_codeblock_postprocess(
            completion_text,
            codeblock_seps=self._get_code_languages(),
            keep_separators=self._should_keep_code_separators(),
        )
        correct_format = bool(self.check_format(parsed_code))

        outs = await self.check_answer(parsed_code, step_idx)

        next_state: State | None = None
        if outs.correctness > 0:
            try:
                next_state = self._create_next_state(step_idx, parsed_code, outs)
                self.sampler.update_states([next_state], [self.initial_state], save=False)
            except Exception as e:
                logger.warning(f"Failed to create next state: {e}")
                next_state = None
                if hasattr(self.sampler, "record_failed_rollout"):
                    self.sampler.record_failed_rollout(self.initial_state)
        elif hasattr(self.sampler, "record_failed_rollout"):
            self.sampler.record_failed_rollout(self.initial_state)

        return RolloutResult(
            reward=outs.reward,
            correctness=outs.correctness,
            raw_score=outs.raw_score,
            msg=outs.msg,
            parsed_code=parsed_code,
            correct_format=correct_format,
            next_state=next_state,
        )
