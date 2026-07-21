"""The ICL discovery loop.

Faithful analogue of TTT-Discover's ``train.do_sync_training`` with the gradient step removed:

  each generation:
    1. PUCT-select ``groups_per_batch`` parents from the buffer      (sampler.sample_states)
    2. for each parent: build prompt = env.get_question() + n-best-solutions context block,
       generate ``group_size`` completions from the frozen model      (vLLM)
    3. grade every completion in the sandbox and feed valid children back into the buffer
       (env.rollout_step -> sampler.update_states / record_failed_rollout)
    4. flush the buffer to disk                                        (sampler.flush)

No weights ever change; improvement comes purely from search + in-context conditioning.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from puct import PUCTSampler, State
from sandbox import init_ray
from envs import EnvConfig, get_problem
from generation import VLLMClient
from context import build_context_block, get_strategy, SelectionParams, dedupe_seeds
from results import ExperimentTracker
from icl.config import ICLConfig

logger = logging.getLogger("icl")


def _setup_logging(log_path: str, console_level: str = "INFO") -> None:
    """Console at ``console_level``; ``icl.log`` always captures DEBUG. Covers the whole ``icl.*``
    namespace so the vendored sandbox/puct debug lines land in the file, not the console."""
    os.makedirs(log_path, exist_ok=True)
    root = logging.getLogger("icl")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_path, "icl.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def _best_native(states: list[State], maximize: bool) -> float | None:
    vals = [s.value for s in states if s.value is not None]
    if not vals:
        return None
    best = max(vals)  # value is stored higher = better
    return best if maximize else -best


class ICLRunner:
    def __init__(self, cfg: ICLConfig):
        self.cfg = cfg
        self.spec = get_problem(cfg.problem)
        self.num_cpus = cfg.num_cpus_per_task or self.spec.num_cpus_per_task
        self.eval_timeout = cfg.eval_timeout or self.spec.eval_timeout

        self.env_config = EnvConfig(
            problem_type=self.spec.problem_type,
            log_path=cfg.log_path,
            num_cpus_per_task=self.num_cpus,
            eval_timeout=self.eval_timeout,
            timeout=cfg.grade_timeout,
        )
        self.llm = VLLMClient(
            base_url=cfg.vllm_base_url,
            model=cfg.model_name,
            reasoning_effort=cfg.reasoning_effort,
            max_concurrency=cfg.max_gen_concurrency,
        )
        self._select = get_strategy(cfg.context_strategy)   # context-selection strategy fn
        self._select_params = SelectionParams(
            mix_fraction=cfg.mix_fraction,
            mmr_lambda=cfg.mmr_lambda,
            jump_alpha=cfg.jump_alpha,
            context_seed=cfg.context_seed,
        )
        self.sampler: PUCTSampler | None = None  # created in run() after init_ray
        self.tracker: ExperimentTracker | None = None
        self._gen_latencies: list[float] = []    # per-group generate() latencies, reset each generation

    def _make_sampler(self, file_path: str) -> PUCTSampler:
        cfg, spec = self.cfg, self.spec
        return PUCTSampler(
            file_path=file_path,
            env_type=spec.env_type,
            problem_type=spec.problem_type,
            max_buffer_size=cfg.max_buffer_size,
            batch_size=cfg.groups_per_batch,
            resume_step=cfg.resume_step,
            puct_c=cfg.puct_c,
            topk_children=cfg.topk_children,
        )

    def _build_prompt(self, env, parent: State):
        """Assemble the full prompt for one parent: base question + selected-solutions context block.

        Returns (prompt, selection, base_prompt, block) where ``selection`` is a SelectionResult.
        """
        cfg, spec = self.cfg, self.spec
        base_prompt = env.get_question()
        # The buffer seeds with groups_per_batch identical seed copies. Collapse them to one logical
        # seed, and drop the seed entirely when this rollout starts *from* the seed (all of gen 0),
        # so the seed is never duplicated nor shown as "past experience" for the state we started at.
        initial_ids = {s.id for s in self.sampler._initial_states}
        pool = dedupe_seeds(self.sampler._states, initial_ids, drop_initial=parent.id in initial_ids)
        selection = self._select(pool, cfg.n_context, self._select_params, exclude_id=parent.id)
        block = build_context_block(
            selection,
            metric_name=spec.metric_name,
            maximize=spec.maximize,
            max_context_tokens=cfg.max_context_tokens,
            include_code=cfg.include_code,
            include_strategy=cfg.include_strategy,
        )
        return base_prompt + block, selection, base_prompt, block

    async def _run_group(self, gen: int, slot: int, parent: State) -> list:
        cfg, spec = self.cfg, self.spec
        env = spec.env_type(initial_state=parent, sampler=self.sampler, config=self.env_config)
        prompt, selection, _base, _block = self._build_prompt(env, parent)

        k, N = len(selection.all()), cfg.n_context
        shortfall = "" if k >= N else " (buffer filling)"
        logger.info(f"gen {gen} p{slot}: prompting LLM (n={cfg.group_size}, "
                    f"context={k}/{N}{shortfall}, prompt~{len(prompt)//4} tok)")

        t0 = time.perf_counter()
        try:
            completions = await self.llm.generate(
                prompt, n=cfg.group_size, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            )
        except Exception as e:
            logger.warning(f"gen {gen} p{slot}: generation FAILED: {e}")
            if self.tracker is not None:
                self.tracker.record_group(gen, slot, parent, prompt, [], [])
            return []
        gen_dt = time.perf_counter() - t0
        self._gen_latencies.append(gen_dt)
        logger.info(f"gen {gen} p{slot}: {len(completions)} completions in {gen_dt:.1f}s -> grading")

        results = await asyncio.gather(*[env.rollout_step(c, gen) for c in completions])
        if self.tracker is not None:
            self.tracker.record_group(gen, slot, parent, prompt, completions, results)

        valid = [r for r in results if r.correctness > 0 and r.next_state is not None]
        best = max(valid, key=lambda r: r.next_state.value) if valid else None
        best_str = "n/a" if best is None else f"{best.raw_score:.6f}"
        logger.info(f"gen {gen} p{slot}: {len(valid)}/{len(results)} valid, "
                    f"best {spec.metric_name}={best_str}")
        return results

    async def run(self) -> None:
        cfg, spec = self.cfg, self.spec
        _setup_logging(cfg.log_path, cfg.log_level)

        n_cand = cfg.groups_per_batch * cfg.group_size
        gen_par = min(cfg.groups_per_batch, cfg.max_gen_concurrency)
        logger.info(f"ICL run: problem={cfg.problem} model={cfg.model_name} strategy={cfg.context_strategy} "
                    f"n_context={cfg.n_context}")
        logger.info(f"shape: {cfg.groups_per_batch} parents x {cfg.group_size} candidates "
                    f"= {n_cand} candidates/generation, {cfg.num_generations} generations")
        logger.info(f"throughput levers: generation parallelism={gen_par} "
                    f"(groups_per_batch vs max_gen_concurrency={cfg.max_gen_concurrency}); "
                    f"grading parallelism ~= host_cpus // num_cpus_per_task (={self.num_cpus}); "
                    f"eval_timeout={self.eval_timeout}s")

        if spec.env_type.uses_sandbox:
            init_ray(self.num_cpus)
        else:
            logger.info("skipping Ray init: problem uses an in-process (sandbox-free) evaluator")
        # Tracker first: it creates the run-dir layout (incl. buffer/) the sampler writes into.
        self.tracker = ExperimentTracker(cfg.log_path, cfg.to_dict(), spec, cfg.save_completions)
        self.sampler = self._make_sampler(os.path.join(cfg.log_path, "buffer", "puct_sampler.json"))

        try:
            start = cfg.resume_step or 0
            for gen in range(start, cfg.num_generations):
                t_gen = time.perf_counter()
                self._gen_latencies = []
                parents = self.sampler.sample_states(cfg.groups_per_batch)
                logger.info(f"gen {gen}/{cfg.num_generations - 1} | sampling {len(parents)} parents "
                            f"(buffer={len(self.sampler._states)})")
                self.tracker.start_generation(gen, parents)

                group_results = await asyncio.gather(
                    *[self._run_group(gen, slot, p) for slot, p in enumerate(parents)]
                )
                self.sampler.flush(step=gen + 1)
                self.tracker.end_generation(gen, self.sampler)

                n_valid = sum(1 for group in group_results for r in group if r.correctness > 0)
                n_total = sum(len(group) for group in group_results)
                pct = (100 * n_valid / n_total) if n_total else 0.0
                best = _best_native(self.sampler._states, spec.maximize)
                stats = self.sampler.get_sample_stats()
                gen_wall = time.perf_counter() - t_gen
                gen_latency = max(self._gen_latencies) if self._gen_latencies else 0.0
                logger.info(
                    f"gen {gen}/{cfg.num_generations - 1} done | valid {n_valid}/{n_total} ({pct:.0f}%) "
                    f"| buffer {stats.get('puct/buffer_size')} | puct_expansions {stats.get('puct/T')} "
                    f"| best {spec.metric_name}={'n/a' if best is None else f'{best:.6f}'} "
                    f"| {gen_wall:.1f}s (generate {gen_latency:.1f}s)"
                )
        except BaseException:
            if self.tracker is not None:
                self.tracker.close(status="failed")
            raise
        else:
            self.tracker.close(status="complete")
        logger.info("ICL run complete.")

    def dry_run(self) -> str:
        """Build and print one fully-assembled prompt for the first PUCT-selected parent, then stop.

        No ray, no model server, no side effects (uses a throwaway sampler file).
        """
        import tempfile

        cfg, spec = self.cfg, self.spec
        with tempfile.TemporaryDirectory() as td:
            self.sampler = self._make_sampler(os.path.join(td, "puct_sampler.json"))
            parent = self.sampler.sample_states(cfg.groups_per_batch)[0]
            env = spec.env_type(initial_state=parent, sampler=self.sampler, config=self.env_config)
            prompt, selection, base_prompt, block = self._build_prompt(env, parent)

        ctx_states = selection.all()
        approx_tokens = len(prompt) // 4
        bar = "=" * 80
        print(f"{bar}\nDRY RUN — problem={cfg.problem} model={cfg.model_name}\n{bar}")
        print(prompt)
        print(bar)
        print(f"strategy                   : {cfg.context_strategy}  "
              f"(include_code={cfg.include_code}, include_strategy={cfg.include_strategy})")
        print(f"context solutions injected : {len(ctx_states)}  "
              f"(positives={len(selection.positives)}, negatives={len(selection.negatives)}, "
              f"n_context={cfg.n_context})")
        print(f"base prompt chars          : {len(base_prompt)}")
        print(f"context block chars        : {len(block)}")
        print(f"total prompt chars         : {len(prompt)}  (~{approx_tokens} tokens @ 4 chars/tok)")
        if not ctx_states:
            print("\nNOTE: no context solutions yet — this is generation 0, so the buffer holds only")
            print("the seed (which is the parent already shown above). Below is an ILLUSTRATIVE render")
            print("of what the context block will look like once the buffer has solutions:")
            print(build_context_block([parent], metric_name=spec.metric_name, maximize=spec.maximize,
                                      include_code=cfg.include_code, include_strategy=cfg.include_strategy))
        return prompt


async def run(cfg: ICLConfig) -> None:
    await ICLRunner(cfg).run()


def dry_run(cfg: ICLConfig) -> str:
    return ICLRunner(cfg).dry_run()
