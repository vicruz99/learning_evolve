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
import json
import logging
import os
import time

from puct import PUCTSampler, State
from sandbox import init_ray
from envs import EnvConfig, get_problem
from generation import GenResult, VLLMClient
from context import build_context_block, get_strategy, SelectionParams
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


def _sum_usage(results: list[GenResult]) -> dict:
    """Sum token accounting over vLLM requests.

    ``prompt_tokens`` is counted per *request*, so with chunking a parent's prompt is counted once
    per chunk — which is the truth we want to see: those are the tokens the server actually had to
    (re)process, and ``cached_prompt_tokens`` shows how many of them the prefix cache served for free.
    """
    return {
        "requests": len(results),
        "completions": sum(len(r) for r in results),
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "cached_prompt_tokens": sum(r.cached_prompt_tokens for r in results),
        "completion_tokens": sum(r.completion_tokens for r in results),
        "reasoning_tokens": sum(r.reasoning_tokens for r in results),
        # Char count, not tokens: servers that expose the reasoning text often omit
        # completion_tokens_details.reasoning_tokens, so this is the reliable volume signal.
        "reasoning_chars": sum(len(x) for r in results for x in r.reasonings),
        "truncated": sum(r.truncated for r in results),
    }


def _percentiles(values: list[int]) -> dict[str, int]:
    """Per-candidate decode-length distribution. The MEAN is the wrong statistic for sizing
    ``--max-tokens``: a vLLM request returns only when its slowest sequence finishes, so the tail is
    what gates a generation. Nearest-rank (no interpolation) — these are token counts, not estimates."""
    if not values:
        return {}
    ordered = sorted(values)
    def at(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    return {"p50": at(0.50), "p90": at(0.90), "p99": at(0.99), "max": ordered[-1]}


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Delta of the server's cumulative prefix-cache counters over one generation. Empty if either
    snapshot is missing (metrics unavailable), so downstream code just sees no cache fields."""
    if not before or not after:
        return {}
    return {k: max(0, after.get(k, 0) - v) for k, v in before.items()}


def _format_usage(u: dict, wall: float) -> str:
    """One-line token report for a generation: where the compute went and how much was cached.

    ``decode`` is the aggregate decode rate over the whole generation (the number to compare against
    single-sequence throughput to see how much batching is buying). ``cached`` is the share of prompt
    tokens the prefix cache served — the direct measure of whether the prompt layout is paying off.
    """
    comps = u["completions"] or 1
    pt, ct = u["prompt_tokens"], u["completion_tokens"]
    # Prefer the /metrics counters (queried/hit prompt tokens); fall back to usage.cached_tokens,
    # which some vLLM builds leave at 0 even with prefix caching on.
    q, h = u.get("cache_queries", 0), u.get("cache_hits", 0)
    hit = (100.0 * h / q) if q else ((100.0 * u["cached_prompt_tokens"] / pt) if pt else 0.0)
    parts = [
        f"prompt {pt:,} ({hit:.0f}% cached)",
        f"decode {ct:,} ({ct / comps:.0f}/completion, {ct / max(wall, 1e-9):.0f} tok/s)",
    ]
    if u["reasoning_tokens"]:
        rt = u["reasoning_tokens"]
        parts.append(f"reasoning {rt:,} + answer {max(0, ct - rt):,} "
                     f"({100.0 * rt / max(ct, 1):.0f}% of decode is reasoning)")
    elif u.get("reasoning_chars"):
        # Reasoning text captured but neither the server nor /tokenize gave us a count: fall back to a
        # chars/4 estimate, clearly marked, so the share of decode is still visible.
        est = u["reasoning_chars"] // 4
        parts.append(f"reasoning ~{est:,} est ({100.0 * est / max(ct, 1):.0f}% of decode)")
    parts.append(f"truncated {u['truncated']}/{u['completions']}")
    return " | ".join(parts)


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
            thinking_token_budget=cfg.thinking_token_budget,
            enable_thinking=cfg.enable_thinking,
            max_concurrency=cfg.max_gen_concurrency,
        )
        self._select = get_strategy(cfg.context_strategy)   # context-selection strategy fn
        self._select_params = SelectionParams(
            mix_fraction=cfg.mix_fraction,
            mmr_lambda=cfg.mmr_lambda,
            jump_alpha=cfg.jump_alpha,
            context_seed=cfg.context_seed if cfg.context_seed is not None else cfg.seed,
        )
        self.sampler: PUCTSampler | None = None  # created in run() after init_ray
        self.tracker: ExperimentTracker | None = None
        self._gen_latencies: list[float] = []    # per-group generate() latencies, reset each generation
        self._gen_results: list[GenResult] = []  # every vLLM request of the current generation (token accounting)
        self._gen_decode: list[int] = []          # per-candidate decode tokens this generation (percentiles)
        self._reasoning_warned = False            # "server exposes no reasoning" warning fires at most once
        # Context pool: EVERY valid solution graded in prior generations, not just the PUCT top-k buffer.
        # Context selection draws from here so strategies (best_worst, contrastive, ...) can see genuine
        # low-scoring negatives that `topk_children` prunes out of the buffer. PUCT search is untouched.
        self._context_pool: list[State] = []
        self._pool_fh = None                      # append-only JSONL mirror (for --resume-step)

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
            rng_seed=cfg.seed,
        )

    def _sample_parents(self, n: int) -> list[State]:
        """This generation's parents: PUCT-selected from the buffer, or always the seed (Best-of-N)."""
        if self.cfg.parent_source == "initial":
            return self.sampler.sample_initial_states(n)
        return self.sampler.sample_states(n)

    def _build_prompt(self, env, parent: State):
        """Assemble the full prompt for one parent, in three cache-friendly zones:

            [intro] + [ICL context block] + [rules + current-solution-to-improve]

        The context block is woven BETWEEN the constant intro (``env.problem_intro()``) and the
        rules/current-solution tail (``env.improvement_task()``). Because the current solution is
        rendered last inside the tail, everything before it (intro + block + rules) is a shared
        prefix across the generation's parents, so vLLM only re-prefills the trailing solution.

        Returns (prompt, selection, intro, tail, block) where ``selection`` is a SelectionResult.
        """
        cfg, spec = self.cfg, self.spec
        intro = env.problem_intro()
        tail = env.improvement_task()
        # Select context from the pool of ALL valid solutions graded in previous generations (built in
        # run()), NOT from the PUCT top-k buffer: the buffer holds only high-scoring survivors, so
        # strategies like best_worst/contrastive would never see genuine low-scoring negatives. Seeds
        # never enter this pool (they produce no graded solution), so no seed de-duplication is needed,
        # and generation 0 sees an empty pool -> an empty context block.
        #
        # exclude_id is what makes the block differ per parent (each parent drops itself, shifting a
        # different solution in). With exclude_parent_from_context=False it is the same block for every
        # parent -> a shared prefix vLLM prefills once per generation. See ICLConfig.
        exclude_id = parent.id if cfg.exclude_parent_from_context else None
        selection = self._select(self._context_pool, cfg.n_context, self._select_params,
                                 exclude_id=exclude_id)
        block = build_context_block(
            selection,
            metric_name=spec.metric_name,
            maximize=spec.maximize,
            max_context_tokens=cfg.max_context_tokens,
            include_code=cfg.include_code,
            include_strategy=cfg.include_strategy,
        )
        return intro + block + tail, selection, intro, tail, block

    def _open_context_pool(self, path: str, resume: bool = False) -> None:
        """Open the append-only context-pool log. On resume, reload prior valid solutions into memory
        first (so context is complete from the first resumed generation); otherwise start fresh."""
        if resume and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._context_pool.append(State.from_dict(json.loads(line)))
            logger.info(f"resumed context pool: {len(self._context_pool)} valid solutions from {path}")
            self._pool_fh = open(path, "a")
        else:
            self._pool_fh = open(path, "w")

    def _extend_context_pool(self, states: list[State]) -> None:
        """Add a generation's valid solutions to the context pool (in-memory + on-disk mirror)."""
        for s in states:
            self._context_pool.append(s)
            self._pool_fh.write(json.dumps(s.to_dict()) + "\n")
        self._pool_fh.flush()

    def _chunk_sizes(self, total: int) -> list[int]:
        """Split ``total`` completions into per-request chunk sizes. ``grade_chunk_size`` None/0 ->
        one chunk of ``total`` (original behavior); K -> chunks of K (last one smaller)."""
        k = self.cfg.grade_chunk_size or total
        k = max(1, min(k, total))
        sizes, rem = [], total
        while rem > 0:
            sizes.append(min(k, rem))
            rem -= sizes[-1]
        return sizes

    async def _run_group(self, gen: int, slot: int, parent: State) -> list:
        cfg, spec = self.cfg, self.spec
        env = spec.env_type(initial_state=parent, sampler=self.sampler, config=self.env_config)
        prompt, selection, _intro, _tail, _block = self._build_prompt(env, parent)

        k, N = len(selection.all()), cfg.n_context
        shortfall = "" if k >= N else " (buffer filling)"
        sizes = self._chunk_sizes(cfg.group_size)
        chunk_note = "" if len(sizes) == 1 else f", {len(sizes)}x chunks<={sizes[0]} grade-as-ready"
        logger.info(f"gen {gen} p{slot}: prompting LLM (n={cfg.group_size}{chunk_note}, "
                    f"context={k}/{N}{shortfall}, prompt~{len(prompt)//4} tok)")

        t0 = time.perf_counter()
        multi_chunk = len(sizes) > 1
        graded_done = 0  # completions graded so far in this group (across concurrent chunks)

        async def _gen_grade_chunk(sz: int):
            """Generate ``sz`` completions in one request, then grade them the moment they arrive.
            Running these coroutines concurrently overlaps one chunk's grading (CPU/sandbox) with
            another chunk's still-in-flight generation (GPU)."""
            nonlocal graded_done
            gen_res = await self.llm.generate(
                prompt, n=sz, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            )
            self._gen_results.append(gen_res)
            comps = gen_res.texts
            if multi_chunk:
                logger.info(f"gen {gen} p{slot}: chunk returned ({sz} completion(s)), grading "
                            f"[{graded_done}/{cfg.group_size} graded so far]")
            res = await asyncio.gather(*[env.rollout_step(c, gen) for c in comps])
            if multi_chunk:
                graded_done += len(res)
                nv = sum(1 for r in res if r.correctness > 0 and r.next_state is not None)
                logger.info(f"gen {gen} p{slot}: chunk graded ({nv}/{len(res)} valid) "
                            f"[{graded_done}/{cfg.group_size} graded]")
            return gen_res, res

        try:
            chunk_out = await asyncio.gather(*[_gen_grade_chunk(sz) for sz in sizes])
        except Exception as e:
            logger.warning(f"gen {gen} p{slot}: generation FAILED: {e}")
            if self.tracker is not None:
                self.tracker.record_group(gen, slot, parent, prompt, [], [])
            return []

        gen_results = [g for g, _ in chunk_out]
        completions = [c for g, _ in chunk_out for c in g.texts]
        reasonings = [r for g, _ in chunk_out for r in g.reasonings]
        finish_reasons = [f for g, _ in chunk_out for f in g.finish_reasons]
        results = [r for _, res in chunk_out for r in res]
        grp_dt = time.perf_counter() - t0
        self._gen_latencies.append(grp_dt)
        # The server does not break completion_tokens into reasoning vs answer, so count the captured
        # reasoning text with its own tokenizer. Written back onto each GenResult so the
        # generation-level _sum_usage reports real tokens, not an estimate.
        reasoning_tokens, answer_tokens = await self._count_decode_tokens(
            gen_results, reasonings, completions)
        # Per-candidate decode totals feed this generation's percentiles (the --max-tokens signal).
        self._gen_decode.extend(r + a for r, a in zip(reasoning_tokens, answer_tokens))
        if self.tracker is not None:
            self.tracker.record_group(gen, slot, parent, prompt, completions, results,
                                      reasonings=reasonings, finish_reasons=finish_reasons,
                                      reasoning_tokens=reasoning_tokens,
                                      answer_tokens=answer_tokens,
                                      usage=_sum_usage(gen_results))

        valid = [r for r in results if r.correctness > 0 and r.next_state is not None]
        best = max(valid, key=lambda r: r.next_state.value) if valid else None
        best_str = "n/a" if best is None else f"{best.raw_score:.6f}"
        logger.info(f"gen {gen} p{slot}: {len(completions)} gen+graded in {grp_dt:.1f}s | "
                    f"{len(valid)}/{len(results)} valid, best {spec.metric_name}={best_str}")
        return results

    async def _count_decode_tokens(self, gen_results: list[GenResult], reasonings: list[str],
                                   completions: list[str]) -> tuple[list[int], list[int]]:
        """Exact per-candidate (reasoning_tokens, answer_tokens), counted with the served model's own
        tokenizer via the server's ``/tokenize``.

        The API reports ``usage`` per *request*, so per-candidate decode cost is otherwise invisible —
        yet that is exactly what sets ``--max-tokens``, because a request returns only when its
        slowest sequence finishes. Measured cost: ~0.7 s per 90-candidate generation (CPU-only on the
        API server, off the GPU, and overlapped with grading), i.e. ~0.1 % of a generation.

        Reasoning totals are mirrored onto ``gen_results`` so the generation-level ``_sum_usage``
        reports real tokens; if a server ever reports ``reasoning_tokens`` itself, that authoritative
        value wins and is left untouched.
        """
        if not any(reasonings) and not any(completions):
            return [], []
        # One gather for both lists: the calls are concurrent anyway, so this is a single round of work.
        counts = await self.llm.count_tokens(reasonings + completions)
        if not counts:
            return [], []
        split = len(reasonings)
        r_counts, a_counts = counts[:split], counts[split:]
        i = 0
        for g in gen_results:
            n = len(g.texts)
            if not g.reasoning_tokens:
                g.reasoning_tokens = sum(r_counts[i:i + n])
            i += n
        return r_counts, a_counts

    def _warn_if_no_reasoning(self, usage: dict) -> None:
        """Warn once if a whole generation came back with no reasoning text while we were asked to save it.

        Almost always means the server has no reasoning parser for this model, which is not merely a
        lost trace: the chain of thought then stays inside ``content``, and code extraction takes the
        LAST ```python fence in ``content`` — so a completion that fences code while thinking but not
        in its final answer silently yields code parsed out of the *reasoning*. Cheaper to catch here
        than to debug as mysterious `invalid_result`s a generation later.
        """
        if self._reasoning_warned or not self.cfg.save_reasoning:
            return
        if not usage.get("completions") or usage.get("reasoning_chars"):
            return
        self._reasoning_warned = True
        logger.warning(
            f"--save-reasoning is on but all {usage['completions']} completions of this generation "
            f"returned NO reasoning text (decode was {usage['completion_tokens']:,} tokens, so the "
            f"model is reasoning — it just isn't being separated out). Launch the vLLM server with a "
            f"reasoning parser for this model (gpt-oss: --reasoning-parser openai_gptoss; Qwen3: "
            f"--reasoning-parser qwen3). Until then the chain of thought stays inside the answer text, "
            f"which can make code extraction pick up a fence from the reasoning. "
            f"Pass --no-save-reasoning to silence this.")

    async def run(self) -> None:
        cfg, spec = self.cfg, self.spec
        _setup_logging(cfg.log_path, cfg.log_level)

        n_cand = cfg.groups_per_batch * cfg.group_size
        gen_par = min(cfg.groups_per_batch, cfg.max_gen_concurrency)
        logger.info(f"ICL run: problem={cfg.problem} model={cfg.model_name} strategy={cfg.context_strategy} "
                    f"n_context={cfg.n_context} seed={cfg.seed}")
        if cfg.parent_source == "initial":
            logger.info("parents: ALWAYS the seed solution (--parent-source initial) -> Best-of-N; the "
                        "buffer is still recorded but never read to pick a parent"
                        + ("" if cfg.n_context else " and no context is injected: no past experience "
                                                    "reaches the model at all"))
        else:
            logger.info("parents: PUCT-selected from the buffer"
                        + ("" if cfg.n_context else " (no context injected: past experience reaches the "
                                                    "model only through which parent it is given)"))
        # Whether a generation's parents share one context block decides whether vLLM prefills that
        # block once per generation or once per parent -- worth seeing at a glance, since at n_context=20
        # the block is ~16k of a ~17k prompt.
        if cfg.n_context:
            if cfg.exclude_parent_from_context:
                logger.info("context block: per-parent (each parent excludes itself) -> no cross-parent "
                            "prefix reuse; pass --no-exclude-parent --context-seed N to share it")
            elif cfg.context_seed is None:
                logger.info("context block: shared selection but tie-breaking is random per parent "
                            "-> blocks may still differ; add --context-seed N to make them identical")
            else:
                logger.info("context block: IDENTICAL across parents (--no-exclude-parent + "
                            "--context-seed) -> prefilled once per generation")
        logger.info(f"shape: {cfg.groups_per_batch} parents x {cfg.group_size} candidates "
                    f"= {n_cand} candidates/generation, {cfg.num_generations} generations")
        logger.info(f"throughput levers: generation parallelism={gen_par} "
                    f"(groups_per_batch vs max_gen_concurrency={cfg.max_gen_concurrency}); "
                    f"grading parallelism ~= host_cpus // num_cpus_per_task (={self.num_cpus}); "
                    f"eval_timeout={self.eval_timeout}s")
        # With chunked grade-as-ready there are groups_per_batch * chunks_per_group concurrent gen
        # requests; if max_gen_concurrency is below that, the semaphore throttles them and vLLM can't
        # co-batch all sequences -> generation serializes into waves and gets much slower.
        n_chunks = len(self._chunk_sizes(cfg.group_size))
        if n_chunks > 1:
            reqs = cfg.groups_per_batch * n_chunks
            if cfg.max_gen_concurrency < reqs:
                logger.warning(
                    f"grade_chunk_size={cfg.grade_chunk_size}: {reqs} concurrent gen requests/generation "
                    f"({cfg.groups_per_batch} parents x {n_chunks} chunks) but max_gen_concurrency="
                    f"{cfg.max_gen_concurrency} -> vLLM will NOT co-batch them all (generation "
                    f"throttled into waves). Set --max-gen-concurrency >= {reqs} to keep it parallel.")

        if spec.env_type.uses_sandbox:
            init_ray(self.num_cpus)
        else:
            logger.info("skipping Ray init: problem uses an in-process (sandbox-free) evaluator")
        # Tracker first: it creates the run-dir layout (incl. buffer/) the sampler writes into.
        self.tracker = ExperimentTracker(cfg.log_path, cfg.to_dict(), spec, cfg.save_completions,
                                         cfg.save_reasoning)
        self.sampler = self._make_sampler(os.path.join(cfg.log_path, "buffer", "puct_sampler.json"))
        self._open_context_pool(os.path.join(cfg.log_path, "buffer", "context_pool.jsonl"),
                                resume=bool(cfg.resume_step))

        try:
            start = cfg.resume_step or 0
            for gen in range(start, cfg.num_generations):
                t_gen = time.perf_counter()
                self._gen_latencies = []
                self._gen_results = []
                self._gen_decode = []
                cache0 = await self.llm.cache_counters()
                parents = self._sample_parents(cfg.groups_per_batch)
                logger.info(f"gen {gen}/{cfg.num_generations - 1} | sampling {len(parents)} parents "
                            f"(buffer={len(self.sampler._states)})")
                self.tracker.start_generation(gen, parents)

                group_results = await asyncio.gather(
                    *[self._run_group(gen, slot, p) for slot, p in enumerate(parents)]
                )
                self.sampler.flush(step=gen + 1)
                # Feed this generation's valid solutions into the context pool for LATER generations
                # (done after the whole generation so all parents in a generation share one snapshot).
                new_valid = [r.next_state for group in group_results for r in group
                             if r.correctness > 0 and r.next_state is not None]
                self._extend_context_pool(new_valid)
                usage = _sum_usage(self._gen_results)
                usage.update(_counter_delta(cache0, await self.llm.cache_counters()))
                gen_wall = time.perf_counter() - t_gen
                decode_pct = _percentiles(self._gen_decode)
                self.tracker.end_generation(gen, self.sampler, usage=usage, wall_seconds=gen_wall,
                                            decode_percentiles=decode_pct)

                n_valid = sum(1 for group in group_results for r in group if r.correctness > 0)
                n_total = sum(len(group) for group in group_results)
                pct = (100 * n_valid / n_total) if n_total else 0.0
                best = _best_native(self.sampler._states, spec.maximize)
                stats = self.sampler.get_sample_stats()
                gen_latency = max(self._gen_latencies) if self._gen_latencies else 0.0
                logger.info(
                    f"gen {gen}/{cfg.num_generations - 1} done | valid {n_valid}/{n_total} ({pct:.0f}%) "
                    f"| buffer {stats.get('puct/buffer_size')} | ctx_pool {len(self._context_pool)} "
                    f"| puct_expansions {stats.get('puct/T')} "
                    f"| best {spec.metric_name}={'n/a' if best is None else f'{best:.6f}'} "
                    f"| {gen_wall:.1f}s (generate {gen_latency:.1f}s)"
                )
                logger.info(f"gen {gen} tokens | {_format_usage(usage, gen_wall)}")
                if decode_pct:
                    headroom = 100.0 * decode_pct["max"] / cfg.max_tokens
                    logger.info(
                        f"gen {gen} decode/candidate | p50 {decode_pct['p50']:,} p90 "
                        f"{decode_pct['p90']:,} p99 {decode_pct['p99']:,} max {decode_pct['max']:,} "
                        f"| max_tokens={cfg.max_tokens:,} ({headroom:.0f}% used by the longest) "
                        f"— size --max-tokens off p99, the tail gates each request")
                self._warn_if_no_reasoning(usage)
                if usage["truncated"]:
                    logger.warning(
                        f"gen {gen}: {usage['truncated']}/{usage['completions']} completions hit the "
                        f"max_tokens={cfg.max_tokens} cap (finish_reason=length) — these burn the full "
                        f"budget, usually emit no code block, and gate their request's return. "
                        f"Consider lowering --reasoning-effort or raising --max-tokens.")
        except BaseException:
            if self.tracker is not None:
                self.tracker.close(status="failed")
            raise
        else:
            self.tracker.close(status="complete")
        finally:
            if self._pool_fh is not None:
                self._pool_fh.close()
        logger.info("ICL run complete.")

    def dry_run(self) -> str:
        """Build and print one fully-assembled prompt for the first PUCT-selected parent, then stop.

        No ray, no model server, no side effects (uses a throwaway sampler file).
        """
        import tempfile

        cfg, spec = self.cfg, self.spec
        with tempfile.TemporaryDirectory() as td:
            self.sampler = self._make_sampler(os.path.join(td, "puct_sampler.json"))
            parent = self._sample_parents(cfg.groups_per_batch)[0]
            env = spec.env_type(initial_state=parent, sampler=self.sampler, config=self.env_config)
            prompt, selection, intro, tail, block = self._build_prompt(env, parent)

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
        print(f"intro (zone 1) chars       : {len(intro)}  [constant, before context block]")
        print(f"context block chars        : {len(block)}  [woven between intro and rules]")
        print(f"rules+solution (zone 3)    : {len(tail)}  [current solution rendered last]")
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
