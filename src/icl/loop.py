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
from sandbox import memwatch
from envs import EnvConfig, get_problem
from generation import GenResult, VLLMClient
from context import build_context_block, get_strategy, SelectionParams
from results import ExperimentTracker
from icl.config import ICLConfig

logger = logging.getLogger("icl")


class GenerationAborted(RuntimeError):
    """A generation could not be completed, so the run stops here rather than walking past it.

    Raised once every group of the generation has settled (never mid-flight), and always BEFORE
    ``tracker.end_generation``, so no ``meta.json`` is written for it. That is what makes the stop
    resumable: ``results.resume`` reads the partial generation directory as "never finished", puts the
    resume point at the last complete generation, and ``rewind`` moves the partial one aside.

    The alternative, which this replaces, was to catch the error per group, record a group with no
    candidates and continue. The run then finished all its generations, and every one of them after
    the failure was conditioned on a buffer and a context pool missing that group's children — the
    same configuration on paper, a different experiment in fact.
    """


class LLMUnavailable(GenerationAborted):
    """The model server did not come back within ``--llm-max-wait``."""


# Errors that will not fix themselves by waiting: the request is wrong, not the server. Retrying an
# unknown model name or an over-length prompt for an hour only delays finding out.
_PERMANENT_STATUS = {400, 401, 403, 404, 413, 422}


def _is_permanent(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _PERMANENT_STATUS


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
            evaluator_options=cfg.evaluator_options,
            # --parent-source none is a PROMPT setting, so it is the env that has to honour it: it
            # drops the trailing "current solution to improve upon" and the wording that introduces
            # it. Grading is untouched -- the env still gets a state.
            show_parent_solution=cfg.parent_source != "none",
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
        """This generation's parents — see ``ICLConfig.parent_source`` for what each source means.

        ``none`` shares ``initial``'s bookkeeping deliberately: the prompt shows no parent, so the
        children belong to no particular solution, and attributing them to the seed is the only
        choice that leaves the PUCT statistics meaning what they say. What makes ``none`` different
        is on the prompt side (``EnvConfig.show_parent_solution``), not here.
        """
        source = self.cfg.parent_source
        if source in ("initial", "none"):
            return self.sampler.sample_initial_states(n)
        if source == "best":
            return self.sampler.sample_best_states(n)
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
        first (so context is complete from the first resumed generation); otherwise start fresh.

        A resume whose pool file is gone used to fall through to the fresh branch and TRUNCATE it: the
        run continued with an empty pool, so every generation after the resume was prompted with less
        context than the ones before it — invisible in the logs and fatal to the comparison. Refuse
        instead; ``results.resume`` decides which generation still has a pool behind it."""
        if resume and not os.path.exists(path) and self.cfg.n_context > 0:
            raise FileNotFoundError(
                f"--resume-step {self.cfg.resume_step} needs the run's context pool, but {path} does "
                f"not exist. Resuming without it would prompt every later generation with an empty "
                f"context block. Run `python -m results.resume {self.cfg.log_path}` to see the last "
                f"resumable generation (or start the run over).")
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

    async def _generate_waiting(self, prompt: str, n: int, gen: int, slot: int) -> GenResult:
        """``llm.generate``, but a server that is merely *absent* is waited out rather than lost.

        The client already retries transient errors inside one request (``max_retries``); this is the
        layer above that, for the case those retries cannot cover — the server's own job being
        requeued, a node reboot, a restart to change flags. Backs off 5s -> 60s and re-probes
        ``/health`` so the log says whether we are waiting on a dead server or a sick one.

        Gives up after ``cfg.llm_max_wait`` seconds (0 = never) by raising ``LLMUnavailable``, which
        stops the run at its last complete generation.
        """
        waited, delay = 0.0, 5.0
        while True:
            try:
                return await self.llm.generate(prompt, n=n, temperature=self.cfg.temperature,
                                               max_tokens=self.cfg.max_tokens)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if _is_permanent(e):
                    raise LLMUnavailable(
                        f"gen {gen} p{slot}: the server rejected the request and retrying will not "
                        f"change that ({e!r}). Check --model against what the server serves and "
                        f"--max-tokens/--max-context-tokens against its --max-model-len.") from e
                limit = self.cfg.llm_max_wait
                if limit and waited >= limit:
                    raise LLMUnavailable(
                        f"gen {gen} p{slot}: no response from {self.cfg.vllm_base_url} after "
                        f"{waited:.0f}s of retrying ({e!r}). Stopping at the last complete "
                        f"generation — resume this run once the server is back.") from e
                healthy = await self.llm.health()
                budget = "no limit" if not limit else f"{limit - waited:.0f}s left"
                logger.warning(
                    f"gen {gen} p{slot}: model server unreachable ({e!r}); /health "
                    f"{'answers but the request still failed' if healthy else 'does not answer'}. "
                    f"Waiting {delay:.0f}s and retrying — the generation does NOT advance until it "
                    f"comes back ({budget}).")
                await asyncio.sleep(delay)
                waited += delay
                delay = min(delay * 2, 60.0)

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
            gen_res = await self._generate_waiting(prompt, sz, gen, slot)
            self._gen_results.append(gen_res)
            comps = gen_res.texts
            if multi_chunk:
                logger.info(f"gen {gen} p{slot}: chunk returned ({sz} completion(s)), grading "
                            f"[{graded_done}/{cfg.group_size} graded so far]")
            # Grading failures are kept distinct from generation ones: a dead Ray worker used to be
            # recorded as "the LLM was unreachable", and re-generating on a grading failure would pay
            # for the decode twice. Neither is retried here -- both abort the generation.
            try:
                res = await asyncio.gather(*[env.rollout_step(c, gen) for c in comps])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                raise GenerationAborted(
                    f"gen {gen} p{slot}: grading failed after the model had answered ({e!r}). The "
                    f"candidates are in this generation's directory but were never scored, so the "
                    f"buffer would be missing them; stopping at the last complete generation.") from e
            if multi_chunk:
                graded_done += len(res)
                nv = sum(1 for r in res if r.correctness > 0 and r.next_state is not None)
                logger.info(f"gen {gen} p{slot}: chunk graded ({nv}/{len(res)} valid) "
                            f"[{graded_done}/{cfg.group_size} graded]")
            return gen_res, res

        # No swallow-and-continue here. A group that cannot be completed ends the generation, and the
        # generation is abandoned before meta.json is written, so the run stays resumable at the last
        # complete one. See GenerationAborted.
        chunk_out = await asyncio.gather(*[_gen_grade_chunk(sz) for sz in sizes])

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

    def _failure_mix(self, gen: int) -> dict[str, int]:
        """This generation's failure_type counts, as the tracker recorded them.

        Which failure dominates is what separates the causes of a barren generation: `cpu_starvation`
        says the scheduler ran out of CPU groups, `timeout` says the candidates are too slow for
        --eval-timeout, `invalid_result` says the model's code is wrong. Read off the tracker rather
        than recounted here so the log line and meta.json cannot disagree.
        """
        per_gen = self.tracker._per_gen if self.tracker is not None else []
        for stats in reversed(per_gen):
            if stats.get("generation") == gen:
                return dict(stats.get("failure_types") or {})
        return {}

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
        no_ctx = not cfg.n_context
        if cfg.parent_source == "initial":
            logger.info("parents: ALWAYS the seed solution (--parent-source initial) -> Best-of-N; the "
                        "buffer is still recorded but never read to pick a parent"
                        + (" and no context is injected: no past experience reaches the model at all"
                           if no_ctx else ""))
        elif cfg.parent_source == "best":
            logger.info("parents: ALWAYS the buffer's best-so-far solution (--parent-source best) -> "
                        "greedy hill-climbing; every slot of a generation gets the SAME parent, so "
                        "PUCT's exploration term is switched off, not the buffer"
                        + (" and no context is injected: past experience reaches the model only as "
                           "that one best parent" if no_ctx else ""))
        elif cfg.parent_source == "none":
            logger.info("parents: NONE (--parent-source none) -> the prompt shows no current solution "
                        "to improve upon, only the objective and the target; children are attributed "
                        "to the seed"
                        + (" and no context is injected: the model gets NOTHING but the problem "
                           "statement (from-scratch zero-shot arm)" if no_ctx else
                           " -> past experience reaches the model through the context block and "
                           "nothing else"))
        else:
            logger.info("parents: PUCT-selected from the buffer"
                        + (" (no context injected: past experience reaches the model only through "
                           "which parent it is given)" if no_ctx else ""))
        # Whether a generation's parents share one context block decides whether vLLM prefills that
        # block once per generation or once per parent -- worth seeing at a glance, since at n_context=20
        # the block is ~16k of a ~17k prompt.
        if cfg.n_context:
            # Self-exclusion is what usually makes each parent's block different. It cannot, when
            # every slot of a generation gets the same parent (best) or a fresh seed that is not in
            # the context pool at all (initial / none) -- then every parent drops the same id, or
            # none, and the block is shared whatever --exclude-parent says.
            if cfg.exclude_parent_from_context and cfg.parent_source == "puct":
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
            init_ray(self.num_cpus, ray_num_cpus=cfg.ray_num_cpus)
        else:
            logger.info("skipping Ray init: problem uses an in-process (sandbox-free) evaluator")
        # Tracker first: it creates the run-dir layout (incl. buffer/) the sampler writes into.
        self.tracker = ExperimentTracker(cfg.log_path, cfg.to_dict(), spec, cfg.save_completions,
                                         cfg.save_reasoning, resume_step=cfg.resume_step)
        self.sampler = self._make_sampler(os.path.join(cfg.log_path, "buffer", "puct_sampler.json"))
        self._open_context_pool(os.path.join(cfg.log_path, "buffer", "context_pool.jsonl"),
                                resume=bool(cfg.resume_step))

        # Resolved once: detection can shell out to `bjobs`, and the ceiling cannot change under a
        # running job. A run that cannot find a ceiling still logs its own RSS every generation --
        # the absolute number is the useful half, and it is what a post-mortem has to work with.
        mem_limit, mem_limit_where = memwatch.ceiling()
        if mem_limit:
            action = (f"stopping cleanly above {100 * cfg.memory_stop_fraction:.0f}%"
                      if cfg.memory_stop_fraction else "self-stop disabled")
            logger.info(f"memory ceiling {mem_limit / memwatch.GiB:.0f}G "
                        f"({mem_limit_where}); {action}")
        else:
            logger.info(f"memory ceiling: {mem_limit_where} — logging job RSS per generation anyway, "
                        "but nothing can warn before the batch system kills this job")
        # Said once, not on every generation's line: how to read the two numbers that follow.
        logger.info("reading the per-generation `memory` line: job rss is the process-tree total the "
                    "batch system kills on; per-eval peak is one candidate's own high-water mark. "
                    "Job climbing while per-eval stays flat = accumulation in the long-lived "
                    "processes (Ray workers, driver), not a greedy candidate.")

        stopped_for_memory = False
        stopped_for_no_yield = False
        empty_streak = 0                     # consecutive generations that graded nothing valid
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

                # return_exceptions: let every group settle before deciding. Without it the first
                # failure unwinds run() while its sibling groups are still generating and grading in
                # the background -- evals left running against a Ray head the process is about to
                # abandon, and a partial generation dir still growing while resume inspects it.
                settled = await asyncio.gather(
                    *[self._run_group(gen, slot, p) for slot, p in enumerate(parents)],
                    return_exceptions=True,
                )
                failures = [r for r in settled if isinstance(r, BaseException)]
                if failures:
                    for f in failures:
                        logger.error(f"gen {gen}: {f}")
                    logger.error(
                        f"gen {gen}: {len(failures)}/{len(parents)} group(s) did not complete — "
                        f"ABANDONING this generation instead of recording it with missing groups. "
                        f"Generations 0..{gen - 1} are complete on disk; resume this run to continue "
                        f"from generation {gen}.")
                    first = next((f for f in failures if isinstance(f, GenerationAborted)), failures[0])
                    raise first
                group_results = list(settled)
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
                # Sampled at the boundary, where the previous generation's eval workers have exited
                # and the next generation's have not started: the closest thing to a steady-state
                # reading, and the point a clean stop can happen from.
                mem = memwatch.sample(mem_limit)
                self.tracker.end_generation(gen, self.sampler, usage=usage, wall_seconds=gen_wall,
                                            decode_percentiles=decode_pct, memory=mem)

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
                ev = self.tracker._per_gen[-1].get("eval_percentiles") or {}
                gr = self.tracker._per_gen[-1].get("grade_percentiles") or {}
                if ev:
                    # What fills the grade-eval gap depends on the evaluator: the Ray sandbox queues
                    # for a CPU group and pays pickle/IPC per candidate, while a sandbox-free problem
                    # (envs with uses_sandbox=False, e.g. trimul) queues for whatever device lock its
                    # evaluator holds. Naming the wrong one sends you hunting for Ray contention in a
                    # run that never started Ray.
                    gap_note = ("CPU-group queueing + ~2s/candidate Ray+pickle overhead"
                                if spec.env_type.uses_sandbox else
                                "in-process evaluator queueing (e.g. the GPU lock), not Ray")
                    logger.info(
                        f"gen {gen} eval/candidate | p50 {ev['p50']:.1f}s p90 {ev['p90']:.1f}s "
                        f"p99 {ev['p99']:.1f}s max {ev['max']:.1f}s | eval_timeout="
                        f"{self.eval_timeout}s ({100.0 * ev['max'] / self.eval_timeout:.0f}% used by "
                        f"the slowest) | grade p50 {gr.get('p50', 0):.1f}s max {gr.get('max', 0):.1f}s "
                        f"(grade-eval gap = {gap_note})")
                rss = self.tracker._per_gen[-1].get("rss_percentiles") or {}
                logger.info(
                    f"gen {gen} memory | job {memwatch.describe(mem)}"
                    + (f" | per-eval peak p50 {rss['p50']:.0f}M p99 {rss['p99']:.0f}M "
                       f"max {rss['max']:.0f}M" if rss else ""))
                self._warn_if_no_reasoning(usage)
                if usage["truncated"]:
                    logger.warning(
                        f"gen {gen}: {usage['truncated']}/{usage['completions']} completions hit the "
                        f"max_tokens={cfg.max_tokens} cap (finish_reason=length) — these burn the full "
                        f"budget, usually emit no code block, and gate their request's return. "
                        f"Consider lowering --reasoning-effort or raising --max-tokens.")

                # A generation that graded nothing valid is a broken evaluator, not a hard problem:
                # every candidate reached the model and came back, and every one of them failed. The
                # run used to keep going and record those generations as ordinary — and because they
                # are structurally perfect (full groups, full complement of children, all invalid),
                # results.resume verifies such a run as COMPLETE and --resume skips it. So the sweep
                # hands back a green run holding nothing. Stop, loudly, while the cause is still on
                # the box to look at.
                empty_streak = empty_streak + 1 if n_valid == 0 else 0
                if cfg.max_empty_generations and empty_streak >= cfg.max_empty_generations:
                    top = sorted(self._failure_mix(gen).items(), key=lambda kv: -kv[1])[:3]
                    logger.error(
                        f"gen {gen}: STOPPING — {empty_streak} consecutive generation(s) produced no "
                        f"valid candidate at all ({n_total} candidates each). This is an evaluator "
                        f"fault, not a search result: check the sandbox before trusting anything "
                        f"here. Commonest failure types this generation: "
                        + (", ".join(f"{k}={v}" for k, v in top) or "none recorded") + ". "
                        f"A starved cpu_scheduler (leaked CPU groups), a full disk and a Ray head "
                        f"that lost its workers all look exactly like this. Raise "
                        f"--max-empty-generations if a barren stretch is genuinely expected here.")
                    stopped_for_no_yield = True
                    break

                # Stop ourselves rather than let the batch system decide where to land. Everything
                # this generation produced is already on disk (sampler.flush + end_generation above),
                # so --resume has a whole generation to continue from -- which is not true of a kill
                # that arrives mid-generation.
                if (cfg.memory_stop_fraction and mem.get("job_rss_pct") is not None
                        and mem["job_rss_pct"] >= 100 * cfg.memory_stop_fraction):
                    # job_rss_pct is a JOB-level figure: every run sharing this Ray head sees the
                    # same one and would otherwise stop in the same breath. Arbitrate so the sweep
                    # sheds one run and lets the others use the headroom that frees.
                    shed, why = memwatch.claim_shed(
                        os.path.dirname(os.path.abspath(cfg.log_path)),
                        os.path.basename(os.path.abspath(cfg.log_path)))
                    if not shed:
                        logger.warning(
                            f"gen {gen}: over the {100 * cfg.memory_stop_fraction:.0f}% memory "
                            f"threshold ({memwatch.describe(mem)}) but {why}")
                    else:
                        logger.error(
                            f"gen {gen}: STOPPING — {memwatch.describe(mem)}, at or above the "
                            f"{100 * cfg.memory_stop_fraction:.0f}% self-stop threshold ({why}). "
                            f"Generations 0..{gen} are complete on disk; resume this run to "
                            f"continue. If the job needs more headroom, LSF memory is PER SLOT: "
                            f"`-M 8192MB -R \"rusage[mem=8192]\"`.")
                        stopped_for_memory = True
                        break
        except BaseException as e:
            if self.tracker is not None:
                # "aborted" rather than "failed": the run stopped itself at a generation boundary and
                # everything before it is intact and resumable. Neither status is trusted by
                # results.resume, but the distinction is what a post-mortem reads first.
                self.tracker.close(status="aborted" if isinstance(e, GenerationAborted) else "failed")
            raise
        else:
            # Never "complete": the run stopped short of num_generations on purpose, and a summary
            # claiming otherwise is exactly the lie results.resume was written to stop trusting.
            self.tracker.close(status="stopped_memory" if stopped_for_memory else
                               "stopped_no_yield" if stopped_for_no_yield else "complete")
        finally:
            if self._pool_fh is not None:
                self._pool_fh.close()
        if stopped_for_memory:
            logger.info("ICL run stopped early to stay under the memory ceiling — resume it to continue.")
        elif stopped_for_no_yield:
            logger.info("ICL run stopped: the evaluator returned nothing valid. Fix the cause before "
                        "resuming — resuming into a broken sandbox just reproduces this.")
        else:
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
        print(f"parent source              : {cfg.parent_source}  "
              f"(current solution shown in the prompt: "
              f"{'no' if cfg.parent_source == 'none' else 'yes'})")
        print(f"context solutions injected : {len(ctx_states)}  "
              f"(positives={len(selection.positives)}, negatives={len(selection.negatives)}, "
              f"n_context={cfg.n_context})")
        print(f"intro (zone 1) chars       : {len(intro)}  [constant, before context block]")
        print(f"context block chars        : {len(block)}  [woven between intro and rules]")
        print(f"rules+solution (zone 3)    : {len(tail)}  [current solution rendered last]")
        print(f"total prompt chars         : {len(prompt)}  (~{approx_tokens} tokens @ 4 chars/tok)")
        if not ctx_states:
            print("\nNOTE: no context solutions yet — this is generation 0, so the buffer holds only")
            print("the seed" + ("" if cfg.parent_source == "none" else
                                " (which is the parent already shown above)")
                  + ". Below is an ILLUSTRATIVE render")
            print("of what the context block will look like once the buffer has solutions:")
            print(build_context_block([parent], metric_name=spec.metric_name, maximize=spec.maximize,
                                      include_code=cfg.include_code, include_strategy=cfg.include_strategy))
        return prompt


async def run(cfg: ICLConfig) -> None:
    await ICLRunner(cfg).run()


def dry_run(cfg: ICLConfig) -> str:
    return ICLRunner(cfg).dry_run()
