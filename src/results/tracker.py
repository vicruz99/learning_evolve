"""Experiment results storage & tracking for the ICL harness.

Writes, incrementally per generation, a run directory that is both human-browsable and
machine-readable:

    <run_dir>/
      config.json      run config + git sha / timestamp / host / entrypoint
      summary.json     live: totals, best, worst_valid, per_generation[]  (status running->complete)
      progress.csv     one row per generation
      events.jsonl     one line per candidate (valid + failed)
      solutions/       sol_NNNNNN.py (de-fenced model code + header) + manifest.jsonl
      generations/gen_XXXX/  meta.json + parent_SS/{prompt.txt, child_CC.txt}
      buffer/          PUCT snapshots (written by the sampler)
    <run_dir>/../index.csv   one row per run (cross-experiment view)
      stale_<timestamp>/     a rewound resume's discarded tail (results.resume, never written here)

Pass ``resume_step=N`` to continue a run rather than start one: the books of generations before N are
reloaded from the per-generation ``meta.json`` files and ``solutions/manifest.jsonl`` (see
``results.resume.prior_state``), so summary.json stays ONE cumulative account of the run and solution
ids keep counting. ``status`` in summary.json is therefore a claim about the last process to write it —
whether a run is finished is a question about the artifacts, which ``results.resume`` answers.

Standalone (no dependency on puct/envs internals beyond the public State/RolloutResult shapes) so
the RL/SFT variants can reuse it. All methods are synchronous and contain no ``await``, so they are
safe to call from concurrent coroutines on the single-threaded asyncio loop.
"""
from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
from datetime import datetime
from typing import Any

# Cap on the failure message stored per candidate. Big enough to keep the *terminal* exception of a
# Ray traceback (the part that says "No CPU group available" / "Process timed out" / etc.), which the
# old 200/500-char caps clipped — that clipping is exactly why infra-vs-genuine failures were
# indistinguishable. See sandbox.classify_failure.
MAX_MSG_CHARS = 4000

# Token-accounting fields summed per generation and per run (produced by icl.loop._sum_usage).
# All are exact server-reported counts; any the server omits stay 0.
#
# cache_queries/cache_hits come from the server's /metrics counters. Two things about them:
#   * they are counted PER SEQUENCE, while prompt_tokens counts each prompt once per request, so with
#     n>1 queries ~= n * prompt_tokens. Only their RATIO (cache_hit_rate) is meaningful — never divide
#     cache_hits by prompt_tokens.
#   * they are server-GLOBAL, so they mix traffic when several runs share one vLLM server.
USAGE_KEYS = ("requests", "completions", "prompt_tokens", "cached_prompt_tokens",
              "completion_tokens", "reasoning_tokens", "reasoning_chars", "truncated",
              "cache_queries", "cache_hits")


def _usage_derived(u: dict) -> dict:
    """Add the two ratios worth reading at a glance: prefix-cache hit rate and decode per completion.

    The hit rate prefers the /metrics counters; some vLLM builds report ``cached_tokens = 0`` in the
    per-request usage even while the cache is serving most of the prefill.
    """
    pt, ct, comps = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("completions", 0)
    queries, hits = u.get("cache_queries", 0), u.get("cache_hits", 0)
    hit_rate = (hits / queries) if queries else (u.get("cached_prompt_tokens", 0) / pt if pt else 0.0)
    rt = u.get("reasoning_tokens", 0)
    return {
        **{k: u.get(k, 0) for k in USAGE_KEYS},
        "cache_hit_rate": round(hit_rate, 4),
        "tokens_per_completion": round(ct / comps, 1) if comps else 0.0,
        # The reasoning/answer split of decode -- the number that says how much of the wall clock went
        # to hidden thinking. answer_tokens is the remainder, so it absorbs any template/special tokens.
        "answer_tokens": max(0, ct - rt),
        "reasoning_share": round(rt / ct, 4) if ct else 0.0,
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _strip_fences(code: str) -> str:
    """Remove a leading ```lang line and trailing ``` from a fenced code block."""
    c = (code or "").strip()
    if c.startswith("```"):
        lines = c.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        c = "\n".join(lines)
    return c


def _pctl(values: list[float]) -> dict[str, float]:
    """Nearest-rank percentiles of a per-candidate cost, in seconds. Same convention as
    ``icl.loop._percentiles``: the tail is the actionable statistic, because one slow candidate
    holds its whole parent group (and therefore the generation barrier)."""
    if not values:
        return {}
    ordered = sorted(values)

    def at(p: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 2)

    return {"p50": at(0.50), "p90": at(0.90), "p99": at(0.99), "max": round(ordered[-1], 2)}


def _native(value: float | None, maximize: bool) -> float | None:
    """Native (human) metric from the stored higher-is-better value."""
    if value is None:
        return None
    return value if maximize else -value


class ExperimentTracker:
    def __init__(self, run_dir: str, cfg_dict: dict[str, Any], spec, save_completions: bool = True,
                 save_reasoning: bool = True, resume_step: int | None = None):
        self.run_dir = run_dir
        self.resume_step = resume_step or 0
        self.spec = spec
        self.save_completions = save_completions
        self.save_reasoning = save_reasoning
        self.problem = cfg_dict.get("problem")
        self.metric = spec.metric_name
        self.maximize = spec.maximize

        # config knobs echoed into summary/index
        self._strategy = cfg_dict.get("context_strategy")
        self._n_context = cfg_dict.get("n_context")
        self._group_size = cfg_dict.get("group_size")
        self._groups_per_batch = cfg_dict.get("groups_per_batch")
        self._num_generations = cfg_dict.get("num_generations")

        self.sol_dir = os.path.join(run_dir, "solutions")
        self.gen_dir = os.path.join(run_dir, "generations")
        for d in (run_dir, self.sol_dir, self.gen_dir, os.path.join(run_dir, "buffer")):
            os.makedirs(d, exist_ok=True)

        # A resumed run continues the SAME run: keep its original start time (and record every resume)
        # so summary.json/config.json stay one account of one run rather than of the last relaunch.
        prior_config = {}
        if self.resume_step:
            try:
                with open(os.path.join(run_dir, "config.json")) as f:
                    prior_config = json.load(f)
            except Exception:
                prior_config = {}
        config = dict(cfg_dict)
        config["_meta"] = {
            "git_sha": _git_sha(),
            "created_at": ((prior_config.get("_meta") or {}).get("created_at")
                           or datetime.now().isoformat(timespec="seconds")),
            "host": socket.gethostname(),
            "entrypoint": spec.entrypoint,
            "metric_name": spec.metric_name,
            "maximize": spec.maximize,
        }
        self.resumes = list((prior_config.get("_meta") or {}).get("resumes") or [])
        if self.resume_step:
            self.resumes.append({"at": datetime.now().isoformat(timespec="seconds"),
                                 "from_generation": self.resume_step})
            config["_meta"]["resumes"] = self.resumes
        self.started_at = config["_meta"]["created_at"]
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self._events = open(os.path.join(run_dir, "events.jsonl"), "a")
        self._manifest = open(os.path.join(self.sol_dir, "manifest.jsonl"), "a")

        self._progress_path = os.path.join(run_dir, "progress.csv")
        self._progress_cols = [
            "generation", "valid_candidates", "failed_candidates", "success_rate",
            "gen_best_score", "best_so_far_score", "buffer_size", "puct_expansions",
            "wall_seconds",
            # token accounting (0 if the server reports no usage) — see USAGE_KEYS
            "prompt_tokens", "cached_prompt_tokens", "cache_hit_rate", "completion_tokens",
            "reasoning_tokens", "tokens_per_completion", "truncated",
            # per-candidate decode distribution — the tail is what sizes --max-tokens
            "decode_p50", "decode_p90", "decode_p99", "decode_max",
            # per-candidate GRADING cost. eval_* is the sandbox program's own runtime and is the only
            # thing --eval-timeout bounds, so size that flag off eval_max/eval_p99 (a timed-out
            # candidate contributes ~the limit itself). grade_* is end-to-end and exceeds eval_* by
            # CPU-group queueing PLUS a fixed ~2 s/candidate of Ray dispatch + pickle + NFS writes
            # (measured on an idle box), so read queue_* to tell contention from that floor.
            "eval_p50", "eval_p90", "eval_p99", "eval_max",
            "queue_p50", "queue_max", "grade_p50", "grade_max",
        ]
        if not os.path.exists(self._progress_path):
            with open(self._progress_path, "w", newline="") as f:
                csv.writer(f).writerow(self._progress_cols)

        # running state
        self._sol_seq = 0
        self._state_to_sol: dict[str, str] = {}
        self.total_candidates = 0
        self.total_success = 0
        self.total_failed = 0
        self._failure_types: dict[str, int] = {}   # run-total counts by failure_type (failed only)
        self._usage: dict[str, int] = dict.fromkeys(USAGE_KEYS, 0)   # run-total token accounting
        self.best: dict | None = None          # ranked by value; native shown for display
        self.worst_valid: dict | None = None
        self._per_gen: list[dict] = []
        self._cur: dict | None = None
        if self.resume_step:
            self._load_prior_state()

    def _load_prior_state(self) -> None:
        """Reload the books of generations before ``resume_step``.

        Without this a resumed run started its totals, its best-so-far, its solution numbering and its
        per_generation list from zero, then overwrote summary.json with only the generations done after
        the resume — so a 15-generation run that was resumed at 12 ended up claiming 3 generations and
        ``status: complete``, and the next --resume rewound to generation 3. The prior numbers are
        rebuilt from the per-generation meta.json files and solutions/manifest.jsonl, which are written
        once each (see results.resume). Import is local: results.resume imports this module.
        """
        from results.resume import prior_state

        prior = prior_state(self.run_dir, self.resume_step)
        self._per_gen = list(prior.per_generation)
        self._usage = dict(prior.usage)
        self.total_candidates = prior.candidates
        self.total_success = prior.succeeded
        self.total_failed = prior.failed
        self._failure_types = dict(prior.failure_types)
        self._sol_seq = prior.sol_seq
        self._state_to_sol = dict(prior.state_to_sol)
        self.best, self.worst_valid = prior.best, prior.worst_valid
        if prior.started_at:
            self.started_at = prior.started_at

    # ---- paths -------------------------------------------------------------
    def _gen_path(self, gen: int) -> str:
        return os.path.join(self.gen_dir, f"gen_{gen:04d}")

    def _parent_path(self, gen: int, slot: int) -> str:
        return os.path.join(self._gen_path(gen), f"parent_{slot:02d}")

    def _rel(self, path: str) -> str:
        return os.path.relpath(path, self.run_dir)

    def _parent_ref(self, state) -> str:
        """A solution id for a parent that was itself a proposed solution, else 'seed'."""
        if state.id in self._state_to_sol:
            return self._state_to_sol[state.id]
        return "seed" if getattr(state, "timestep", -1) == -1 else state.id

    # ---- lifecycle ---------------------------------------------------------
    def start_generation(self, gen: int, parents: list) -> None:
        os.makedirs(self._gen_path(gen), exist_ok=True)
        self._cur = {
            "generation": gen,
            "valid_candidates": 0, "failed_candidates": 0,
            "gen_best_value": None, "gen_best_score": None, "gen_best_sol": None,
            "failure_types": {},        # per-generation counts by failure_type (failed only)
            "usage": dict.fromkeys(USAGE_KEYS, 0),
            "parents": {},
            # grading cost of every candidate this generation, valid or not (see _progress_cols)
            "eval_times": [], "queue_times": [], "grade_times": [],
        }
        for slot, p in enumerate(parents):
            self._cur["parents"][slot] = {
                "slot": slot,
                "parent_sol": self._parent_ref(p),
                "parent_state_id": p.id,
                "parent_score": _native(p.value, self.maximize),
                "prompt_file": None,
                "children": [],
            }

    def record_group(self, gen: int, slot: int, parent, prompt: str, completions: list, results: list,
                     reasonings: list | None = None, finish_reasons: list | None = None,
                     reasoning_tokens: list | None = None, answer_tokens: list | None = None,
                     usage: dict | None = None) -> None:
        """Record one parent's group. ``reasonings`` / ``finish_reasons`` / ``reasoning_tokens`` /
        ``answer_tokens`` are per candidate (same order as ``completions``); ``usage`` is the group's
        request-level token accounting (see _sum_usage)."""
        reasonings = reasonings or []
        finish_reasons = finish_reasons or []
        reasoning_tokens = reasoning_tokens or []
        answer_tokens = answer_tokens or []
        if usage:
            for k, v in usage.items():
                self._cur["usage"][k] = self._cur["usage"].get(k, 0) + v

        pinfo = self._cur["parents"].setdefault(slot, {
            "slot": slot, "parent_sol": self._parent_ref(parent), "parent_state_id": parent.id,
            "parent_score": _native(parent.value, self.maximize), "prompt_file": None, "children": [],
        })
        pdir = self._parent_path(gen, slot)
        os.makedirs(pdir, exist_ok=True)
        prompt_file = os.path.join(pdir, "prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)
        pinfo["prompt_file"] = self._rel(prompt_file)

        for child_idx, (comp, res) in enumerate(zip(completions, results)):
            self.total_candidates += 1
            reasoning = reasonings[child_idx] if child_idx < len(reasonings) else ""
            finish_reason = finish_reasons[child_idx] if child_idx < len(finish_reasons) else ""
            r_tokens = reasoning_tokens[child_idx] if child_idx < len(reasoning_tokens) else None
            a_tokens = answer_tokens[child_idx] if child_idx < len(answer_tokens) else None
            # Total decode this candidate cost. This is the per-candidate number to build a
            # distribution from when sizing --max-tokens; it may sum slightly below a request's
            # usage.completion_tokens, which also covers per-sequence template/special tokens.
            d_tokens = None if r_tokens is None or a_tokens is None else r_tokens + a_tokens
            completion_file = None
            if self.save_completions:
                completion_file = os.path.join(pdir, f"child_{child_idx:02d}.txt")
                with open(completion_file, "w") as f:
                    f.write(comp)
            # Reasoning goes to its OWN file, never appended to child_NN.txt: recheck_failures.py
            # re-extracts code from the completion file, so that file must stay the raw answer text.
            if self.save_reasoning and reasoning:
                with open(os.path.join(pdir, f"child_{child_idx:02d}.reasoning.txt"), "w") as f:
                    f.write(reasoning)

            sol = None
            if res.correctness > 0 and res.next_state is not None:
                sol = self._write_solution(gen, parent, res)
                self.total_success += 1
                self._cur["valid_candidates"] += 1
                v = res.next_state.value            # rank value (higher = better)
                score = res.raw_score               # human/native metric
                if self._cur["gen_best_value"] is None or v > self._cur["gen_best_value"]:
                    self._cur["gen_best_value"] = v
                    self._cur["gen_best_score"] = score
                    self._cur["gen_best_sol"] = sol
                if self.best is None or v > self.best["rank_value"]:
                    self.best = {"score": score, "rank_value": v, "sol": sol, "generation": gen}
                if self.worst_valid is None or v < self.worst_valid["rank_value"]:
                    self.worst_valid = {"score": score, "rank_value": v, "sol": sol, "generation": gen}
            else:
                self.total_failed += 1
                self._cur["failed_candidates"] += 1
                ft = res.failure_type or "unknown"
                self._cur["failure_types"][ft] = self._cur["failure_types"].get(ft, 0) + 1
                self._failure_types[ft] = self._failure_types.get(ft, 0) + 1

            failure_type = "" if res.correctness > 0 else (res.failure_type or "unknown")
            for key, attr in (("eval_times", "eval_seconds"), ("queue_times", "queue_seconds"),
                              ("grade_times", "grade_seconds")):
                val = getattr(res, attr, None)
                if val is not None:
                    self._cur[key].append(val)
            child_rec = {
                "child": child_idx,
                "correctness": res.correctness,
                "correct_format": res.correct_format,
                "raw_score": res.raw_score if res.correctness > 0 else None,
                "sol": sol,
                "failure_type": failure_type,
                "finish_reason": finish_reason,   # "length" = truncated by max_tokens
                "reasoning_tokens": r_tokens,
                "answer_tokens": a_tokens,
                "msg": res.msg[:MAX_MSG_CHARS],
                "completion_file": self._rel(completion_file) if completion_file else None,
            }
            pinfo["children"].append(child_rec)

            self._events.write(json.dumps({
                "generation": gen,
                "parent_slot": slot,
                "parent_sol": pinfo["parent_sol"],
                "parent_state_id": parent.id,
                "child": child_idx,
                "correctness": res.correctness,
                "correct_format": res.correct_format,
                "raw_score": res.raw_score if res.correctness > 0 else None,
                "reward": res.reward,
                "sol": sol,
                "failure_type": failure_type,
                "finish_reason": finish_reason,
                "msg": res.msg[:MAX_MSG_CHARS],
                "completion_chars": len(comp),
                "reasoning_chars": len(reasoning),
                # exact, from the server's /tokenize; None if that endpoint was unavailable
                "reasoning_tokens": r_tokens,
                "answer_tokens": a_tokens,
                "decode_tokens": d_tokens,
                # grading cost of THIS candidate; eval_seconds is what --eval-timeout bounds
                "eval_seconds": getattr(res, "eval_seconds", None),
                "queue_seconds": getattr(res, "queue_seconds", None),
                "grade_seconds": getattr(res, "grade_seconds", None),
                "completion_file": self._rel(completion_file) if completion_file else None,
                "prompt_file": pinfo["prompt_file"],
            }) + "\n")
        self._events.flush()

    def _write_solution(self, gen: int, parent, res) -> str:
        self._sol_seq += 1
        sol = f"sol_{self._sol_seq:06d}"
        st = res.next_state
        self._state_to_sol[st.id] = sol
        parent_sol = self._parent_ref(parent)
        code = _strip_fences(res.parsed_code or st.code or "")
        stdout = (st.observation or "").strip().replace("\n", " ")[:200]

        header = (
            f"# {sol} | problem={self.problem} entrypoint={self.spec.entrypoint}\n"
            f"# generation={gen} parent={parent_sol} (state {parent.id[:8]}) state={st.id[:8]} "
            f"{self.metric}={res.raw_score:.6f} correctness={res.correctness}\n"
            f"# stdout(first 200): {stdout}\n"
            f"# NOTE: model code as-parsed; at eval time the harness also injects a preamble\n"
            f"#       (validator source + construction globals) via envs/<problem>.py.\n"
        )
        with open(os.path.join(self.sol_dir, sol + ".py"), "w") as f:
            f.write(header + "\n" + code + "\n")

        self._manifest.write(json.dumps({
            "sol": sol,
            "state_id": st.id,
            "gen": gen,
            "parent_sol": parent_sol,
            "parent_state_id": parent.id,
            "raw_score": res.raw_score,
            "value": st.value,
            "correctness": res.correctness,
            "entrypoint": self.spec.entrypoint,
        }) + "\n")
        self._manifest.flush()
        return sol

    def end_generation(self, gen: int, sampler, usage: dict | None = None,
                       wall_seconds: float | None = None,
                       decode_percentiles: dict | None = None) -> None:
        try:
            stats = sampler.get_sample_stats()
        except Exception:
            stats = {}
        buffer_size = stats.get("puct/buffer_size")
        puct_expansions = stats.get("puct/T")   # PUCT global visit counter (total node expansions)

        cur = self._cur
        n_valid, n_failed = cur["valid_candidates"], cur["failed_candidates"]
        total = n_valid + n_failed
        success_rate = (n_valid / total) if total else 0.0
        best_score = self.best["score"] if self.best else None

        # Prefer the generation-level usage passed in by the loop (it covers every request of the
        # generation); fall back to the sum accumulated from record_group calls.
        gen_usage = _usage_derived(usage if usage is not None else cur.get("usage", {}))
        for k in USAGE_KEYS:      # run totals stay exactly the sum of the per-generation totals
            self._usage[k] += gen_usage[k]
        gen_stats = {
            "generation": gen,
            "valid_candidates": n_valid,
            "failed_candidates": n_failed,
            "success_rate": round(success_rate, 4),
            "gen_best_score": cur["gen_best_score"],
            "gen_best_sol": cur["gen_best_sol"],
            "best_so_far_score": best_score,
            "buffer_size": buffer_size,
            "puct_expansions": puct_expansions,
            "wall_seconds": round(wall_seconds, 1) if wall_seconds is not None else None,
            "failure_types": dict(cur.get("failure_types", {})),
            "usage": gen_usage,
            # Kept OUT of `usage` on purpose: usage fields are summed into run totals, and summing
            # percentiles is meaningless. Per-candidate decode tokens; see icl.loop._percentiles.
            "decode_percentiles": dict(decode_percentiles or {}),
            # Same reasoning, for grading cost. Read eval_percentiles to decide --eval-timeout.
            "eval_percentiles": _pctl(cur.get("eval_times", [])),
            "queue_percentiles": _pctl(cur.get("queue_times", [])),
            "grade_percentiles": _pctl(cur.get("grade_times", [])),
        }
        meta = {
            "generation": gen,
            "stats": gen_stats,
            "parents": [cur["parents"][s] for s in sorted(cur["parents"])],
        }
        with open(os.path.join(self._gen_path(gen), "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        with open(self._progress_path, "a", newline="") as f:
            csv.writer(f).writerow([
                gen, n_valid, n_failed, round(success_rate, 4),
                cur["gen_best_score"], best_score, buffer_size, puct_expansions,
                gen_stats["wall_seconds"],
                gen_usage["prompt_tokens"], gen_usage["cached_prompt_tokens"],
                gen_usage["cache_hit_rate"], gen_usage["completion_tokens"],
                gen_usage["reasoning_tokens"], gen_usage["tokens_per_completion"],
                gen_usage["truncated"],
                *(gen_stats["decode_percentiles"].get(k) for k in ("p50", "p90", "p99", "max")),
                *(gen_stats["eval_percentiles"].get(k) for k in ("p50", "p90", "p99", "max")),
                *(gen_stats["queue_percentiles"].get(k) for k in ("p50", "max")),
                *(gen_stats["grade_percentiles"].get(k) for k in ("p50", "max")),
            ])

        self._per_gen.append(gen_stats)
        self._write_summary(status="running")
        self._update_index(status="running")

    def _write_summary(self, status: str) -> None:
        total = self.total_candidates
        summary = {
            "problem": self.problem,
            "strategy": self._strategy,
            "n_context": self._n_context,
            "group_size": self._group_size,
            "groups_per_batch": self._groups_per_batch,
            "num_generations": self._num_generations,
            "status": status,
            "metric_name": self.metric,
            "maximize": self.maximize,
            "totals": {
                "candidates": total,
                "succeeded": self.total_success,
                "failed": self.total_failed,
                "success_rate": round(self.total_success / total, 4) if total else 0.0,
                "unique_solutions": self._sol_seq,
                "failure_types": dict(self._failure_types),
            },
            "usage": _usage_derived(self._usage),
            "best": self.best,
            "worst_valid": self.worst_valid,
            "per_generation": self._per_gen,
            "started_at": self.started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            # Empty for a run that never stopped; one entry per relaunch otherwise. A run with entries
            # here has generations produced by different processes (same config, different wall clock).
            "resumes": self.resumes,
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def _update_index(self, status: str) -> None:
        try:
            parent = os.path.dirname(os.path.abspath(self.run_dir))
            index = os.path.join(parent, "index.csv")
            name = os.path.basename(os.path.abspath(self.run_dir))
            cols = [
                "run", "problem", "strategy", "n_context", "group_size", "groups_per_batch",
                "num_generations", "generations_done", "best_score", "status", "updated_at",
            ]
            row = {
                "run": name, "problem": self.problem, "strategy": self._strategy,
                "n_context": self._n_context, "group_size": self._group_size,
                "groups_per_batch": self._groups_per_batch, "num_generations": self._num_generations,
                "generations_done": len(self._per_gen),
                "best_score": self.best["score"] if self.best else None,
                "status": status,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            rows = []
            if os.path.exists(index):
                with open(index) as f:
                    rows = [r for r in csv.DictReader(f) if r.get("run") != name]
            rows.append(row)
            with open(index, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k) for k in cols})
        except Exception:
            pass

    def close(self, status: str = "complete") -> None:
        self._write_summary(status)
        self._update_index(status)
        for fh in (self._events, self._manifest):
            try:
                fh.close()
            except Exception:
                pass
