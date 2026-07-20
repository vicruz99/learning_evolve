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


def _native(value: float | None, maximize: bool) -> float | None:
    """Native (human) metric from the stored higher-is-better value."""
    if value is None:
        return None
    return value if maximize else -value


class ExperimentTracker:
    def __init__(self, run_dir: str, cfg_dict: dict[str, Any], spec, save_completions: bool = True):
        self.run_dir = run_dir
        self.spec = spec
        self.save_completions = save_completions
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

        config = dict(cfg_dict)
        config["_meta"] = {
            "git_sha": _git_sha(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "entrypoint": spec.entrypoint,
            "metric_name": spec.metric_name,
            "maximize": spec.maximize,
        }
        self.started_at = config["_meta"]["created_at"]
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self._events = open(os.path.join(run_dir, "events.jsonl"), "a")
        self._manifest = open(os.path.join(self.sol_dir, "manifest.jsonl"), "a")

        self._progress_path = os.path.join(run_dir, "progress.csv")
        self._progress_cols = [
            "generation", "valid_candidates", "failed_candidates", "success_rate",
            "gen_best_score", "best_so_far_score", "buffer_size", "puct_expansions",
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
        self.best: dict | None = None          # ranked by value; native shown for display
        self.worst_valid: dict | None = None
        self._per_gen: list[dict] = []
        self._cur: dict | None = None

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
            "parents": {},
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

    def record_group(self, gen: int, slot: int, parent, prompt: str, completions: list, results: list) -> None:
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
            completion_file = None
            if self.save_completions:
                completion_file = os.path.join(pdir, f"child_{child_idx:02d}.txt")
                with open(completion_file, "w") as f:
                    f.write(comp)

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

            child_rec = {
                "child": child_idx,
                "correctness": res.correctness,
                "correct_format": res.correct_format,
                "raw_score": res.raw_score if res.correctness > 0 else None,
                "sol": sol,
                "msg": res.msg[:200],
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
                "msg": res.msg[:500],
                "completion_chars": len(comp),
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

    def end_generation(self, gen: int, sampler) -> None:
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
            },
            "best": self.best,
            "worst_valid": self.worst_valid,
            "per_generation": self._per_gen,
            "started_at": self.started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
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
