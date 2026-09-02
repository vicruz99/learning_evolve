#!/usr/bin/env python3
"""ShinkaEvolve driver for the GPU-mode TriMul kernel task (H100).

Parity with the ICL arm (src/envs/kernel_trimul.py::TrimulH100Env):
  * task_sys_msg = TRIMUL_PROMPT verbatim (loaded from src/envs/kernel_prompt.py by path, so the
    Shinka venv needs no `src` install) + the same Rules block, with `_HW_RULE_H100` verbatim.
  * Two ICL-only rules are dropped because they are harness mechanics Shinka replaces with its own
    format instructions: "Define all of your code in one final ```python``` block" and the trailing
    "<strategy> ... then return the final program" request.
  * No seed kernel (initial.py is a NotImplemented stub), see initial.py.
Grading happens in evaluate.py through the frozen GPU-mode harness on the job's single card.
"""
import argparse
import importlib.util
import os
import re
from pathlib import Path

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

HERE = Path(__file__).resolve().parent          # resolve(): run through the examples/ symlink
REPO = HERE.parents[3]                          # .../learning_evolve
ENVS = REPO / "src" / "envs"

# src/envs/kernel_trimul.py: `_HW_RULE_H100 = "- You must use trition 3.3.1 and these kernels will be run on an H100."`
# (TTT-Discover's line, typo included). Kept literal here and checked against the source below so a
# change there cannot silently desynchronise the two arms.
HW_RULE_H100 = "- You must use trition 3.3.1 and these kernels will be run on an H100."


def _load_trimul_prompt() -> str:
    spec = importlib.util.spec_from_file_location("kernel_prompt", ENVS / "kernel_prompt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRIMUL_PROMPT


def _check_hw_rule() -> None:
    src = (ENVS / "kernel_trimul.py").read_text()
    m = re.search(r'_HW_RULE_H100\s*=\s*"([^"]+)"', src)
    if m and m.group(1) != HW_RULE_H100:
        raise SystemExit(f"_HW_RULE_H100 in kernel_trimul.py changed to: {m.group(1)!r} -- update run_evo.py")


RULES = f"""
Rules:
- The tensors arguments passed in will be already on your cuda device.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
{HW_RULE_H100}
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
- Your program must define `custom_kernel(data)` as described above; the evaluator imports the file and calls it directly. Lower geometric-mean runtime over the benchmark shapes is better.
"""


def main(config_path: str, results_dir: str | None = None, embedding_model: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _check_hw_rule()
    config["evo_config"]["task_sys_msg"] = _load_trimul_prompt().rstrip() + "\n" + RULES
    if results_dir is not None:
        config["evo_config"]["results_dir"] = results_dir
    if embedding_model is not None:
        config["evo_config"]["embedding_model"] = embedding_model

    # evaluate.py reads these; make the defaults explicit so a run records what graded it.
    os.environ.setdefault("KPY", os.path.expanduser("~/venvs/kernel-eval/bin/python"))
    os.environ.setdefault("TRIMUL_EVALUATE_PY", str(REPO / "coding_agent_evolve" / "gpumode" / "evaluate.py"))
    print(f"[run_evo] grader: KPY={os.environ['KPY']}  harness={os.environ['TRIMUL_EVALUATE_PY']}")

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time=config.get("job_time", "00:30:00"),
        numeric_threads_per_job=config.get("numeric_threads_per_job", 2),
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_qwen.yaml")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--embedding_model", type=str, default=None,
                        help="e.g. local/Qwen/Qwen3-Embedding-0.6B@http://127.0.0.1:8003/v1")
    args = parser.parse_args()
    main(args.config_path, args.results_dir, args.embedding_model)
