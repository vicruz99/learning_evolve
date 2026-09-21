"""Render cells from cells.yaml into run folders.

    mkcell.py --campaign smoke [--cells cells.yaml] [--only 'ac2_.*bnb'] [--set hours=0.25 --set min_evals=3]
              [--seeds 1] [--dry] [--force]

Every cell is built the same way, so the only differences between cells are the ones under test.
The prompt is a Jinja template: the sentences that describe the site (cores, hours, interpreter)
render from the cell config so the prompt can never claim a budget the run does not have
(the campaign's fixcores.py lesson). The method block is the ONLY thing that differs between the
plain / evo / many variants of one problem; PROMPT_DIFF.md in each cell proves it.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import jinja2
import yaml

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "tracker"))
from tracker import install_tracker, load_meta  # noqa: E402

HARNESS_SHORT = {"bnbcode": "bnb", "claude": "cc"}
# `site` (campaign key): where the cells run. Names the cluster in the prompt, decides whether the
# "no cells on /home" guard applies (rng-dl01's ~100 GB quota; Marvin's home is the only fs there)
# and which submitter bin/submit uses (bsub on rng-dl01, subbin on Marvin). 2026-09-21.
SITES = {"rngdl01": "rng-dl01", "marvin": "rb-hpc (Marvin)"}
REASONING = ("low", "medium", "xhigh", "off")
PROMPTS = ("plain", "evo", "many")


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def coerce(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def apply_sets(cfg: dict, sets: list[str]) -> dict:
    for s in sets:
        k, v = s.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = coerce(v)
    return cfg


def cell_name(c: dict) -> str:
    return f"{c['problem'].lower()}_{c['prompt']}_{HARNESS_SHORT[c['harness']]}_r{c['reasoning']}_s{c['seed']}"


def expand(spec: dict, sets: list[str], seeds_override: list[int] | None) -> list[dict]:
    defaults = apply_sets(copy.deepcopy(spec.get("defaults", {})), sets)
    cells = []
    for raw in spec["cells"]:
        c = deep_merge(defaults, raw)
        seeds = seeds_override or c.get("seeds", [1])
        for s in seeds:
            cc = copy.deepcopy(c)
            cc["seed"] = int(s)
            cc.pop("seeds", None)
            if cc["harness"] not in HARNESS_SHORT:
                raise SystemExit(f"unknown harness {cc['harness']}")
            if cc["prompt"] not in PROMPTS:
                raise SystemExit(f"unknown prompt variant {cc['prompt']}")
            if cc["reasoning"] not in REASONING:
                raise SystemExit(f"reasoning must be one of {REASONING}, got {cc['reasoning']}")
            cc["name"] = cell_name(cc)
            cells.append(cc)
    return cells


def render_prompt(c: dict, meta: dict, variant: str) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader([str(V2 / "prompts"), str(V2 / "problems" / c["problem"])]),
        undefined=jinja2.StrictUndefined, keep_trailing_newline=True, trim_blocks=False, lstrip_blocks=False,
    )
    parallel = max(1, int(c["cores"]) // int(meta["cpus_per_candidate"]))
    hours = float(c["hours"])
    hours_txt = str(int(hours)) if hours == int(hours) else f"{hours:g}"
    ctx = {
        "P": meta, "CORES": int(c["cores"]), "PARALLEL": parallel, "HOURS": hours_txt,
        "PYTHON": c["python"], "MIN_EVALS": int(c["min_evals"]), "MAX_EVALS": int(c.get("max_evals") or 0),
        "EVALS_PER_HOUR": max(1, math.ceil(int(c["min_evals"]) / max(hours, 1e-9))),
        "BUDGET_S": int(c.get("budget_s") or meta["budget_s"]), "KILL_S": int(c.get("kill_s") or meta["kill_s"]),
        "SITE": SITES[c.get("site", "rngdl01")],
        "method_file": f"method_{variant}.md",
    }
    text = env.get_template("prompt.md.j2").render(**ctx)
    # The scoring rule and method files are themselves templates (they mention PYTHON etc.).
    text = env.from_string(text).render(**ctx) if "{{" in text else text
    if "{{" in text or "{%" in text:
        raise SystemExit(f"unrendered template syntax left in prompt for {c['name']}")
    # The budget the prompt states must be the budget the run gets.
    if f"at most **{ctx['CORES']} CPU cores**" not in text or f"**{parallel} candidates in flight" not in text:
        raise SystemExit(f"{c['name']}: rendered prompt does not state cores={ctx['CORES']} parallel={parallel}")
    return text


def bnbcode_config(c: dict, cell_dir: Path) -> dict:
    base = json.loads((V2 / "config" / "opencode.base.json").read_text())
    b = c["bnbcode"]
    model = c["model"]
    fam = c.get("model_family", "qwen3.8")
    provider = "vllm" + fam.replace("qwen", "").replace(".", "")      # vllm38 / vllm36
    if c["reasoning"] == "off":
        ctk = {"enable_thinking": False}
    elif fam == "qwen3.8":
        ctk = {"reasoning_effort": c["reasoning"]}                 # Qwen3.8 template: low|medium|xhigh
    else:
        ctk = {}                                                    # Qwen3.6 has no effort levels: thinking on
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{provider}/{model}",
        "provider": {provider: {
            "name": f"vLLM {model} via llm_relay :{c['relay_port']}",
            "options": {"baseURL": f"http://127.0.0.1:{c['relay_port']}/v1", "apiKey": "dummy"},
            "models": {model: {
                "name": model, "temperature": True, "reasoning": True, "tool_call": True,
                "limit": {"context": int(b["context_tokens"]), "output": int(b["output_tokens"])},
                **({"options": {"chat_template_kwargs": ctk}} if ctk else {}),
            }},
        }},
        "continual_work": {"enabled": bool(b["continual_work"])},
        "tool_output": {"max_lines": int(b["tool_output"]["max_lines"]), "max_bytes": int(b["tool_output"]["max_bytes"])},
        "agent": {"build": {"temperature": b["temperature"], "top_p": b["top_p"]},
                  "plan": {"temperature": b["temperature"], "top_p": b["top_p"]}},
        "permission": base["permission"],
    }
    comp = {}
    if b.get("compaction_context_limit"):
        comp["context_limit"] = int(b["compaction_context_limit"])
    if "compaction_notes" in b:
        comp["notes"] = bool(b["compaction_notes"])
    if comp:
        cfg["compaction"] = comp
    return cfg


def claude_settings(c: dict, cell_dir: Path) -> dict:
    src = "claude_guard_stop.json" if c["claude"]["stop_hook"] else "claude_guard.json"
    s = json.loads((V2 / "config" / src).read_text())
    ws = (cell_dir / "workspace").resolve()
    extra = []
    for name in ("eval.py", "_official_evaluator.py", "_adrs_track.json", "INITIAL_PROMPT.md"):
        for tool in ("Edit", "Write"):
            extra.append(f"{tool}(/{ws}/{name})")
    extra.append("Bash(chmod:*)")
    s["permissions"]["deny"] = s["permissions"]["deny"] + extra
    if c["claude"]["stop_hook"]:
        # the driver's interpreter, not `python3`: on Marvin `python3` is 3.6 (2026-09-21)
        s["hooks"] = {"Stop": [{"matcher": "", "hooks": [{"type": "command",
                      "command": f"{sys.executable} {V2 / 'hooks' / 'cc_stop_hook.py'}"}]}]}
    return s


def assign_ports(c: dict, campaign: str, index: int) -> None:
    """One relay (+ one LiteLLM) per cell, on ports unique to (campaign, cell index), and the
    server preference list rotated by index so concurrent cells spread over the GPUs while each
    cell stays sticky to one server (vLLM prefix cache). Two campaigns running at once on one node
    get different bases from the campaign-name hash; override with relay_port_base/litellm_port_base."""
    import zlib
    if index >= 100:
        raise SystemExit("more than 100 cells in one campaign would collide on ports; split the campaign")
    c.pop("_index", None)
    h = zlib.crc32(campaign.encode()) % 20
    c["relay_port"] = int(c.get("relay_port_base") or 9300 + 100 * h) + index
    c["litellm_port"] = int(c.get("litellm_port_base") or 4300 + 100 * h) + index
    c["cliff_port"] = c["litellm_port"] + 2000        # CliffCompaction proxy in front of LiteLLM (claude.cliff)
    toks = [t.strip() for t in str(c["relay_jobs"]).split(",") if t.strip()]
    if c.get("balance", True) and len(toks) > 1:
        k = index % len(toks)
        toks = toks[k:] + toks[:k]
    c["relay_jobs"] = ",".join(toks)
    c["cell_index"] = index


def litellm_yaml(c: dict) -> str:
    """Per-cell copy of config/litellm_qwen38.yaml with this cell's relay port."""
    fam = c.get("model_family", "qwen3.8")
    src = (V2 / "config" / f"litellm_{fam.replace('.', '')}.yaml").read_text()
    return re.sub(r"api_base: http://127\.0\.0\.1:\d+/v1", f"api_base: http://127.0.0.1:{c['relay_port']}/v1", src)


SCRATCH_HINT = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/vicruz/agent_runs_v2"


def check_not_home(cell_dir: Path) -> None:
    """Refuse to put a cell's data on /home.

    /home/<user> on rng-dl01 carries a ~100 GB per-user quota that weka reports as
    `[Errno 28] No space left on device`, not EDQUOT, while `df` shows ~100 TB free. On
    2026-09-15 `q38ac_r3` was the only campaign whose cells were real directories in home
    (every other campaign resolves into scratch); its pg dumps and .npy snapshots pushed home
    over the cap and killed all 15 cells, several of them repeatedly.

    The check resolves symlinks on purpose: a cell reached through
    `~/agent_runs/v2/<camp>/<cell>` is fine as long as it *lands* on scratch, which is the
    layout every healthy campaign uses. A cell that already exists as a symlink into scratch
    therefore still re-renders; only a cell that would be created in home is refused.
    """
    if cell_dir.exists():
        real = cell_dir.resolve()
    else:
        parent = cell_dir.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        real = parent.resolve() / cell_dir.name
    home = Path(os.path.expanduser("~")).resolve()
    if real == home or home in real.parents:
        raise SystemExit(
            f"refusing to create {cell_dir.name} at {real}: that is the home filesystem, which\n"
            f"has a ~100 GB per-user quota reported as ENOSPC and killed the q38ac_r3 wave on\n"
            f"2026-09-15. Put the campaign on scratch instead, e.g.\n"
            f"  mkdir -p {SCRATCH_HINT}/{cell_dir.parent.name}\n"
            f"  ln -s {SCRATCH_HINT}/{cell_dir.parent.name} {cell_dir.parent}"
        )

def build_cell(c: dict, runs_root: Path, campaign: str, dry: bool, force: bool, index: int = 0) -> Path:
    assign_ports(c, campaign, index)
    cell_dir = runs_root / campaign / c["name"]
    if c.get("site", "rngdl01") not in SITES:
        raise SystemExit(f"unknown site {c.get('site')!r}; one of {sorted(SITES)}")
    if c.get("site", "rngdl01") == "rngdl01":
        check_not_home(cell_dir)
    meta = load_meta(V2 / "problems" / c["problem"] / "meta.yaml")
    prompt = render_prompt(c, meta, c["prompt"])
    plain = render_prompt(c, meta, "plain")
    if dry:
        print(f"[dry] {cell_dir}  ({len(prompt)} chars) relay:{c['relay_port']} litellm:{c['litellm_port']} servers:{c['relay_jobs']}")
        return cell_dir
    if cell_dir.exists():
        if not force:
            raise SystemExit(f"exists already, refusing to clobber: {cell_dir} (use --force)")
        shutil.rmtree(cell_dir)
    ws = cell_dir / "workspace"
    ws.mkdir(parents=True)
    (ws / "INITIAL_PROMPT.md").write_text(prompt)
    shutil.copy2(V2 / "problems" / c["problem"] / meta["npy"], ws / meta["npy"])
    official = V2 / "problems" / c["problem"] / "eval.py"
    meta_for_tracker = dict(meta, problem=c["problem"], official_md5=hashlib.md5(official.read_bytes()).hexdigest())
    install_tracker(cell_dir, official, meta_for_tracker, max_evals=int(c.get("max_evals") or 0))
    os.chmod(ws / "INITIAL_PROMPT.md", 0o444)
    diff = list(difflib.unified_diff(plain.splitlines(), prompt.splitlines(), "plain", c["prompt"], lineterm=""))
    (cell_dir / "PROMPT_DIFF.md").write_text("\n".join(diff) + "\n" if diff else "(identical to plain)\n")
    resolved = dict(c, cell_dir=str(cell_dir), workspace=str(ws), campaign=campaign, v2=str(V2),
                    problem_meta=meta, parallel=max(1, int(c["cores"]) // int(meta["cpus_per_candidate"])))
    if c["harness"] == "bnbcode":
        cfg = bnbcode_config(c, cell_dir)
        (ws / "opencode.json").write_text(json.dumps(cfg, indent=2) + "\n")
        os.chmod(ws / "opencode.json", 0o444)
        resolved["opencode_config"] = str(ws / "opencode.json")
    else:
        s = claude_settings(c, cell_dir)
        (cell_dir / "claude_settings.json").write_text(json.dumps(s, indent=2) + "\n")
        resolved["claude_settings"] = str(cell_dir / "claude_settings.json")
        resolved["cc_model"] = f"{c.get('model_family', 'qwen3.8')}-{c['reasoning']}"
        (cell_dir / "litellm.yaml").write_text(litellm_yaml(c))
        resolved["litellm_yaml"] = str(cell_dir / "litellm.yaml")
    (cell_dir / "cell.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (cell_dir / "run").mkdir(exist_ok=True)   # host-side scratch (stophook.json, procsample)
    print(f"created {cell_dir}")
    return cell_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--cells", type=Path, default=V2 / "cells.yaml")
    ap.add_argument("--only", default=None, help="regex on the cell name")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="override a default, e.g. hours=0.25 or bnbcode.continual_work=true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    spec = yaml.safe_load(a.cells.read_text())
    runs_root = Path(os.path.expanduser(spec.get("runs_root", "~/agent_runs/v2")))
    cells = expand(spec, a.sets, a.seeds)
    # ports derive from the cell's index in the FULL expansion, so re-rendering a subset with
    # --only keeps every cell's ports stable and unique (a subset re-render on 2026-09-08 gave six
    # cells the ports of six others; pairs then shared one relay and reaped each other's)
    for i, c in enumerate(cells):
        c["_index"] = i
    if a.only:
        cells = [c for c in cells if re.search(a.only, c["name"])]
    if not cells:
        raise SystemExit("no cells selected")
    for c in cells:
        build_cell(c, runs_root, a.campaign, a.dry, a.force, index=c["_index"])
    print(f"--- {len(cells)} cell(s) -> {runs_root / a.campaign}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
