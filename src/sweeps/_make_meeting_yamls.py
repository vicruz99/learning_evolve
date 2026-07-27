"""Emit the six pre-meeting sweep files: {bon, puct, ctx} x {gptoss, qwen}."""
import os, sys
OUT = sys.argv[1]

# Shape requested: 5 parents x 16 children x 15 generations = 1200 candidates/run.
PARENTS, CHILDREN, GENS = 5, 16, 15
TOTAL = PARENTS * CHILDREN * GENS            # 1200
BON_GROUPS = TOTAL // CHILDREN               # 75 groups of 16, in ONE generation

BON_WHY = f"""\
# WHY BEST-OF-N IS ITS OWN SWEEP, AND WHY IT IS ONE "GENERATION" OF {BON_GROUPS} GROUPS
#   Best-of-N has NO dependency between generations, and this is verifiable in the code rather than a
#   guess: with `parent-source: initial` the parents come from `sample_initial_states()`, which ignores
#   the buffer entirely, and with `n-context 0` the context block is empty. The buffer and context pool
#   are still WRITTEN (for analysis) but never READ. So nothing flows from one generation to the next
#   and the generation barrier is pure overhead.
#   Measured cost of that barrier on the earlier Best-of-N runs: 19-45 % of the barrier slot-time was
#   idle, waiting for the slowest parent. On cp32 that was 4.67 h of wall for 2.55 h of work.
#
#   So this file removes the barrier instead of paying it: ONE generation of {BON_GROUPS} groups x {CHILDREN} children
#   = {TOTAL} candidates, identical to the {GENS} x {PARENTS} x {CHILDREN} = {TOTAL} of every other arm.
#   `max-gen-concurrency: {PARENTS}` is what makes this a PIPELINE rather than a stampede: it is a semaphore on
#   in-flight vLLM requests (icl/loop.py:152), released the moment `generate()` returns. So {PARENTS} requests
#   of {CHILDREN} = {PARENTS*CHILDREN} sequences stay in flight -- exactly the footprint of the other arms -- and when a
#   group returns it grades WHILE the next group is already generating. The LLM never waits for grading
#   and the CPUs never wait for the LLM.
#
#   >>> CONSEQUENCE FOR ANALYSIS -- do not skip this. <<<
#   progress.csv will have exactly ONE row, so there is no per-generation curve for this arm. That is
#   fine, because for i.i.d. sampling the right x-axis is CANDIDATE COUNT, not generation. Reconstruct
#   it from events.jsonl / solutions/manifest.jsonl, which are written in completion order:
#       best-so-far after k candidates, compared against the other arms at k = 80, 160, ... 1200
#   (the other arms' generation g corresponds to k = 80*(g+1)).
#   Two smaller consequences: `--status` will show 0/1 generations for the whole run, and the buffer is
#   flushed only at the end, so a crash loses the buffer (events.jsonl and solutions/ survive).
#   If you would rather have intermediate checkpoints, `num-generations: 3` with `groups-per-batch: 25`
#   is the same {TOTAL} candidates with 3 barriers instead of 0 -- a small price for 3 progress rows.
"""

COST_WARNING = """\
# ---------------------------------------------------------------------------------------------
# COST: READ BEFORE LAUNCHING THE ac1 RUNS. THE FULL MATRIX DOES NOT FIT IN TWO DAYS.
# ---------------------------------------------------------------------------------------------
# Across the three pre-meeting files the requested matrix is 50 runs x 1,200 candidates = 60,000
# candidates per model, split 26,400 on cp26 and 33,600 on ac1. Grading is the binding resource:
#
#   cp26   ~5.8M core-seconds  -> ~17 h on 96 cores   (BoN evals ~10 s; PUCT/ctx ~280 s each)
#   ac1   ~13.4M core-seconds  -> ~39 h on 96 cores   (assuming ~400 s per eval)
#   total                      -> ~56 h, against ~40 h of usable time before the meeting
#
# The cp26 half fits comfortably. The ac1 half does not, and worse, its cost rests on the ONE number
# nobody has measured: ac1's per-eval runtime. The ~400 s is DERIVED from old generation timings, not
# observed -- the eval instrumentation did not exist when the ac1 runs were stopped.
#
# THAT IS WHY ac1 IS A SEPARATE FILE (ac1_<model>.yaml) rather than mixed into these three: it can be
# dropped, deferred, or resized without touching the cp26 comparison. RUN ORDER:
#   1. bon_<model>.yaml -> puct_<model>.yaml -> ctx_<model>.yaml   (all cp26, ~22 h total)
#   2. THEN ac1_<model>.yaml, and only after reading the first ac1 run's `eval_p50` / `eval_p90` /
#      `eval_max` in progress.csv -- that turns the 39 h estimate into a real number within two
#      generations. If it lands near 400 s/eval: drop the n-context 5 block (9 runs, ~12 h), cut to 8
#      generations, or defer ac1 to next week.
"""

SHARED = """\
# DESIGN (see notebooks/experiment0_calibration.ipynb for the measurements behind these):
#   * 15 generations: of 8 baseline runs, one reached its final score in GENERATION 0 and never improved
#     again; among those that progressed, the median share of total gain done by generation 3 was 100 %.
#     Generations past ~10 buy almost nothing.
#   * Multiple seeds: with the SAME --seed and config, two campaigns' best-so-far diverged by up to
#     0.075 (mean 0.016), while the entire spread between three strategies in the old single-seed
#     comparison was 0.0228. One replicate per arm proves nothing; ~0.02 is the resolution floor.
#   * Runs are seed-major, so stopping early leaves a valid comparison with fewer seeds.
#   * eval-timeout is NOT set anywhere -> registry defaults (cp 530 s, ac1 1100 s). At 220 s the cap
#     killed 45 % of the PUCT arm's candidates against 1.3 % of Best-of-N's, and valid solutions ran to
#     the cap exactly. A tight cap is a selection pressure toward cheap programs, not a speed knob.
#   * num-cpus-per-task 1: eval children are single-threaded and 1-core-pinned; 2 stranded half the box.
#     MUST be identical in every file sharing a Ray head (the cpu_scheduler actor freezes it).
"""

GPTOSS = dict(
    tag="gptoss", model="openai/gpt-oss-120b", box="INESC-ID, 2x A100 80GB, 96 cores", cores=96,
    seqs=320, sweepdir="/scratch/vicstorage/runs",
    reasoning="""  reasoning-effort: medium       # ~4,000 tok/candidate measured, 0 truncations; `high` produced
                                 # ~70 % no-code completions in the one observation available""",
    server="""#   CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/scratch/vicstorage \\
#   vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --async-scheduling \\
#       --gpu-memory-utilization 0.95 --max-model-len 128000 --max-num-seqs 320 \\
#       --max-num-batched-tokens 16384 --enable-prefix-caching \\
#       --reasoning-parser openai_gptoss --port 8001""")

QWEN = dict(
    tag="qwen", model="Qwen/Qwen3.6-27B-FP8", box="Bosch, B200 (H200 unchanged), 64 cores", cores=64,
    seqs=256, sweepdir="<big-disk>/runs",
    reasoning="""  reasoning-effort: none         # gpt-oss-only knob; `none` keeps it out of the request
  thinking-token-budget: 10000   # HARD reasoning cap; vLLM force-emits </think> at the limit so the
                                 # candidate still answers. Watch `truncated` / `reasoning_tokens`: if
                                 # many candidates land exactly on 10k with no code, raise it.""",
    server="""#   HF_HOME=<scratch> \\
#   vllm serve Qwen/Qwen3.6-27B-FP8 --tensor-parallel-size 1 --async-scheduling \\
#       --gpu-memory-utilization 0.90 --max-model-len 128000 --max-num-seqs 256 \\
#       --max-num-batched-tokens 16384 --enable-prefix-caching \\
#       --reasoning-parser qwen3 --port 8001
#
#   vLLM >= 0.19.0 REQUIRED: `thinking_token_budget` landed in 0.19.0 and earlier versions ACCEPT then
#   SILENTLY IGNORE it (extra="allow"), and Qwen3_5ForConditionalGeneration is absent from the registry
#   before 0.19.1. VERIFY the cap is live before launching -- a silent no-op is the failure mode:
#     curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
#       "model":"Qwen/Qwen3.6-27B-FP8","messages":[{"role":"user","content":"Prove sqrt(2) is irrational."}],
#       "max_tokens":2000,"thinking_token_budget":64}' | python -c \\
#       'import json,sys;m=json.load(sys.stdin)["choices"][0]["message"];print(len(m.get("reasoning_content") or ""))'
#   --reasoning-parser qwen3 is REQUIRED: without it the chain of thought stays in `content` and code
#   extraction can grab a ```python fence out of the REASONING instead of the answer.""")


def header(M, title, extra, mp, nruns, cands, sess, fn, name):
    return f"""\
# =================================================================================================
# PRE-MEETING EXPERIMENT -- {title}
# {M['box']}.  Model: {M['model']}
#
# One of three files that together answer: does SELECTING which past solutions enter the prompt beat
# TTT-Discover-without-RL, and does the strategy matter beyond merely having examples?
#     sweeps/bon_{M['tag']}.yaml    Best-of-N            (no past experience at all)
#     sweeps/puct_{M['tag']}.yaml   PUCT, n-context 0    (past experience only via parent choice)
#     sweeps/ctx_{M['tag']}.yaml    context strategies   (past experience in the prompt)
# All three use {TOTAL} candidates per run so the arms are directly comparable.
#
{extra}#
{SHARED}#
# ---------------------------------------------------------------------------------------------
# SERVER (one server serves all three files)
{M['server']}
#
# LAUNCH
#   ray stop && ray start --head        # all files use num-cpus-per-task 1, so one head serves all
#   cd src && tmux new -s {sess}
#   python run_sweep.py sweeps/{fn} --print-cmds        # inspect first
#   python run_sweep.py sweeps/{fn} --sweep-dir {M['sweepdir']}/{name}
#   python run_sweep.py --status {M['sweepdir']}/{name}
#   To halt: kill the run_sweep.py pid FIRST -- `--stop` leaves the supervisor alive and it immediately
#   launches the next queued run -- then `python run_sweep.py --stop <dir>`.
#   Use a LOCAL disk, not an NFS home: the sandbox round-trips each candidate's program and result
#   through the run dir, ~80 ms/candidate on NFS versus ~0.3 ms on local disk.
#
# THIS FILE: {nruns} runs x {cands} candidates = {nruns*cands:,} candidates, max_parallel {mp}.
{COST_WARNING}# =================================================================================================
"""


def common(M, problem_free=True):
    return f"""
common:
  # --- model / server -------------------------------------------------------------------------
  model: {M['model']}
  vllm-base-url: http://localhost:8001/v1
{M['reasoning']}

  # --- token budgets (fixed across every experiment in this project) --------------------------
  max-tokens: 34000              # max decode observed 21,241 (62 %) on cp26, ~8,900 on ac1; 0 truncations
  max-context-tokens: 94000      # the budget the context block is packed against; identical everywhere
  temperature: 1.0

  # --- execution -----------------------------------------------------------------------------
  save-reasoning: false
  num-cpus-per-task: 1
"""


def emit(M):
    t = M["tag"]
    mp_bon = 3 if M["cores"] >= 96 else 2
    mp_puct = 2
    mp_ctx = 2

    # ---------------------------------------------------------------- Best-of-N
    runs = [f"  - {{name: cp26_s{s}, problem: circle_packing_26, seed: {s}}}" for s in range(1, 6)]
    body = header(M, "BEST-OF-N (no past experience), circle_packing_26, 5 runs",
                  BON_WHY, mp_bon, 5, TOTAL, f"bon_{t}", f"bon_{t}.yaml", f"bon_{t}")
    body += f"""
sweep:
  name: bon_{t}
  max_parallel: {mp_bon}                # Best-of-N grading is CHEAP on cp26 (~10 s/eval measured, ~1.6 cores
                                 # per run) and expensive on ac1, so a mixed queue balances GPU against
                                 # CPU well. Raise only if queue_p50 in progress.csv stays ~0.
  stagger: 60                    # short: there is no generation barrier to keep out of phase
  server_max_num_seqs: {M['seqs']}
{common(M)}
  # --- the Best-of-N shape: ONE generation, no barrier (see the note at the top) ---------------
  parent-source: initial         # THE Best-of-N property: every group restarts from the seed solution
  n-context: 0                   # and no past solution is injected either
  groups-per-batch: {BON_GROUPS}           # {BON_GROUPS} independent groups ...
  group-size: {CHILDREN}                 # ... x {CHILDREN} children = {TOTAL} candidates, matching every other arm
  num-generations: 1             # nothing flows between generations, so there is nothing to sync
  max-gen-concurrency: {PARENTS}         # semaphore on in-flight requests -> {PARENTS} x {CHILDREN} = {PARENTS*CHILDREN} sequences in flight,
                                 # the same footprint as the other arms, but as a sliding window: a
                                 # group grades while the next one generates.

runs:
""" + "\n".join(runs) + "\n"
    write(f"bon_{t}.yaml", body)

    # ---------------------------------------------------------------- PUCT, no context
    runs = [f"  - {{name: cp26_s{s}, problem: circle_packing_26, seed: {s}}}" for s in range(1, 6)]
    extra = """\
# THIS IS THE BASELINE TO BEAT: TTT-Discover without RL. PUCT picks parents from the buffer, so past
# experience reaches the model ONLY through which solution it is asked to improve -- never as prompt
# context. Comparing this against sweeps/ctx_*.yaml is the whole point of the meeting.
#
# Unlike Best-of-N, this arm GENUINELY needs the generation barrier: generation N+1's parents are
# selected from the buffer that generation N wrote. So it keeps the 15 x 5 x 16 structure.
#
# NOTE it is also the CPU-expensive arm. PUCT selects good parents, the model elaborates them, and eval
# cost rises with generation: measured 25.3 cores held per cp26-PUCT run against 1.6 for Best-of-N, and
# that was at a 220 s timeout which censored 54 % of its evals. At the registry's 530 s expect more.
"""
    body = header(M, "PUCT baseline (TTT-Discover w/o RL), circle_packing_26, 5 runs",
                  extra, mp_puct, 5, TOTAL, f"puct_{t}", f"puct_{t}.yaml", f"puct_{t}")
    body += f"""
sweep:
  name: puct_{t}
  max_parallel: {mp_puct}                # PUCT is the CPU-expensive arm (25 cores/run measured at a 220 s
                                 # timeout, more at 530). Two concurrent runs is deliberately
                                 # conservative; the dial is queue_p50 in progress.csv (~0.02 s = fine,
                                 # seconds = evals are queueing for cores).
  stagger: 180
  server_max_num_seqs: {M['seqs']}
{common(M)}
  # --- search shape --------------------------------------------------------------------------
  parent-source: puct            # parents from the buffer
  n-context: 0                   # but NO context in the prompt -- that is what ctx_*.yaml adds
  groups-per-batch: {PARENTS}
  group-size: {CHILDREN}                 # {PARENTS} x {CHILDREN} = {PARENTS*CHILDREN} candidates/generation
  num-generations: {GENS}            # {TOTAL} candidates/run
  max-gen-concurrency: {PARENTS}         # all {PARENTS} parents' requests in flight at once

runs:
""" + "\n".join(runs) + "\n"
    write(f"puct_{t}.yaml", body)

    # ---------------------------------------------------------------- context strategies
    STRATS = [("random", {"context-strategy": "random"}),
              ("best", {"context-strategy": "best"}),
              ("contrastive", {"context-strategy": "contrastive",
                               "mix-fraction": 0.7, "mmr-lambda": 0.7})]
    runs = []
    runs.append("  # --- cp26, n-context 10, 4 seeds x 3 strategies = 12 runs -------------------------------")
    for s in range(1, 5):
        for tag, fl in STRATS:
            f = {"problem": "circle_packing_26", "seed": s, "n-context": 10, **fl}
            runs.append(f"  - {{name: cp26_n10_s{s}_{tag}, " + ", ".join(f"{k}: {v}" for k, v in f.items()) + "}")

    extra = """\
# THREE STRATEGIES, all with `parent-source: puct`, so the ONLY thing that varies against
# sweeps/puct_*.yaml is what the prompt shows:
#   random      -> CONTROL. Separates "having examples" from "choosing well". Without it a win by
#                  `best` is uninterpretable, so do not drop it to save time.
#   best        -> top-scoring solutions
#   contrastive -> high scorers plus deliberately contrasting low scorers, spread across lineages (MMR)
#
# cp26 at n-context 10, 4 seeds. ac1 (including its n-context 5 vs 10 comparison) is in ac1_<model>.yaml.
#
# WHY n-context IS 10 AND 5, NEVER 30 -- a correctness issue, not a cost one.
#   `build_context_block` (context/selection.py) SILENTLY TRIMS the tail of the block to fit
#   `max-context-tokens`, keeping at least one solution. A PUCT-lineage solution has a MEDIAN size of
#   11,270 chars against 2,081 for a Best-of-N one (5.4x larger -- PUCT elaborates good parents), so 30
#   of them is ~85k tokens against the 94k budget. Trimming would fire almost every generation, and
#   DIFFERENTLY PER ARM, because `best` picks the large elaborate solutions while `contrastive` includes
#   shorter low scorers. The arms would then differ in how many examples they actually show and
#   "n-context 30" would be a fiction. At 10 the block is ~28k tokens median, ~66k even at p90.
#   >>> VERIFY, do not assume: icl.log prints "context=k/N" per parent. If k < N, trimming is happening
#       and the arms are no longer comparable -- lower n-context until k == N.
#
# COST PROFILE differs from the other two files: ~28k-token prompts make prefill ~35 % of the GPU work
# (against ~1 % at n-context 0) and roughly double KV per run, and every arm here is the CPU-expensive
# PUCT kind. That is why max_parallel is 2.
"""
    n = 12
    body = header(M, f"CONTEXT STRATEGIES, circle_packing_26, {n} runs",
                  extra, mp_ctx, n, TOTAL, f"ctx_{t}", f"ctx_{t}.yaml", f"ctx_{t}")
    body += f"""
sweep:
  name: ctx_{t}
  max_parallel: {mp_ctx}                # big prompts (KV + prefill) AND every arm is PUCT (CPU-expensive).
                                 # Raise only once queue_p50 is confirmed ~0 and no server preemptions.
  stagger: 180
  server_max_num_seqs: {M['seqs']}
{common(M)}
  # --- search shape --------------------------------------------------------------------------
  parent-source: puct            # held FIXED across arms; only the prompt context varies
  groups-per-batch: {PARENTS}
  group-size: {CHILDREN}                 # {PARENTS} x {CHILDREN} = {PARENTS*CHILDREN} candidates/generation
  num-generations: {GENS}            # {TOTAL} candidates/run, matching the other two files
  max-gen-concurrency: {PARENTS}

runs:
""" + "\n".join(runs) + "\n"
    write(f"ctx_{t}.yaml", body)


def emit_ac1(M):
    t = M["tag"]
    STRATS = [("random", {"context-strategy": "random"}),
              ("best", {"context-strategy": "best"}),
              ("contrastive", {"context-strategy": "contrastive",
                               "mix-fraction": 0.7, "mmr-lambda": 0.7})]
    runs = ["  # --- Best-of-N, 5 seeds: ONE generation of %d groups, no barrier (see note above) ------" % BON_GROUPS]
    for sd in range(1, 6):
        runs.append(f"  - {{name: bon_s{sd}, seed: {sd}, parent-source: initial, n-context: 0, "
                    f"groups-per-batch: {BON_GROUPS}, group-size: {CHILDREN}, num-generations: 1, "
                    f"max-gen-concurrency: {PARENTS}}}")
    runs.append("  # --- PUCT baseline, n-context 0, 5 seeds ------------------------------------------------")
    for sd in range(1, 6):
        runs.append(f"  - {{name: puct_s{sd}, seed: {sd}, parent-source: puct, n-context: 0}}")
    runs.append("  # --- context strategies at n-context 10, 3 seeds x 3 strategies = 9 runs ----------------")
    for sd in range(1, 4):
        for tag, fl in STRATS:
            f = {"seed": sd, "parent-source": "puct", "n-context": 10, **fl}
            runs.append(f"  - {{name: n10_s{sd}_{tag}, " + ", ".join(f"{k}: {v}" for k, v in f.items()) + "}")
    runs.append("  # --- context strategies at n-context 5, 3 seeds x 3 = 9 runs (context-SIZE comparison) --")
    for sd in range(1, 4):
        for tag, fl in STRATS:
            f = {"seed": sd, "parent-source": "puct", "n-context": 5, **fl}
            runs.append(f"  - {{name: n05_s{sd}_{tag}, " + ", ".join(f"{k}: {v}" for k, v in f.items()) + "}")

    extra = """\
# ALL FOUR ac1 ARMS IN ONE FILE, because ac1 is a single droppable unit against the deadline (see the
# cost note): Best-of-N, the PUCT baseline, and the context strategies at n-context 10 AND 5. Per-run
# overrides give each arm its own shape, so `common` below holds only what they share.
#
# >>> RUN THIS FILE LAST, AND ONLY AFTER MEASURING. <<< ac1 is CPU-bound -- grading was 85-93 % of its
# wall clock, with the A100s sampled at 0 % -- and its per-eval runtime has NEVER been measured (the
# eval instrumentation did not exist when the earlier ac1 runs were stopped). The whole ~39 h estimate
# rests on a derived ~400 s/eval. Launch it, let ONE run reach generation 2, then read `eval_p50` /
# `eval_p90` / `eval_max` from progress.csv before committing to all 28 runs.
#
# HEADROOM CAVEAT, worth saying at the meeting: over the baseline generations available, ac1 moved
# 0.0039 in total and erdos 0.0003. Neither CPU-bound problem has been shown to have enough headroom to
# resolve a strategy effect. If the PUCT and Best-of-N arms are flat across all 15 generations, that is
# a statement about the problem, not about context strategies. ac2 has never been run and is a MAXIMISE
# problem like circle_packing -- the natural candidate for next week.
#
# The seed counts are the ones requested and they are UNDERPOWERED for the strategy arms (3 seeds
# against a noise floor of the same order as the between-strategy spread). Treat ac1 as qualitative
# support -- "the direction seen on cp26 also appears on a different problem family" -- and keep the
# statistical claim on cp26's 4-5 seeds.
#
# WHY n-context 10 AND 5, NEVER 30: `build_context_block` SILENTLY TRIMS the block to fit
# `max-context-tokens`. PUCT-lineage solutions have a median 11,270 chars against 2,081 for Best-of-N,
# so 30 of them is ~85k tokens against the 94k budget -- trimming would fire nearly every generation and
# DIFFERENTLY PER ARM. Verify with "context=k/N" in icl.log; if k < N the arms are not comparable.
"""
    n = 5 + 5 + 9 + 9
    body = header(M, f"ac1 -- ALL ARMS (secondary problem family), {n} runs",
                  extra, 2, n, TOTAL, f"ac1_{t}", f"ac1_{t}.yaml", f"ac1_{t}")
    body += f"""
sweep:
  name: ac1_{t}
  max_parallel: 2                # ac1 is CPU-bound, so extra runs share fixed grading throughput rather
                                 # than adding any. The dial is queue_p50 in progress.csv.
  stagger: 240                   # long generations; keep grading bursts out of phase
  server_max_num_seqs: {M['seqs']}
{common(M)}
  # --- shape shared by the generational arms; the Best-of-N runs override it per run -----------
  problem: ac1
  groups-per-batch: {PARENTS}
  group-size: {CHILDREN}                 # {PARENTS} x {CHILDREN} = {PARENTS*CHILDREN} candidates/generation
  num-generations: {GENS}            # {TOTAL} candidates/run, matching every cp26 arm
  max-gen-concurrency: {PARENTS}

runs:
""" + "\n".join(runs) + "\n"
    write(f"ac1_{t}.yaml", body)


def write(fn, text):
    open(os.path.join(OUT, fn), "w").write(text)
    print("wrote", fn)


for M in (GPTOSS, QWEN):
    emit(M)
    emit_ac1(M)
