PROBLEM
Optimize the TriMul (Triangle Multiplicative Update) kernel from the GPU MODE
competition. The operator takes a pair representation X ∈ R^{B×N×N×C} and computes:
LayerNorm → five gated linear projections (with optional masking) → an O(N³)
pairwise contraction over the N dimension → LayerNorm → gated output projection.

Deployment target: a SINGLE NVIDIA RTX 3090 (GA102, SM86, 24GB). Every candidate
must be a complete solution that runs entirely on one GPU. This machine has four
identical 3090s, but they exist only to parallelize the search: multi-GPU
execution, sharding, or any solution that uses more than one device at runtime
is out of scope and counts for nothing.

Evaluation is defined entirely by the frozen harness described in the HARNESS
section below. Score = geometric mean runtime across the fixed benchmark shapes
in benchmarks.txt, executed on ONE 3090, as computed by score.py. Lower is
better.

Before any optimization work: benchmark the unmodified reference implementation
on GPU 0 under the measurement protocol below and record it as the initial
champion in LEDGER.md. All progress is measured as verified speedup over the
current champion. There is no external target time; the objective is to drive
the champion's single-GPU time down as far as possible.

SUCCESS CRITERIA
A candidate is promoted to champion if and only if it:
- Runs entirely on one GPU, visible-devices-restricted, with no cross-device
  communication or work delegated to other devices or the host beyond what the
  harness itself does.
- Passes ALL correctness tests in the harness at the official tolerances, on
  all test shapes, including masked cases and edge shapes, across seeds.
- Improves the geometric-mean benchmark runtime over the current champion by
  more than measurement noise (see MEASUREMENT PROTOCOL).

The following count for NOTHING, no matter how fast:
- Any use of a second device at runtime, however indirect.
- Kernels that fail any correctness test, on any shape or seed.
- Speedups on a subset of shapes that regress the geometric mean.
- Numerical shortcuts that pass by luck on one seed but violate tolerance on
  others.
- Timing hacks: caching outputs across calls, moving required work outside the
  timed region, exploiting harness artifacts. The kernel must do the real work.
- Claimed speedups without harness-verified numbers.

Unlike a proof problem, incremental progress is valuable: every verified
improvement becomes the new champion and the new baseline to beat.

HARNESS (frozen — read this before anything else)
- The evaluation harness lives at /home/robodata/projects/code-discovery/trimul:
  eval.py, reference.py, task.py, utils.py, tests.txt, benchmarks.txt, score.py.
  These files are FROZEN at the current commit. No agent may modify them for
  any reason — including OOM errors, which are a property of this 24GB
  hardware, not a harness bug. utils.py has already been patched (comparison
  on CPU, reference under no_grad) so the reference passes all shapes on a
  3090; if the harness errors on a candidate, the candidate is at fault until
  proven otherwise.
- Environment: use the project venv at
  /home/robodata/projects/code-discovery/.venv for everything. Install
  packages only via `uv pip` or `python -m pip` — bare `pip` escapes the venv
  on this machine.
- Invocation (run from the directory containing the candidate's submission.py;
  the harness imports the module named `submission` from the CWD):
    CUDA_VISIBLE_DEVICES=<gpu> POPCORN_FD=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_CACHE_DIR=/home/robodata/projects/code-discovery/.triton-cache \
    python /home/robodata/projects/code-discovery/trimul/eval.py <mode> <file>
  Modes, with <file> resolved against the trimul directory:
    test tests.txt        — correctness only
    benchmark benchmarks.txt — exploration timing
    leaderboard benchmarks.txt — promotion-grade: re-checks correctness with
      fresh seeds on every timing repetition, so output caching is caught here
    profile benchmarks.txt — torch.profiler tables (base64-encoded reports)
  Always include all three env vars; never vary them between runs that will be
  compared.
- Scoring: the harness logs per-case stats (benchmark.N.mean etc., in ns) but
  does NOT aggregate. The score is the geometric mean of the per-case means,
  computed by score.py. All ledger entries and promotion decisions use
  score.py's output, in µs.
- The harness already warms up and adaptively repeats each case (3–100 runs,
  until relative error < 0.1% or a time cap), reporting mean/std/err/best/
  worst per case. Do not build additional timing loops around it; the
  remaining measurement discipline lives in MEASUREMENT PROTOCOL.
- Every submission's custom_kernel MUST run under @torch.no_grad(). Gradients
  are not part of the task, and without it, retained autograd state OOMs the
  checker on the largest shapes. A candidate that OOMs is examined before it
  is buried: an OOM raised inside custom_kernel is the candidate's own memory
  appetite (a legitimate BLOCKED signal for that design family); an OOM raised
  inside check_implementation or the reference while verifying an otherwise-
  working kernel means the candidate left GPU memory resident (module-level
  caches, missing no_grad). Record which, from the traceback, in the ledger.
- Submissions using @triton.autotune must pin or cache their winning configs;
  a submission that re-sweeps configs on every process launch is mismeasured
  and will be treated as its slow first-call self.

WORKSPACE PROTOCOL (Claude Code)
- Maintain best/submission.py as the current champion. It is only ever
  replaced by a candidate that passed the full correctness suite AND beat it
  under the promotion procedure below, in the same session. Never edit it in
  place; replace atomically.
- Maintain LEDGER.md: one row per candidate — approach family, one-line
  description, correctness pass/fail, score.py result (geomean µs and per-case
  spread), delta vs champion, GPU used, and verdict. Every subagent appends to
  it. OOM verdicts must record kernel-OOM vs checker-OOM per the HARNESS
  section.
- Each subagent works in its own subdirectory (or git worktree) containing its
  candidate as submission.py. Subagents never edit shared files except
  appending to LEDGER.md, and never touch the frozen harness files.
- Preserve failed candidates in attempts/ with the concrete failure cause
  (wrong results vs slow vs compile error vs resource limit). Failure causes
  are search signal and must be recorded, not discarded.

MEASUREMENT PROTOCOL
- The four GPUs parallelize the search, not the solution. Every run — whether
  exploratory or promotional — executes the candidate on exactly one device,
  isolated via CUDA_VISIBLE_DEVICES so a candidate cannot see, or accidentally
  use, a second GPU.
- GPUs 1–3 are exploration devices: subagents use them, one subagent per GPU,
  for correctness runs and rough benchmarking of independent candidates in
  parallel.
- GPU 0 is the adjudication device. All promotion decisions are made on GPU 0
  only, by running champion and candidate back-to-back in the same session
  using `leaderboard` mode via score.py. Numbers measured on different GPUs,
  or at different times, are never compared for promotion: nominally identical
  cards differ by a few percent from thermals and silicon variance, which is
  the same order as the margins being adjudicated. Exploration numbers from
  GPUs 1–3 are triage signal only — sufficient to kill a candidate, never
  sufficient to promote one.
- A candidate wins only if its geomean improvement over the champion's
  fresh, same-session number exceeds the run-to-run spread score.py reports.
  Never compare against a stale champion number.
- Clocks on GPU 0 are locked before the session (nvidia-smi -lgc, done by the
  operator; agents do not need or use sudo). If clock locking is ever absent,
  interleave champion/candidate runs so both see the same thermal state; a
  cold-vs-warm comparison is invalid.
- Verify correctness BEFORE benchmarking. Never report a time for a kernel
  that has not passed the full test suite in its current form. Re-verify after
  any edit, however small.

MULTIAGENT SEARCH MANAGEMENT
Use subagents aggressively and dynamically. Do not use a fixed assignment such
as "N agents for strategy X." Manage the search using the following heuristics:

- Begin with a genuinely diverse portfolio of approaches. Agents should explore
  substantially different fusion structures, decompositions of the computation,
  data layouts, scheduling and parallelization strategies, numerical
  strategies, implementation backends, delegation-to-library formulations,
  and empirical characterizations of where the time actually goes. Let
  measurement on this machine, not prior assumptions about the hardware,
  determine which directions are promising.
- Do not tell most agents the currently favored approach. Preserve independence
  during early rounds so that agents do not all converge to the same attractive
  but suboptimal design.
- Maintain an explicit registry of approach families in LEDGER.md. Group
  candidates by the underlying optimization idea they embody, not by
  superficial code differences. If many agents converge to one family, redirect
  some of them toward underexplored formulations.
- Do not allow one approach to dominate merely because it produced the first
  verified speedup. A design that wins on some benchmark shapes and loses on
  others has not settled anything; the geometric mean across all shapes is the
  only judge.
- When an approach stalls at a hard limit — evidenced concretely by profiles,
  resource exhaustion, tolerance violations, or repeated verified regressions —
  mark that route as BLOCKED in the ledger together with the blocking evidence.
  Only continue assigning agents to it if someone proposes a materially new
  mechanism that addresses the recorded evidence.
- Keep several incompatible implementations alive through multiple rounds.
  Cross-pollinate ideas only after independent agents have developed them far
  enough to expose their real strengths and gaps; any hybrid must re-verify
  correctness from scratch.
- Require agents to return concrete artifacts: runnable code, ledger rows with
  harness-verified numbers, and measured evidence for any claim about WHY
  something is fast or slow (the harness's profile mode is available for this).
  Reject status reports, vague optimism, and "should be faster" reasoning. An
  unmeasured kernel is not progress.
- Never promote on an author-agent's claim. Promotion is gated by a mechanical
  procedure the orchestrator runs itself: copy the candidate file exactly as
  submitted into a clean adjudication directory as submission.py, run the full
  correctness suite fresh, then run the promotion benchmark on GPU 0
  back-to-back with the current champion via score.py. Reported numbers from
  any agent are triage signal only; they never feed a promotion decision.
- The harness defines correctness but not legality. Before promotion, one
  agent that did not write the candidate reads the diff for rule violations
  the harness cannot detect: caching or state carried across calls, work moved
  outside the timed region, use of a second device, or exploitation of harness
  artifacts. This is the only remaining adversarial role; it reviews code
  against the rules, not results against the tests.