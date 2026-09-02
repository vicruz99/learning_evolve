# TriMul / H100 — the coding-agent arm (Marvin)

The H100 sibling of `../b200_task/`. Read that README for the design (no seed kernel, why the
published kernel is kept out of the run folder, why one card is never enough when vLLM shares the
box). Everything here is the same except:

- `INITIAL_PROMPT.md` carries `_HW_RULE_H100` from `src/envs/kernel_trimul.py` **verbatim** — the
  single line TTT-Discover used — instead of `_HW_RULE_B200`, so this arm sees exactly what the
  `trimul_h100` ICL runs see. The Blackwell toolchain bullet is dropped (irrelevant on sm_90).
- Scores pool with the `trimul_h100` ICL arm and compare to TTT-Discover's 1161 µs. The
  reference kernel measures **1182 µs** on Marvin's H100s (5 repeats, spread 0.6 %;
  `src/gpumode_local/reference/README.md`).
- On Marvin the LLM lives on rng-dl01 (reached through `llmtun`), so the job needs only ONE card:
  the grader's. Launch with `../../local_model/jobs/marvin_agent.bsub`, headless.

```bash
./make_run.sh ~/agent_runs/trimul_h100_qwen1        # or let marvin_agent.bsub do it
```
