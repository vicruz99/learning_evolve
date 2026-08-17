# ShinkaEvolve baseline — portable copy (for Bosch replicates)

Everything needed to repeat the guadiana ShinkaEvolve baseline runs (AC1 / AC2 /
Erdos min-overlap, local Qwen3.6-27B, 2400 generations) on another machine. The
outer repo gitignores `ShinkaEvolve/` (it's a nested clone), so the run-defining
files are tracked here instead:

| file | what |
|---|---|
| `ttt_discover_math/` | the three problem ports: `run_evo.py`, `evaluate.py`, `initial.py`, and the run configs `shinka_qwen.yaml` (base) / `shinka_qwen_meta.yaml` (meta scratchpad on). `RUNS.md` documents the original guadiana campaign. |
| `shinka_local.patch` | the 5 local fixes to the shinka package the runs depend on: eval kill-clock starts at eval (not proposal) time, embed only a 5000-char prefix (nomic's 2048-token context), `extra_body` passthrough so `thinking_token_budget: 12000` reaches vLLM, and `reasoning`/`reasoning_content` compat. |
| `setup_shinka.sh` | one-shot: clone upstream ShinkaEvolve pinned to `b67a073`, apply the patch, symlink `examples/ttt_discover_math` back to this dir. Prints the venv commands. |

On **guadiana** the live checkout at `../../ShinkaEvolve` is the original — this dir
is a committed copy of it (keep them in sync if the yamls change). On **Bosch**,
`setup_shinka.sh` makes this dir the single source of truth via the symlink.

## Running the replicates on Bosch

```bash
cd ~/work/projects/phd/learning_evolve
bash src/shinka_baseline/setup_shinka.sh     # then build ShinkaEvolve/.venv as it prints

# point the yamls at the chat server: after vllm_server.bsub is RUNNING, replace
# localhost with the node from src/jobs/vllm_host.txt in every shinka_qwen*.yaml
# (llm_models and meta_llm_models entries), e.g.
#   sed -i "s|@http://localhost:8001/v1|@http://$(cat src/jobs/vllm_host.txt):8001/v1|" \
#       src/shinka_baseline/ttt_discover_math/*/shinka_qwen*.yaml

cd src && mkdir -p jobs/logs
bsub < jobs/vllm_server.bsub    # chat model; wait for jobs/vllm_host.txt
bsub < jobs/vllm_embed.bsub     # embedding server; wait for jobs/vllm_embed_host.txt
for p in ac1 ac2 erdos_min_overlap; do
  PROBLEM=$p REP=bosch_r1 bsub < jobs/shinka_run.bsub                                # base arm
  PROBLEM=$p REP=bosch_meta_r1 CONFIG=shinka_qwen_meta.yaml bsub < jobs/shinka_run.bsub  # meta arm
done
```

Details (queues, resume-by-resubmitting, throughput math) are in the headers of
`src/jobs/shinka_run.bsub` and `docs/BOSCH_CLUSTER.md`.

## "Different seeds"

Shinka has no RNG-seed knob for the search: replicates differ through LLM sampling
(temperature 1.0) alone. To repeat a run as a new seed, just launch the same config
with a fresh `REP` label — the label only names the results dir. Use `bosch_*`
labels so results stay distinguishable from the guadiana runs when analyses merge.
