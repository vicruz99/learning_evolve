# ShinkaEvolve baseline runs — checklist (local on guadiana)

cd ~/work/projects/phd/learning_evolve/ShinkaEvolve/examples/ttt_discover_math
export LOCAL_OPENAI_API_KEY=local SHINKA_PRICING_MODE=offline

# Attached (foreground) launches, one terminal/tmux window each. NO nohup, NO trailing
# `&` — a stray nohup'd copy once double-wrote ac1's results dir. tee saves the log.

# AC1 (meta)
cd ac1 && ../../../.venv/bin/python run_evo.py \
    --config_path shinka_qwen_meta.yaml \
    --results_dir results/ac1_qwen_meta_r1 \
    --embedding_model "local/nomic-embed-text:latest@http://localhost:11434/v1" \
    2>&1 | tee ac1_qwen_meta_r1.log

# AC2 (meta)
cd ac2 && ../../../.venv/bin/python run_evo.py \
    --config_path shinka_qwen_meta.yaml \
    --results_dir results/ac2_qwen_meta_r1 \
    --embedding_model "local/nomic-embed-text:latest@http://localhost:11434/v1" \
    2>&1 | tee ac2_qwen_meta_r1.log

# Erdos (meta)
cd erdos_min_overlap && ../../../.venv/bin/python run_evo.py \
    --config_path shinka_qwen_meta.yaml \
    --results_dir results/erdos_qwen_meta_r1 \
    --embedding_model "local/nomic-embed-text:latest@http://localhost:11434/v1" \
    2>&1 | tee erdos_qwen_meta_r1.log









12 runs total, all against the local vLLM server on `localhost:8001`
(`Qwen/Qwen3.6-27B`). Per problem: **3 replicates** of the base config
(`shinka_qwen.yaml`, meta scratchpad off) **+ 1 run** with the meta scratchpad on
(`shinka_qwen_meta.yaml`). Embeddings (Shinka's duplicate gate) are on in every run via a
local embedding server. Tick a box when the run reaches generation 2399 / prints its final
summary and exits 0.

## Checklist

| done | problem | config | rep | results dir |
|------|---------|--------|-----|-------------|
| [ ] | ac1 | shinka_qwen.yaml | r1 | `ac1/results/ac1_qwen_r1` |
| [ ] | ac1 | shinka_qwen.yaml | r2 | `ac1/results/ac1_qwen_r2` |
| [ ] | ac1 | shinka_qwen.yaml | r3 | `ac1/results/ac1_qwen_r3` |
| [ ] | ac1 | shinka_qwen_meta.yaml | meta_r1 | `ac1/results/ac1_qwen_meta_r1` |
| [ ] | ac2 | shinka_qwen.yaml | r1 | `ac2/results/ac2_qwen_r1` |
| [ ] | ac2 | shinka_qwen.yaml | r2 | `ac2/results/ac2_qwen_r2` |
| [ ] | ac2 | shinka_qwen.yaml | r3 | `ac2/results/ac2_qwen_r3` |
| [ ] | ac2 | shinka_qwen_meta.yaml | meta_r1 | `ac2/results/ac2_qwen_meta_r1` |
| [ ] | erdos_min_overlap | shinka_qwen.yaml | r1 | `erdos_min_overlap/results/erdos_qwen_r1` |
| [ ] | erdos_min_overlap | shinka_qwen.yaml | r2 | `erdos_min_overlap/results/erdos_qwen_r2` |
| [ ] | erdos_min_overlap | shinka_qwen.yaml | r3 | `erdos_min_overlap/results/erdos_qwen_r3` |
| [ ] | erdos_min_overlap | shinka_qwen_meta.yaml | meta_r1 | `erdos_min_overlap/results/erdos_qwen_meta_r1` |

## Prerequisites (once)

1. **Chat server** on `localhost:8001` serving `Qwen/Qwen3.6-27B` with
   `--reasoning-parser qwen3` (the usual run_shinka setup — the thinking cap was verified
   enforced on this build: budget 64 → 69 completion tokens).
2. **Embedding server: the CPU ollama server** (`CUDA_VISIBLE_DEVICES="" ollama serve`,
   port 11434) with **`nomic-embed-text:latest`** — kept off the GPUs on purpose, and the
   load is trivial anyway (one ≤10k-char code snippet per generation). Verified working
   through Shinka's embed client (`local/...@.../v1` scheme, dim 768, cost 0).
   - Heads-up: nomic embeddings are spread out — near-identical code scores ~0.93 cosine —
     hence `code_embed_sim_threshold: 0.96` in the yamls (Shinka's 0.99 default never fires).
   - **Context limit**: the nomic GGUF hard-rejects inputs past its 2048-token context
     (~5.5k chars of code; typical proposals are 10–25k chars). Shinka is patched to embed
     only the first 5000 chars (`shinka/edit/async_apply.py::get_code_embedding_async`,
     was 10000) so embeds succeed at all — the duplicate gate compares 5k-char prefixes.
   - If the server is down or an embed fails, runs DON'T crash — Shinka logs an embedding
     error and skips the duplicate gate for that proposal. Check occasionally:
     `grep -ic "error.*embedding" <results_dir>/evolution_run.log` (should stay 0).

## Launching a run

```bash
cd ~/work/projects/phd/learning_evolve/ShinkaEvolve/examples/ttt_discover_math
export LOCAL_OPENAI_API_KEY=local SHINKA_PRICING_MODE=offline

PROBLEM=ac1 REP=r1 CONFIG=shinka_qwen.yaml          # <- vary per checklist row
# erdos results dirs are named erdos_* (not erdos_min_overlap_*), matching the yaml:
NAME=$( [ "$PROBLEM" = erdos_min_overlap ] && echo erdos || echo "$PROBLEM" )

cd "$PROBLEM" && ../../../.venv/bin/python run_evo.py \
    --config_path "$CONFIG" \
    --results_dir "results/${NAME}_qwen_${REP}" \
    --embedding_model "local/nomic-embed-text:latest@http://localhost:11434/v1" \
    2>&1 | tee "${NAME}_qwen_${REP}.log"
```

Attached on purpose (survives nothing, shows everything): run each inside its own tmux
window so an SSH drop doesn't kill it. Never mix in `nohup`/`&` — a leftover detached
copy writing to the same results dir corrupts the run (two processes number generations
independently into the same `gen_N` dirs).

- **Resume = re-run the same command** (same results dir): Shinka continues from
  `programs.sqlite`. Safe after crashes, reboots, or server restarts.
- **Progress:** `sqlite3 <results_dir>/programs.sqlite 'select max(generation) from programs;'`
  — done at 2399.
- **Concurrency:** run **up to 6 at once**. Sharing the one decode-bound server across runs
  is efficient (vLLM batches concurrent requests — same reason the ICL sweeps share a
  server). CPU-side, evals are hard-capped at 2 OMP/BLAS threads each
  (`numeric_threads_per_job: 2`), so a run's worst case is 5 evals × 2 ≈ 10–11 cores and
  6 runs ≤ ~66 of the 96 cores — and the typical draw is far lower because the LLM, not the
  evaluator, paces each run. If `uptime`'s 15-min load stays well under ~70 you can add the
  remaining runs as others finish.

## Analysis caveats

- The seed program appears **twice at generation 0** (once per island), and **each resume
  re-inserts it** (one more gen-0 row). Count tried solutions from proposal generations,
  not raw DB rows.
- True LLM call count (incl. patch resamples and, in the meta arm, scratchpad calls) is in
  `<results_dir>/gen_*/attempts/**/metadata.json`.
