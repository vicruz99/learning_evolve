# Throughput knobs: vLLM server + `run_icl.py`

> What actually moves wall-clock time, and what only looks like it does. Numbers are measured on
> **gpt-oss-120b, 2×A100, circle_packing_26, `reasoning-effort medium`, 6 parents × 15 = 90
> candidates/gen, `n_context=20`** (`runs/_sweep_c15_g6`, 739 s/gen). Re-measure before trusting any
> of this on `ac`/`erdos` or at much larger contexts.

## The one thing to internalise first

**This workload is decode-bound and the GPU is already saturated.**

| Where the 739 s of a generation goes | |
|---|---|
| Decode (token-by-token generation) | ~all of it, at ~700–1100 tok/s aggregate |
| Prefill (6 parents × 17.2k-token prompts ≈ 104k tok) | ~30–50 s, **~5 %** |
| Grading (sandbox) | cheap here; the barrier idles the GPU, it does not extend generation |

Two consequences that kill most "obvious" optimisations:

- The 2×A100 saturates at **~20 concurrent sequences**. Going from 20 → 90 does not speed anything up
  (measured: 762 s vs 746 s). So flags that buy you *more concurrency or more KV room* buy you nothing.
- **The only real lever on a single run is generating fewer tokens.** Everything else is either noise
  or a way to stop wasting the GPU between generations.

An earlier note in `IMPLEMENTATION_LOG.md` claimed prefill dominates. That was measured at
`n_context=30` (~103k prompts). At `n_context=20` prefill is ~5 %.

---

## vLLM server

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/scratch/vicstorage \
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --gpu-memory-utilization 0.92 \
    --max-model-len 48000 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --reasoning-parser openai_gptoss \
    --port 8001
```

| Flag | Recommendation | Why / expected effect |
|---|---|---|
| `--max-model-len` | **Size it to `largest_prompt + max_tokens`** (48000 at `n_context=20`) | 131000 is the one setting worth changing regardless. **Risk:** at `n_context=30` prompts reached ~103k, so 103k + 26k = 129k sits 2k under the ceiling — one context-growth away from truncated or rejected requests. Cap the prompt with `--max-context-tokens` too. |
| `--reasoning-parser` | **Add it** (`openai_gptoss` for gpt-oss, `qwen3` for Qwen3) | Separates the chain of thought from the answer, so `--save-reasoning` writes traces and code extraction can't pick up a fence from the reasoning. **Note the field name differs by parser** — `openai_gptoss` emits `reasoning`, the qwen/deepseek parsers emit `reasoning_content`; the client accepts either. Verify with a raw `curl` of `/v1/chat/completions` before concluding a field is missing. |
| `--max-num-seqs` | ≥ your peak in-flight sequences (128 is plenty) | Below the peak, the excess queues server-side instead of co-batching. Above it, harmless. |
| `--max-num-batched-tokens` | 16384 | Helps the prefill fraction only, so ~1–2 % overall. Cheap, do it, don't expect more. |
| `--tensor-parallel-size` | 2 for one shared server | MoE TP scaling is sublinear, but 2×TP1 servers would split the KV pool and 63 GB of MXFP4 weights on an 80 GB card leaves ~12 GB for KV — too tight for 17k-token prompts at batch. One TP=2 server shared by several runs beats two TP=1 servers. |
| `--enable-prefix-caching` | Leave on (V1 default) | Real, but see the caching section below — it saves far less than you'd hope. |
| `--kv-cache-dtype fp8` | Only for `n_context ≥ 30` | Doubles KV capacity. At `n_context=20` you're compute-bound, not KV-bound, so it's noise. Verify it starts at all: gpt-oss uses attention sinks + sliding window and FP8 KV is storage-only on sm80. |
| `--async-scheduling` | Keep | Already in use. |
| `--speculative-config` (n-gram) | **Don't bother** | Tempting (the model largely reproduces the parent's code) but it trades compute for fewer sequential steps, which only pays when memory-bandwidth-bound at low batch. At 90 concurrent sequences you're compute-bound → neutral to negative. |

### Server-side reality check

Hit `/metrics` while a run is going:

```bash
curl -s localhost:8001/metrics | grep -E "num_requests_(running|waiting)|prefix_cache_(queries|hits)_total"
```

`num_requests_waiting > 0` means `--max-num-seqs` or client concurrency is the binding constraint.

---

## `run_icl.py`

| Flag | Recommendation | Why / expected effect |
|---|---|---|
| `--reasoning-effort` | `medium`; **A/B `low`** | The highest-leverage untested knob. Decode volume *is* wall-clock, so if `low` roughly halves reasoning tokens without hurting the best score, it's close to a 2× speedup. `high` is strictly bad: ~70 % `no_code` because completions exhaust the budget on hidden reasoning. |
| `--max-tokens` | 26000 → **set near the measured `decode_p99`** | A request returns only when its **slowest** sequence finishes, so this sets the worst case for every parent. Any completion that hits the cap burns the full budget and usually emits no code. Read the `decode/candidate` log line (or `decode_p99` in `progress.csv`) from one generation and size off it — *not* off the mean, which sits below the median on this workload. |
| `--max-gen-concurrency` | `groups_per_batch × group_size` | Anything ≥ 20 is within 3 % of optimal. Only `conc=6 + chunk=1` genuinely starves the GPU. Not worth tuning. |
| `--grade-chunk-size` | Default (off) for circle_packing; `1` for `ac`/`erdos` | Its only job is overlapping grading with generation. Where generation ≫ grading it buys ~nothing (measured 7 s *slower*). Keep `1` in mind for observability (live `[k/15 graded]`) at ~3 % cost. |
| `--eval-timeout` | 530 → **120** for circle_packing | p95 of valid solves is 72 s; 530 is 7× dead weight on the per-generation barrier whenever a straggler appears. Use `results/recheck_failures.py` to see what you'd lose first. |
| `--n-context` / `--max-context-tokens` | Science variable; cap the tokens | 20 costs ~5 % in prefill; 30 costs ~15–30 % *and* risks the context ceiling above. Set `--max-context-tokens` as a guard rail even when `n_context` is the thing you're varying. |
| `--no-exclude-parent` + `--context-seed N` | Only if you want a shared prefix | Makes the context block identical across a generation's parents so vLLM prefills it once instead of once per parent. Costs prompt diversity between parents. See caching below. |
| `--save-completions` / `--save-reasoning` | Both on | Disk, not speed. `--save-reasoning` needs the server's reasoning parser. |
| `--num-cpus-per-task` | Leave to the registry | `ac1`/`ac2` reserve 2 cores; probably overkill, but unverified — check on the first real `ac` run. |

### Prefix caching: what it's actually worth

Measured, so you don't re-litigate it:

- **Within a parent:** all `n` children of one request share the prompt and are prefilled once, for
  free, already. (Confirmed: with `n=2` the cache hit rate on the second sequence is ~100 %.)
- **Across parents:** the longest common prefix is **500 tok / 2.9 %** of a 17.2k prompt, because each
  parent drops itself from its own block (`exclude_id`) and tie-breaking is random per parent. The
  2026-07-25 prompt reordering therefore buys almost nothing cross-parent as currently configured.
- `--no-exclude-parent --context-seed N` is what makes the block genuinely shared (~16k of the 17k
  prompt), which is worth ~5 % at `n_context=20` and more like 15 % at `n_context=30`. It is a
  **science trade** (less prompt diversity between a generation's parents), not a free win.
- "Priming the prefix" (send one request, await, then fan out) is only worth building on top of a
  shared block. Without one there is nothing to prime.

### Across-generation limits

Generations are strictly sequential — PUCT needs the graded buffer before it can pick the next
parents — so no amount of tuning overlaps generation *N+1* with *N*. What you *can* do is stop
wasting the GPU during each generation's grading barrier: run **2 experiments against one server**
(`max_parallel: 2` in a sweep file). vLLM co-batches across them, so one run's generation fills the
GPU while the other grades. This is the largest available campaign-level win.

---

## Ranked: what to do next

1. **A/B `--reasoning-effort low` vs `medium`** on score, not just speed. Potentially ~2×.
2. **Read `truncated` / `tokens_per_completion` from one generation, then cut `--max-tokens`.**
3. **Run 2 experiments per server** (`run_sweep.py`, `max_parallel: 2`).
4. **`--eval-timeout 120`.**
5. **Set `--max-model-len 48000` and add `--reasoning-parser openai_gptoss`** on the server.
6. Everything else (fp8 KV, batched tokens, chunking, concurrency) is ≤ a few percent at current
   settings — set them once and stop thinking about them.

## Instrumentation to read the answers off

Each generation logs, and `progress.csv` / `summary.json` / `events.jsonl` persist:

```
gen 1 tokens | prompt 4,954 (75% cached) | decode 24,515 (3064/completion, 147 tok/s) | reasoning 12,084 + answer 12,431 (49% of decode is reasoning) | truncated 0/8
gen 1 decode/candidate | p50 3,021 p90 4,298 p99 4,298 max 4,298 | max_tokens=8,000 (54% used by the longest) — size --max-tokens off p99, the tail gates each request
```

Everything except the labelled `/completion` mean and the rates is a **total** for the generation.

- **`reasoning` / `answer` split** — measured, not estimated: this vLLM build omits
  `usage.completion_tokens_details`, so the reasoning text is counted with the served model's own
  tokenizer via the server's `/tokenize`. Cost is ~0.7 s per 90-candidate generation (CPU-only on the
  API server, overlapped with grading) ≈ 0.1 %. If the endpoint is unavailable it falls back to a
  chars÷4 estimate, marked `~… est`, and warns once. Measured on circle_packing_26 at
  `--reasoning-effort medium`: **reasoning is 49–70 % of all decode**, which is why
  `--reasoning-effort` is the biggest single speed lever. (The chars÷4 estimate ran ~23 % low, so use
  the real counts.)
- **`decode/candidate` percentiles** — exact per-candidate `reasoning_tokens` + `answer_tokens` from
  `events.jsonl`, summarised per generation (`decode_p50/p90/p99/max` in `progress.csv`). This is the
  distribution to size `--max-tokens` from. Observed spread within one generation: **6.7×** (1,009 →
  6,747 tokens), and the mean sat *below* the median — the tail is invisible in averages.
  Kept out of the `usage` block deliberately: usage fields are summed into run totals, and summing
  percentiles is meaningless.
- `truncated` — completions with `finish_reason == "length"`. Non-zero fires a warning.
- If a whole generation returns no reasoning text while `--save-reasoning` is on, a one-time warning
  names the parser flag to add. That is a **correctness** matter, not just a lost trace: without a
  parser the chain of thought stays inside `content`, and code extraction takes the last ```python
  fence there — so a completion that fences code while thinking but not in its answer yields code
  parsed out of the reasoning.
- `% cached` — prefix-cache hit rate from the server's `/metrics`. **Server-global** (mixes concurrent
  runs) and counted **per sequence**, so `cache_queries ≈ n × prompt_tokens` — only the ratio means
  anything.
- `wall_seconds` per generation is in `progress.csv`, so cost per generation is reconstructable.
