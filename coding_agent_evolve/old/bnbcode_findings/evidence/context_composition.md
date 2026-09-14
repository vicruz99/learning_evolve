# Context composition — how it was measured

Both arms tokenised with the **same** tokenizer, `Qwen/Qwen3.6-27B-FP8`, via the
`tokenizers` package reading `$HF_HOME/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/tokenizer.json`.
Counts are of *everything accumulated in the session*, not of one context window — so
read them as composition, not as "what the model saw on turn N".

## bnbcode — `ac2_evo_s1`, session `ses_fac771e7effe3pOzSfM05IBIUo`

Source: the `part` table for that session id.

| category | tokens | blocks | share | per block |
|---|---|---|---|---|
| user / injected text | 138,091 | 44 | **68%** | 3,138 |
| tool calls (the request) | 27,712 | 41 | 14% | 676 |
| assistant reasoning | 27,403 | 79 | 13% | 347 |
| tool results (the output) | 8,222 | 41 | 4% | 200 |
| assistant text | 1,632 | 49 | 1% | 33 |
| **total** | **203,060** | | | |

## bnbcode — `ac2_plain_s1`

| category | tokens | share |
|---|---|---|
| tool calls | 29,022 | 41% |
| tool results | 21,862 | 31% |
| user / injected text | 19,416 | 27% |
| assistant text | 1,073 | 2% |
| **total** | **71,373** | |

(Smaller injected share because this session had 9 injected blocks rather than 44 — it was
younger. The per-block size is the same.)

## Claude Code — `ac2_evo_cc_s1`, session `c13b43bb-…`

Source: the transcript JSONL. **One line per content block**, with `message.id`, `usage`
and `stop_reason` repeated on every block of the same response — group by `message.id`
before counting responses, or a thinking+text+tool_use answer counts as three.

| category | tokens | blocks | share | per block |
|---|---|---|---|---|
| assistant reasoning | 178,409 | 173 | 37% | 1,031 |
| tool calls (the request) | 155,052 | 183 | 32% | 847 |
| attachments | 78,156 | 234 | 16% | 334 |
| tool results (the output) | 52,095 | 183 | 11% | 285 |
| user / injected text | 15,973 | 10 | 3% | 1,597 |
| assistant text | 1,434 | 71 | 0% | 20 |
| **total** | **481,119** | | | |

Cumulative API usage over that run: input 8,867,434, output 337,123. Every assistant
response carries `model: "Qwen/Qwen3.6-27B-FP8"` (427 of 427) — the control arm really is
the same model, reached through LiteLLM.

## The two numbers that matter

- bnbcode spends **68%** of its accumulated context on repeated harness text and **4%** on
  the output of the programs the agent ran. Claude Code spends **3%** and **11%**.
- Qwen reasons **347 tokens per block** under bnbcode and **1,031** under Claude Code —
  three times less thinking per turn in the arm that stalls ten times more often.

## Gotchas that cost time

- bnbcode reasoning parts store their text in `data->>'text'`. `data->>'reasoning'` does
  not exist, and reading it reports 0 tokens across dozens of populated blocks.
- Claude Code repeats `usage` on every content-block line; summing it without grouping by
  `message.id` inflates cumulative tokens ~2.5x.
- The two harnesses' `count_tokens` and vLLM's `/tokenize` disagree on short strings
  (28 vs 31 on a 21-token probe) because they wrap messages differently; on a 3,500-token
  body they agree to 0.1%. Calibrate on a long body, never a short one.
