# Project Context: Incorporating Past Experience in Open-Ended Discovery

> **Purpose.** Stable background for this project — the motivation, research questions,
> and definitions of the moving parts (methods, models, benchmarks, baselines). It does
> **not** describe the experiment plan or current tasks; see `EXPERIMENT_PLAN.md` for that.

---

## 1. One-line summary

Under a **fixed test-time compute budget**, what is the most effective way to incorporate
**past experience** into open-ended LLM-driven discovery: **in-context learning (ICL)**,
**reinforcement learning (RL)**, or **supervised fine-tuning (SFT)** — and how does this
depend on the **representation level** of past experience and on **model scale**?

---

## 2. Motivation

Recent work in automated discovery (AlphaEvolve, ShinkaEvolve, TTT-Discover) shows LLMs can
propose solutions that sometimes surpass the state of the art. These systems pair an LLM
with an **evaluator** in an iterative loop: the evaluator executes each proposal and returns
an objective metric, and the LLM uses those scores to guide its next attempts, turning
discovery into a **search** problem. Performance generally improves when models build on
prior solutions, so most systems keep an **archive** and use a selection strategy
(evolutionary algorithms, PUCT, tree search) to pick which candidate to improve next.

Two broad ways to feed past experience back into the model:

- **Prompt-based memory (frozen model):** AlphaEvolve and ShinkaEvolve expose past solutions
  or high-level summaries in the prompt — in-context learning over the search trajectory.
- **Parameter updates (test-time training):** TTT-Discover and ThetaEvolve update the model
  during search via RL, training on the distribution of past solutions using evaluator
  reward. These report test-time RL matching or beating static-agent frameworks with
  **smaller** models, at the added cost of RL.

**The gap.** These approaches all exploit past experience but are hard to compare — they
differ in model size, task family, prompt structure, search strategy, and the amount /
abstraction level of history. So we do not know whether gains come from **memory** or from
**learning**. In RL systems no history sits in the prompt, so any benefit may come purely
from internalizing experience during training — raising the question of whether ICL or SFT
could recover the same effect more cheaply. There is little prior work using **SFT** in this
kind of agentic framework, which is a key thing this project explores.

---

## 3. Research questions

**Central question:** under a fixed test-time compute budget, what is the most effective
mechanism for incorporating past experience into open-ended discovery — ICL, RL, or SFT?

Three axes of variation:

1. **Learning method** — RL vs ICL vs SFT.
2. **Representation of past experience** — low-level (full trajectories, program edits,
   execution feedback) vs high-level (summaries, strategy analyses, distilled knowledge).
3. **Model size** — how the relative benefits shift with model capacity.

Sub-questions:

1. Given identical historical trajectories, which approach gives the best
   performance–efficiency trade-off: conditioning via prompt (ICL), RL, or SFT?
2. If learning beats prompting, can **SFT on search traces recover RL's gains** as a more
   stable / cheaper alternative?
3. In SFT/ICL, does the model benefit more from **successful** trajectories, **failed**
   ones, or **both**? (RL trains on all; SFT/ICL let us choose.)
4. What is the best representation of past experience — low-level signals or high-level
   abstractions? (Only studyable for ICL and SFT.)
5. How does the relative effectiveness of ICL/RL/SFT scale with model size? Is there a
   regime where explicit memory (ICL) suffices and another where parameter updates win?

Additional (nice-to-have) questions:

6. Does an adapted model acquire **transferable** optimization strategies or overfit to the
   local search landscape? (ThetaEvolve hints at transfer; worth checking for SFT too.)
7. How robust are RL/SFT/ICL to **noisy / proxy / binary rewards** (e.g., LLM-based
   evaluators)?
8. In AlphaEvolve’s evolutionary algorithm, island-based evolution is a feature of their algorithm. So some solutions are evolved mostly independently. I wonder if a similar thing could be done in TTT-Discover. Instead of training only one model, could it be interesting to train more than one model, where each model sees mostly different groups of solutions. Could this enable faster specialization or creativity and in turn improved final solutions?
9. In RL, the model may not really learning but more increasing the probability of giving answers that are structurally similar to already good ones, and then from the 64 opportunities it has to test certain things, it uses more of those opportunities to test things that already work well. Perhaps this could be a later study.

---

## 4. Methodology background

The base agentic framework is held **fixed** so that only the learning mechanism and the
representation of history vary. The scaffold mirrors **TTT-Discover** (the primary
comparison point), which uses a relatively simple search compared to ShinkaEvolve.

### 4.1 Search strategy — PUCT (from TTT-Discover)

The only search strategy in use. Each state `s` is scored by:

```
Q(s) + c · P(s) · sqrt(1 + T) / (1 + n(s))
```

- `Q(s)` = **maximum** reward among states generated when `s` was the initial state (or
  `R(s)` if `s` has not been selected yet). Max, not mean — we care about the best outcome
  from a state, not the average.
- `P(s)` ∝ `s`'s rank in the reward-sorted buffer (high-reward states are more likely to
  seed high-reward children).
- `n(s)` = times `s` or its descendants have been expanded.
- `T` = total expansions. `c` = exploration coefficient (keeps under-visited states alive).

### 4.2 Learning methods

- **ICL:** frozen model; trajectory history inserted into the prompt.
- **RL:** weights updated during search; **no** explicit history in the prompt.
- **SFT:** model trained on collected trajectories; we choose *which* traces to train on.

### 4.3 Representation levels

- **Low-level:** full trajectories, program edits, execution feedback.
- **High-level:** distilled summaries, strategy-level regularities, repeated failure modes,
  useful structural patterns — produced by a separate summarization/critique model.

### 4.4 Models

Chosen partly to enable fair comparison with TTT-Discover. Model scale is a planned axis of
the broader study but is not the near-term focus.

| Model         | Context window | Notes                                         |
|---------------|----------------|-----------------------------------------------|
| Qwen3.6 – 27B | ~260k tokens   | Mid-scale                                     |
| GPT-oss 120B  | ~131k tokens   | Matches the model size used by TTT-Discover   |

A fuller scaling sweep (e.g., 1.5B/3B, 7B/8B, 32B/70B, 120B) may come later.

### 4.5 Benchmarks

**Math discovery problems** — fast, clear evaluators; directly comparable to AlphaEvolve /
ShinkaEvolve / TTT-Discover. For each problem, the **same initial solution as TTT-Discover**
should be used to keep comparisons fair.

Later-stage benchmark families (not near-term):

- **Kernel optimization** (runtime/throughput; TTT-Discover reports TriMul gains).
- **Open-ended AI-oriented discovery** — less structured, more research-like.

### 4.6 Training

TTT-Discover uses **LoRA** for efficiency; we follow the same route given compute demands.

### 4.7 Fixed compute budget

The comparison is framed at a **fixed test-time compute budget**: large-context ICL is
compute-heavy, but so is RL, so comparing them at a fixed budget is the fair framing. The
precise definition is still to be set.

---

## 5. Baselines

### 5.1 TTT-Discover (primary RL baseline)

- Stores solutions in a **buffer** (not an evolutionary framework like AlphaEvolve /
  ShinkaEvolve); uses **PUCT** to sample which solution to expand.
- Each generation: select **8** solutions from the buffer; the LLM generates **64** new
  candidates per selected solution ⇒ **512** new solutions per generation.
- Applies **GRPO** to the groups of 64 with an **entropic objective**; within each group
  some solutions get positive reward, others negative.
- Total solutions across a run: **25,600**.

### 5.2 ShinkaEvolve

An evolutionary-search baseline, run with the same models and the same initial solutions.

---

## 6. Appendix — findings from related LLM-search papers

*(Background reading; not required to understand the project.)*

- **Search strategy (AIRA / AIRA2):** best strategy depends on compute regime and
  infrastructure. MCTS does well under moderate budgets but can slide into overfitting;
  greedy can win with longer search; **evolutionary** wins in massively parallel multi-GPU
  settings (suits asynchronous workers, shares discoveries via a population). Parallelism
  alone isn't enough — a **shared state** turns past successes into useful guidance.
- **Model capability (AIRA / AIRA2):** stronger models traverse the search space more
  efficiently and reach a higher ceiling; reasoning/reflection raises the ceiling.
- **Tree search vs LLM-guided (GOME):** tree search scales with inference compute and is
  more robust when reasoning is weak; LLM-guided optimization scales with model capability
  and becomes more attractive as reasoning improves.
- **How novel should proposals be? (LLMs as Local Optimizers):** best-performing models are
  usually **not** the most novel — they make local, controlled refinements near a promising
  region. Overly disruptive proposals tend to break useful parent structure.
- **Search architecture (LEVI):** stronger orchestration can substitute for larger models —
  initial architectural diversity, routing small edits to small models and large structural
  changes to large models, smaller validation subsets when validation is expensive. Initial
  program diversity is a key driver.
- **What evolutionary coding agents change (EvoTrace / EvoReplay):** a **frequency–utility
  gap** — most budget goes to hyperparameter tuning / local refinement, while rare edits
  drive most gains. **Deterministic cycling** (re-adding byte-identical previously-removed
  code) is common. Many breakthroughs are **structural, not lexical**; post-hoc Bayesian
  optimization over numeric constants can sometimes match a full evolutionary run.
- **Overfitting & evaluation:** agent performance is highly **stochastic** — use several
  seeds for reliable rankings; systems can overfit the validation metric, so contamination
  control and careful benchmark design matter.
- **Shared pattern:** success generally comes from a strong model + a structured search that
  keeps promising solutions alive, refines them, and shares progress; papers disagree on
  whether gains come mostly from search architecture, model capability, or local refinement.
