# Experiment Plan

> **Purpose.** The concrete experimental roadmap for the project. For motivation, research
> questions, and definitions of methods / models / benchmarks / baselines, see
> `PROJECT_CONTEXT.md`; this file assumes that background and does not repeat it.
>
> **Read `## 1. Current experiment` first** — it is the live task. Everything after it is
> planned but not yet in progress.

---

## 1. Current experiment (in progress)

**Setup:** ICL, **low-level** representation, on both models (Qwen3.6-27B, GPT-oss-120B),
over the math benchmark problems. This is the ICL + low-level cell of Experiment 1 (§2),
run first before RL and SFT.

**Math problems (5).** Circle packing, first autocorrelation, second autocorrelation, and
**two more still to be chosen**. Each starts from the **same initial solution TTT-Discover
used** — these need to be pulled from the TTT-Discover setup.

**The context / truncation challenge.** For a fully fair comparison with TTT-Discover we'd
like all tried solutions in context, but TTT-Discover generates **25,600** solutions per
run — far beyond any context window. So a **selection / truncation strategy** is needed for
what actually goes in the prompt. Candidate strategies:

- best-so-far solutions,
- most-recent solutions,
- a mixture of diverse + high-reward solutions.

Realistically only a **few hundred** solutions fit. Both the strategy and the exact number
are **still to be defined**. Context budgets to work within: Qwen3.6-27B ~260k tokens,
GPT-oss-120B ~131k tokens.

**Cost framing.** Large-context ICL is compute-heavy, but so is RL — comparing them at a
fixed budget is the intended fair framing (see `PROJECT_CONTEXT.md` §4.7).

**Parallel task — ShinkaEvolve.** Run ShinkaEvolve with the **same two models** and
**equal initial solutions**, to add an evolutionary-search reference point.

**Immediate to-dos for this experiment:**

- [ ] Pick the remaining 2 math problems.
- [ ] Retrieve TTT-Discover's initial solution for each problem and reuse it.
- [ ] Decide the in-prompt selection/truncation strategy.
- [ ] Decide the max number of in-prompt solutions.
- [ ] Set up the ShinkaEvolve runs with matched models and initial solutions.
- [ ] Check if it's possible to recover the buffer and PUCT strategy directly from TTT-Discover repo

---

## 2. Aditional Core experiments

- **ICL with autocompaction** (check the research group's work on this), a ReasoningBank-
  style memory module, or similar — mechanism TBD.
- SFT - do online SFT with low level solutions. Try Success-only vs failure-only vs combined trajectories.
- Do SFT on higher level info

---

## 3. Secondary experiments

- **Generalization under perturbation.** Adapt to one instance, freeze weights, test on a
  nearby modified instance — transferable strategy vs overfitting. For RL and SFT (maybe
  across scale).
- **Robustness to noisy rewards.** Inject synthetic noise into evaluator feedback to
  simulate proxy / binary rewards; compare degradation across RL/SFT/ICL.


---

## 4. SFT design (relative to the RL baseline)

The RL baseline (TTT-Discover) is described in `PROJECT_CONTEXT.md` §5.1. Design notes for
the SFT variant:

- **Cadence.** Mirror RL: each generation, fine-tune on the solutions generated that
  generation — a per-generation **self-distillation**.
- **What to train on.** RL pushes **toward** positive-reward and **away from**
  negative-reward solutions. SFT can fine-tune on **top / positive-reward** solutions. Open
  question: can we also do **"negative" fine-tuning** on negative-reward solutions — and if
  so, does that collapse the practical difference from RL?
- **Objective.** RL uses an entropic objective (GRPO). The SFT analogue is plain likelihood
  on selected traces, possibly weighted.
- **Core distinction.** The biggest difference is that **SFT lets us choose which answers to
  train on** — the lever the data ablations (Experiment 3) exploit.

**Discussion input:**

- **Tim:** if SFT (train on top-k, or on everything) to reinforce the model in this
  direction beats RL, that's a win. Their **SERA** results suggest it didn't much matter —
  training on everything was fine.
- **Bingqing:** consider **weighted-advantage SFT**, sitting between SFT and RL; could help
  if we want to incorporate the entropic objective. *(To investigate.)*
- **Working intuition (possible later study):** the model may not be "learning" so much as
  **raising the probability of structurally-similar-to-good answers**, so that across its 64
  attempts per generation it spends more of them on things that already work.

---

## 5. Later-stage / exploratory ideas

- **Replace the LLM with a coding agent** (possibly a separate paper).

---

## 6. Open questions / TODOs (project-wide)

- [ ] **Define the fixed compute budget** precisely (core to the whole comparison).
- [ ] Define the ICL truncation strategy + max in-prompt solution count *(see §1)*.
- [ ] Pick the remaining 2 math benchmark problems *(see §1)*.
- [ ] Retrieve and reuse TTT-Discover's initial solutions *(see §1)*.
- [ ] Decide the **high-level summarizer** (Experiment 2): fixed external LLM, the same base
      model, or a separate distilled memory module (e.g., ReasoningBank).
- [ ] Choose the **kernel optimization** benchmark (beyond TTT-Discover's TriMul).
- [ ] Choose the **open-ended AI discovery** benchmark.
- [ ] Decide whether the **search strategy itself** is a controlled variable (archive policy,
      novelty pressure, parent sampling — plus prompt history and the human-chosen initial
      solution — can strongly affect outcomes).
- [ ] Be more specific about **how SFT is carried out** (objective, weighting, selection).
- [ ] Investigate **weighted-advantage SFT** (Bingqing's suggestion).
- [ ] Confirm the TTT-Discover numbers (8 × 64 = 512 per generation; 25,600 total) against
      the paper before relying on them.
