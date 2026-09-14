## Search Strategy

Treat this as a search over *approaches*, not as a single program you polish. Do not get stuck
fine-tuning one idea that has stalled in a local optimum. Run several diverse approaches in
parallel, so that there is genuine exploration of new ideas as well as exploitation of the
promising ones.

- **Diverse initial solutions.** Begin with a genuinely diverse portfolio: substantially different
  formulations, not restatements of one idea. Different search algorithms, different optimization
  models, different discretizations, different objective relaxations.
- **Registry of approach families.** Maintain an explicit registry in `run/LEDGER.md`, grouping
  candidates by the *mathematical idea* they use, not by superficial wording. If several candidates
  converge onto one family, deliberately redirect effort into an underexplored formulation.
- **Independence before cross-pollination.** Let independent branches develop far enough to expose
  their real strengths and gaps before you merge ideas across them. Merging too early collapses the
  portfolio into a single family.
- **Creative and novel streams.** Standard techniques are a good baseline, but keep distinct
  branches dedicated to unconventional approaches you have not seen used for this problem.
- **Critical analysis.** Be critical of your own proposals. For every candidate, say exactly *why*
  it worked or failed, and feed that into the next generation. A failed candidate is search signal,
  not waste — record its concrete failure cause (invalid solution / too slow / numerically
  unstable / converged to a worse optimum).
- **Blocked routes.** When an approach stalls, mark it BLOCKED in the ledger together with the
  evidence that blocked it (a measurement, a resource limit, repeated verified regressions).
  Reopen it only when you have a genuinely new mechanism that addresses that evidence.
- **Measure, do not assume.** Every claim about why something is better or worse needs a scored
  number behind it. An unscored candidate is not progress.

## Bookkeeping

- **`run/LEDGER.md`** — one row per candidate: approach family, one-line description, valid yes/no,
  official {{ P.metric_word }}, delta vs the current champion, verdict. Keep the candidate programs in
  `run/attempts/`, numbered in the order you ran them; never renumber or delete one.
- **`run/NOTES.md`** — a running narrative of what you learned about the problem itself, separate from
  the per-candidate ledger.
