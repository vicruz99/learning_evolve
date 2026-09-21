## Search Strategy

Treat this as a search over *approaches*, not as a single program you polish, and treat **the
number of officially scored candidates as your primary output**. Runs like this one usually fail
the same way: the agent spends hours refining two or three ideas, scores a handful of
constructions, and never finds out what the other twenty ideas would have done. Do not be that run.
A search that scores many diverse candidates finds the good basins; a search that polishes one
candidate finds one local optimum.

{% if MIN_EVALS %}- **Volume target.** Aim for **at least {{ MIN_EVALS }} officially scored candidates** over the run —
  that is about {{ EVALS_PER_HOUR }} per hour — and treat that as a floor, not a goal. The host counts
  official evaluations; it will tell you how many you have made when it checks in.
{% else %}- **Volume matters.** There is no fixed quota, but the host counts official evaluations and will tell
  you how many you have made when it checks in; a run that has scored hundreds of diverse candidates
  has learned more about the problem than one that has scored ten.
{% endif %}
- **Time-box every candidate.** From idea to official score should take **at most 20 minutes of your
  own attention**. If a candidate is not scored within that, score whatever it has produced so far
  and move on; you can always come back to it with a new mechanism.
- **Keep the machine full.** You can hold **{{ PARALLEL }} candidates in flight at once**. Whenever fewer
  than that are running, launch more. Do not sit and wait for one program to finish — launch the next
  one, then check back.
- **Breadth before depth.** Start with a genuinely diverse portfolio: substantially different
  formulations, search algorithms, optimization models, discretizations, objective relaxations.
  Generate many variants of each promising idea (different seeds, resolutions, hyper-parameters,
  starting points) rather than one carefully tuned instance — variants are cheap and each one is a
  scored data point.
- **Score first, judge second.** A candidate you have not scored officially is not evidence of
  anything. Score every construction you produce, including the ones you expect to be worse; the
  distribution of scores is what tells you where to search next.
- **Registry of approach families.** Maintain `run/LEDGER.md`, grouping candidates by the
  *mathematical idea* they use. If several candidates converge onto one family, deliberately
  redirect effort into an underexplored formulation.
- **Critical analysis, briefly.** For every candidate, one line on *why* it scored the way it did,
  then the next candidate. A failed candidate is search signal, not waste — record its concrete
  failure cause (invalid solution / too slow / numerically unstable / converged to a worse optimum).
- **Blocked routes.** When an approach stalls, mark it BLOCKED in the ledger with the evidence, and
  spend the freed capacity on a new family. Reopen it only with a genuinely new mechanism.
- **Cross-pollinate late.** Once several families have produced scored candidates, combine the best
  elements across them — but only after each has been explored on its own.

## Bookkeeping

- **`run/LEDGER.md`** — one row per candidate: approach family, one-line description, valid yes/no,
  official {{ P.metric_word }}, delta vs the current champion, verdict. Keep the candidate programs in
  `run/attempts/`, numbered in the order you ran them; never renumber or delete one.
- **`run/NOTES.md`** — a running narrative of what you learned about the problem itself, separate from
  the per-candidate ledger.
