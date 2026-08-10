## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \max_k \int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)


**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.
Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.

## Budget & Resources
- **Time budget**: if you write code that helps you find the constructions, the code must have a limit of 1100s to run
- You have available N_CPU_CORES to run solutions you propose, if you propose programs that need to run.
- You are working in a venv you can install libraries if you want or need

## Rules
- If you write programs to help find your solution, I want to be also be able to run those programs later.
- You can use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math, or other libraries you install

