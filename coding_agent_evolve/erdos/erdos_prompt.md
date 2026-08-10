## Minimization of the Erdős Overlap Integral

## The Problem
Find a step function $h: [0, 2] \to [0, 1]$ that **minimizes** the overlap integral:
$$C_5 = \max_k \int h(x)(1 - h(x+k)) dx$$

**Discretization & Constraints**: 
Represent $h$ as `n_points` samples over $[0, 2]$. With $dx = 2.0 / n\_points$:
1. $0 \le h[i] \le 1$ for all $i$
2. $\sum h \cdot dx = 1$ (equivalently: $\sum h == n\_points / 2$ exactly)
3. **Evaluation**: `C5 = max(np.correlate(h, 1-h, mode="full") * dx)`

**Goal**: The current record is $C_5 \le 0.38092$. The milestone to beat is **$0.38080$**, but **your true goal is the absolute minimum possible value**. Lower is always better. Do not stop iterating just because you hit a specific number; push the mathematical and computational limits to drive the value as low as possible.

## Search Strategy

**Search framework**: Do not get stuck fine-tuning a single approach that stalls at a local minimum. Try several diverse approaches in parallel to ensure there is enough exploration of new ideas but also exploitation/optimization of promising ones. 

- **Diverse Initial Solutions**: Begin with a genuinely diverse portfolio of approaches to explore substantially different formulations (e.g., various search algorithms and optimization models). 
- **Registry**: Maintain an explicit registry of approach families. Group your scripts by the mathematical idea they use, not by superficial wording. If multiple scripts converge to one family, redirect your efforts toward underexplored formulations.
- **Cross-Pollination**: Cross-pollinate ideas only after independent branches have been developed far enough to expose their real strengths and gaps.
- **Creative & Novel Streams**: Standard techniques are a good baseline, but dedicate distinct branches of your search to creative and unconventional approaches you have never seen before.
- **Critical Analysis**: Be critical of approaches you have proposed. Analyze exactly why they work or fail, and use those insights to inform the next generation of solutions.

Return only when a solution surpassing the best-known solution is found and you feel you cannot improve things further. Remember, the goal is to get the lowest possible value.

## Environment, Disk, & Compute
- **Compute**: You can use 12 out of 44 CPU cores available.
- **Execution Limit**: Any optimization script you run must finish within 1100 seconds and can use up to two CPU cores to run.
- **Venvs & Disk Space**: You have a Python virtual environment (`.claude.venv`) where you can install necessary libraries. You can create more venvs if strictly necessary for conflicting libraries, but try to avoid this. **You are limited to 1-2 GB maximum** for all installations and venvs combined. Install standard tools (`scipy`, `numpy`, `cvxpy`, solvers like CBC/GLPK/GUROBI, etc.) or any other libraries useful for solving this problem.
- You can use `eval.py` to check the performance of the solution you found. 

## Output & Reproducibility
- Save files that allow me to understand how things evolved and run your best candidate solutions.

## Web Search Rules
You are allowed to search the web for existing solutions, numerical approaches to the Erdős constant, or algorithmic inspiration. 
- **Requirement**: Maintain a `research_log.md` file. Every time you use something from the web, log the URLs you visited and write a brief note on whether the search was useful or not, and why.
- **Balance**: Use the web for inspiration, but do not let it constrain you. Ensure you are spending significant time running the "Creative & Novel Streams" mentioned above, rather than just copying standard literature.