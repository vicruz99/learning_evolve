import numpy as np

from envs.base import Environment, objective_only_prompt
from sandbox import SandboxRewardEvaluator
from puct import State


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")
    
    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")
    
    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")
    
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    
    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)
    
    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    
    dx = 2.0 / n_points
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)
    
    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")
    
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    
    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result: tuple[np.ndarray, float, int]) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


class ErdosMinOverlapRewardEvaluator(SandboxRewardEvaluator):
    def get_program_entrypoint(self) -> str:
        return "run"

    def preprocess_generation(self, generation, state) -> str:
        import inspect
        verifier_src = inspect.getsource(verify_c5_solution)
        numpy_import = "import numpy as np"
        
        base = numpy_import + "\n\n" + verifier_src + "\n\n"
        
        # State with construction is required - no silent fallback
        if state is None:
            raise ValueError(
                "state is required for preprocess_generation. "
                "Use ExperienceSampler to provide initial state with construction."
            )
        if state.construction is not None:
            initial_h_values = f"initial_h_values = np.array({list(state.construction)!r})"
            base += initial_h_values + "\n\n"

        return base + generation

    def get_reward(self, code: str, state: State) -> float:
        output, error_msg, failure_type = self.execute_code(code, state)
        if error_msg:
            return self._get_failure_entry(error_msg, failure_type=failure_type)

        if not verify_erdos_solution(output):
            return self._get_failure_entry("Invalid solution.", failure_type="invalid_result")
        h_values, c5_bound, n_points = output
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)

        return {
            "reward": float(1.0 / (1e-8 + c5_bound)),
            "correctness": 1.0,
            "raw_score": c5_bound,
            "msg": f"C5 bound: {c5_bound}",
            "result_construction": list(h_values),
            "stdout": getattr(self, '_last_stdout', ''),
        }


class ErdosMinOverlapEnv(Environment):
    reward_function = ErdosMinOverlapRewardEvaluator
    state_type = State
    max_construction_len = 1000
    # Upstream uses an unseeded rng here; we fix a seed for reproducible initial solutions.
    initial_state_seed = 12345

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        rng = np.random.default_rng(cls.initial_state_seed)
        n_points = rng.integers(40, 100)
        construction = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        construction = construction + perturbation
        dx = 2.0 / n_points
        correlation = np.correlate(construction, 1 - construction, mode="full") * dx
        c5_bound = float(np.max(correlation))
        return State(timestep=-1, code="", value=-c5_bound, construction=list(construction))

    def is_maximize(self) -> bool:
        return False # Minimize upper bound

    # --- Prompt zone 1: constant motivation + problem/rules (cached across parents) ---
    def problem_intro(self) -> str:
        return '''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.
Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

'''

    # --- Prompt zone 3: improvement guidance + the current construction to improve (rendered LAST) ---
    def improvement_task(self) -> str:
        state = self.initial_state
        state_ctx = state.to_prompt(0.3808, metric_name="C₅ bound", maximize=False)
        has_construction = (hasattr(state, 'construction') and state.construction is not None
                            and len(state.construction) > 0)

        # Construct construction section (mentions n, which varies per parent). Kept even when no
        # parent solution is shown: `initial_h_values` is handed to the sandbox by the evaluator
        # either way, so the prompt would otherwise hide a global the code can actually use.
        construction_section = ""
        if has_construction and not self.show_parent_solution:
            construction_section = f"""An initial construction is available through the `initial_h_values` global variable (n={len(state.construction)} samples). You may start your search from it or from anywhere else.
"""
        elif has_construction:
            construction_section = f"""You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

        # Construct code section (improvement guidance; constant once we have a prior solution)
        if not self.show_parent_solution:
            code_section = f'''Write code that searches for the best construction you can find.

{objective_only_prompt(0.3808, metric_name="C₅ bound", maximize=False)}
'''
        elif state.code and state.code.strip():
            code_section = f'''Your task is to reason about how you could further improve the current construction and propose an improvement to the current solution. You may want to consider different algorithmic ideas, adjusting your heuristics, or sweeping your hyperparameters.
Unless you make a meaningful improvement, you will not be rewarded.

--- Current solution to improve upon ---
{state_ctx}
'''
        else:
            code_section = f'''Write code to optimize this construction.

--- Current construction to improve upon ---
{state_ctx}
'''

        return f'''
{construction_section}
{code_section}

## Budget & Resources
- **Time budget**: 1000s for your code to run
- **CPUs**: 2 available

## Rules
- Define `run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
'''
