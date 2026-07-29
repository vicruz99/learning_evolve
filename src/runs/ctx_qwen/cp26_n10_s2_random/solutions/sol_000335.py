# sol_000335 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000297 (state 4b6f1fd1) state=6fcc43b1 sum of radii=2.385522 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint structure
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[k, i] = 1.0
        A_LP[k, j] = 1.0
        PAIR_IDX.append((i, j))
        k += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and computes exact subgradient via duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b_ub[k] = dists[i, j]
        k += 1
    for i in range(N):
        b_ub[k] = centers[i, 0]; k += 1
        b_ub[k] = 1.0 - centers[i, 0]; k += 1
        b_ub[k] = centers[i, 1]; k += 1
        b_ub[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(N), 0.0, np.zeros_like(centers)
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b_ub))
    try:
        duals = np.asarray(res.marginals.ineqlin)
    except AttributeError:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except Exception:
            pass
            
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def lbfgs_wrapper(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = x_flat.reshape(N, 2)
    _, val, g = solve_lp_and_grad(c)
    return -val, -g.reshape(-1)

def optimize_lbfgs(c0, max_iter=5000):
    """Optimizes centers using L-BFGS-B with exact LP gradient."""
    bounds = [(1e-6, 1.0 - 1e-6)] * (2 * N)
    try:
        res = minimize(lbfgs_wrapper, c0.flatten(), jac=True, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def generate_starts(rng):
    """Generates diverse initial configurations for multi-start optimization."""
    starts = []
    
    # 1. Hexagonal lattice patterns
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], [6,6,5,5,4], [5,5,6,5,5]]
    for pat in pats:
        for r0 in [0.090, 0.094, 0.098, 0.102, 0.106]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.02, 0.98))
            
    # 2. Force-directed repulsion layouts
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(1200):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.22 and d > 1e-4:
                        push = (0.22 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            # Boundary repulsion
            for i in range(N):
                if c[i,0] < 0.12: f[i,0] += 0.025
                if c[i,0] > 0.88: f[i,0] -= 0.025
                if c[i,1] < 0.12: f[i,1] += 0.025
                if c[i,1] > 0.88: f[i,1] -= 0.025
            c += f * 0.012
            c = np.clip(c, 0.02, 0.98)
        starts.append(c)
        
    # 3. Corner-focused starts
    corners = [[0.06, 0.06], [0.94, 0.06], [0.06, 0.94], [0.94, 0.94]]
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = corners
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B
    for c_init in starts:
        c_opt, s_opt = optimize_lbfgs(c_init, 6000)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
    best_r, best_s, _ = solve_lp_and_grad(best_c)
    
    # Phase 2: Simulated Annealing with local L-BFGS-B refinement
    curr_c = best_c.copy()
    curr_s = best_s
    T = 0.007
    for step in range(700):
        noise_scale = 0.009 * (0.994 ** (step // 40))
        c_try = curr_c + rng.normal(0, noise_scale, curr_c.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        
        # Short refinement from perturbed state
        c_ref, s_ref = optimize_lbfgs(c_try, 1500)
        
        delta = s_ref - curr_s
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            curr_c = c_ref
            curr_s = s_ref
            if curr_s > best_s:
                best_s = curr_s
                best_c = curr_c.copy()
        T *= 0.996
        
    # Phase 3: Targeted Swap & Perturbation Search
    for _ in range(40):
        c_trial = best_c.copy()
        
        # Random pair swap to break permutation symmetry
        idx_swap = rng.choice(N, 2, replace=False)
        c_trial[idx_swap] = c_trial[idx_swap[::-1]]
        
        # Local perturbation
        idx_pert = rng.choice(N, rng.integers(3, 8), replace=False)
        c_trial[idx_pert] += rng.normal(0, 0.006, (len(idx_pert), 2))
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        c_opt, s_opt = optimize_lbfgs(c_trial, 2000)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    # Phase 4: Final LP solve & Repair
    best_r, _, _ = solve_lp_and_grad(best_c)
    final_r = repair(best_c, best_r)
    final_sum = float(np.sum(final_r))
    
    return best_c, final_r, final_sum
