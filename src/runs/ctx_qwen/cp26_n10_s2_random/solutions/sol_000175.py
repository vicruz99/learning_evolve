# sol_000175 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000147 (state da2cd853) state=55d5ff70 sum of radii=2.499217 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def get_lp_structure():
    """Pre-construct the LP constraint matrix structure."""
    n_pairs = N * (N - 1) // 2
    n_bound = 4 * N
    A_ub = np.zeros((n_pairs + n_bound, N))
    pairs = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pairs.append((i, j))
            idx += 1
    for i in range(N):
        for _ in range(4):
            A_ub[idx, i] = 1.0
            idx += 1
    return A_ub, pairs

A_ub_pre, PAIRS = get_lp_structure()
NUM_PAIRS = len(PAIRS)

def solve_lp_and_gradient(centers):
    """
    Solves LP for optimal radii given fixed centers and computes gradient
    of sum of radii w.r.t. center positions using LP dual variables.
    """
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    
    ub = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(A_ub_pre.shape[0])
    idx = 0
    for i, j in PAIRS:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = x[i]; idx += 1
        b_ub[idx] = 1.0 - x[i]; idx += 1
        b_ub[idx] = y[i]; idx += 1
        b_ub[idx] = 1.0 - y[i]; idx += 1
        
    bounds_r = [(0, u) for u in ub]
    res = linprog(-np.ones(n), A_ub=A_ub_pre, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return None, None, None
        
    radii = res.x
    duals = res.ineqlin.marginals
    sum_r = np.sum(radii)
    
    grad = np.zeros_like(centers)
    
    idx = 0
    for i, j in PAIRS:
        lam = duals[idx]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    boundary_start = NUM_PAIRS
    for i in range(n):
        grad[i, 0] += duals[boundary_start + 4*i] - duals[boundary_start + 4*i + 1]
        grad[i, 1] += duals[boundary_start + 4*i + 2] - duals[boundary_start + 4*i + 3]
        
    return radii, sum_r, grad

def obj_func(v):
    """Objective: minimize negative sum of radii."""
    _, s, _ = solve_lp_and_gradient(v.reshape(N, 2))
    if s is None:
        return -1.0
    return -s

def grad_func(v):
    """Gradient of objective w.r.t. flattened centers."""
    _, _, grad = solve_lp_and_gradient(v.reshape(N, 2))
    if grad is None:
        return np.zeros_like(v)
    return -grad.flatten()

def force_directed_init(rng, seed=None):
    """Generate initial configuration using repulsion-based relaxation."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    
    centers = rng.uniform(0.1, 0.9, (N, 2))
    radii = np.full(N, 0.035)
    
    for _ in range(600):
        radii *= 1.001
        forces = np.zeros_like(centers)
        
        for i in range(N):
            cx, cy = centers[i]
            r = radii[i]
            
            if cx < r: forces[i, 0] += (r - cx) * 20.0
            elif cx > 1 - r: forces[i, 0] -= (cx - (1 - r)) * 20.0
            if cy < r: forces[i, 1] += (r - cy) * 20.0
            elif cy > 1 - r: forces[i, 1] -= (cy - (1 - r)) * 20.0
            
            for j in range(i + 1, N):
                dx = cx - centers[j, 0]
                dy = cy - centers[j, 1]
                d = np.hypot(dx, dy)
                if d < r + radii[j] and d > 1e-9:
                    f = (r + radii[j] - d) * 4.0 / d
                    fx, fy = dx * f, dy * f
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    
        centers += forces * 0.004
        centers = np.clip(centers, 0.01, 0.99)
        
    return centers

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 6], [6, 6, 4, 5, 5], [5, 6, 4, 5, 6],
        [5, 5, 5, 5, 5, 1], [6, 5, 5, 4, 6], [5, 7, 5, 5, 4],
        [6, 5, 6, 5, 4], [5, 6, 5, 5, 5], [4, 5, 5, 6, 6]
    ]
    
    for pat in patterns:
        for r_est in [0.085, 0.09, 0.095, 0.10, 0.105, 0.11]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.03, 0.97)
            starts.append(c.flatten())
            
    for _ in range(12):
        c = rng.uniform(0.1, 0.9, (N, 2))
        starts.append(c.flatten())
        
    for seed in range(10):
        c = force_directed_init(rng, seed)
        starts.append(c.flatten())
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize sum of radii."""
    rng = np.random.default_rng(42)
    best_centers = None
    best_sum = -1.0
    best_radii = None
    
    starts = generate_starts(rng)
    bounds_c = [(0.01, 0.99)] * (2 * N)
    
    # Phase 1: L-BFGS-B with gradient from LP duals from multiple starts
    for i, v0 in enumerate(starts):
        try:
            res = minimize(obj_func, v0, method='L-BFGS-B', jac=grad_func, 
                           bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_gradient(c_opt)
            if s_opt is not None and s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Powell refinement to handle non-smooth objective regions
    if best_centers is not None:
        try:
            res = minimize(obj_func, best_centers.flatten(), method='Powell', bounds=bounds_c,
                          options={'maxiter': 20000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_gradient(c_opt)
            if s_opt is not None and s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Perturbation search to escape local minima
    for _ in range(120):
        v_trial = best_centers.flatten().copy()
        v_trial += rng.normal(0, 0.002, v_trial.shape)
        v_trial = np.clip(v_trial, 0.02, 0.98)
        try:
            res = minimize(obj_func, v_trial, method='L-BFGS-B', jac=grad_func,
                           bounds=bounds_c, options={'maxiter': 1200, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_gradient(c_opt)
            if s_opt is not None and s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 4: Simulated Annealing for global exploration
    c_curr = best_centers.copy()
    s_curr = best_sum
    T = 0.004
    for step in range(4000):
        T *= 0.9992
        c_new = c_curr + rng.normal(0, T, c_curr.shape)
        c_new = np.clip(c_new, 0.02, 0.98)
        r_new, s_new, _ = solve_lp_and_gradient(c_new)
        if s_new is None:
            continue
            
        if s_new > s_curr:
            c_curr = c_new
            s_curr = s_new
            if s_curr > best_sum:
                best_sum = s_curr
                best_centers = c_curr.copy()
                best_radii = r_new.copy()
        else:
            if rng.random() < np.exp((s_new - s_curr) / max(T * 2.0, 1e-7)):
                c_curr = c_new
                s_curr = s_new

    # Phase 5: Final high-precision L-BFGS-B polish
    try:
        res = minimize(obj_func, best_centers.flatten(), method='L-BFGS-B', jac=grad_func,
                       bounds=bounds_c, options={'maxiter': 4000, 'ftol': 1e-14})
        c_opt = res.x.reshape(N, 2)
        r_opt, s_opt, _ = solve_lp_and_gradient(c_opt)
        if s_opt is not None and s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
    except Exception:
        pass

    # Phase 6: Strict Numerical Repair for Validator Compliance
    centers = best_centers.copy()
    radii = best_radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
