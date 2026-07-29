# sol_000138 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000119 (state ab7c4e6b) state=75ab673f sum of radii=2.298212 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_lp_and_marginals(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns radii, objective value, and dual variables (marginals).
    """
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary limits
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    # Construct LP: max sum(r) s.t. r_i + r_j <= dist_ij, r_i <= ub_i, r_i >= 0
    # Linprog minimizes, so obj = -1
    c_obj = -np.ones(n)
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs + 4 * n, n))
    b_ub = np.zeros(num_pairs + 4 * n)
    
    idx = 0
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    # Boundary constraints
    for i in range(n):
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 0]; idx += 1       # x
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 0]; idx += 1 # 1-x
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 1]; idx += 1       # y
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 1]; idx += 1 # 1-y
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if res.success:
            return res.x, -res.fun, -res.ineqlin.marginals
    except Exception:
        pass
    return np.zeros(n), 0.0, np.zeros_like(b_ub)

def obj_and_grad(v):
    """Computes objective (negative sum of radii) and gradient w.r.t centers."""
    centers = v.reshape(N, 2)
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    
    radii, obj, marg = compute_lp_and_marginals(centers)
    if obj < 1e-9:
        return 0.0, np.zeros_like(v)
        
    grad_centers = np.zeros_like(centers)
    
    # Pairwise gradient terms from LP duals
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            lam = marg[idx]
            if lam > 1e-9:
                d = dists[i, j]
                if d > 1e-9:
                    vec = centers[i] - centers[j]
                    force = lam * vec / d
                    grad_centers[i] += force
                    grad_centers[j] -= force
            idx += 1
            
    # Boundary gradient terms
    bound_start = N * (N - 1) // 2
    for i in range(N):
        base = bound_start + 4 * i
        grad_centers[i, 0] += marg[base] - marg[base + 1]
        grad_centers[i, 1] += marg[base + 2] - marg[base + 3]
        
    return -obj, grad_centers.flatten()

def objective_func(v):
    return obj_and_grad(v)[0]

def gradient_func(v):
    return obj_and_grad(v)[1]

def generate_initial_configs():
    """Generates diverse initial center configurations."""
    configs = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.09, 0.095, 0.10]:
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
            c += np.random.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.05, 0.95)
            configs.append(c.flatten())
            
    # Add random starts
    for _ in range(15):
        c = np.random.uniform(0.1, 0.9, (N, 2))
        configs.append(c.flatten())
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_v = None
    best_obj = -np.inf
    bounds = [(0.02, 0.98)] * (2 * N)
    
    starts = generate_initial_configs()
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for v0 in starts:
        try:
            res = minimize(objective_func, v0, jac=gradient_func, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-11})
            if hasattr(res, 'fun') and -res.fun > best_obj:
                best_obj = -res.fun
                best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        best_v = starts[0]
        
    # Phase 2: Adaptive perturbation to escape local minima
    curr_v = best_v.copy()
    for step in range(40):
        noise = 0.006 * (0.84 ** step)
        v_pert = curr_v + np.random.normal(0, noise, curr_v.shape)
        v_pert = np.clip(v_pert, 0.03, 0.97)
        
        try:
            res = minimize(objective_func, v_pert, jac=gradient_func, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-14})
            if hasattr(res, 'fun') and -res.fun > best_obj:
                best_obj = -res.fun
                best_v = res.x.copy()
                curr_v = best_v
        except Exception:
            continue
            
    # Extract final centers and compute exact radii
    centers = best_v.reshape(N, 2)
    radii, sum_r, _ = compute_lp_and_marginals(centers)
    
    # Phase 3: Strict numerical repair to guarantee validation passes
    for _ in range(80):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
