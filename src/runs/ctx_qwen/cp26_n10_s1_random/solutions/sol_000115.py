# sol_000115 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=e6e4381b sum of radii=2.507511 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_constraints(centers, t):
    """Computes inequality constraints for a given set of centers and equal radius t."""
    # Boundary constraints: x >= t, 1-x >= t, y >= t, 1-y >= t
    c = np.concatenate([
        centers[:, 0] - t,
        1.0 - centers[:, 0] - t,
        centers[:, 1] - t,
        1.0 - centers[:, 1] - t
    ])
    # Pairwise squared distance constraints: dist^2 >= 4*t^2
    dx = centers[:, 0, None] - centers[:, 0][None, :]
    dy = centers[:, 1, None] - centers[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c = np.concatenate([c, dist_sq[mask] - 4.0 * t**2])
    return c

def objective(vars_flat):
    """Objective function to maximize t by minimizing -t."""
    return -vars_flat[-1]

def constraint_func(vars_flat):
    """Constraint function for the optimizer."""
    t = vars_flat[-1]
    centers = vars_flat[:2 * N].reshape(N, 2)
    return compute_constraints(centers, t)

def generate_hex_pattern(row_counts):
    """Generates a centered and scaled hexagonal lattice pattern from row counts."""
    rel_pts = []
    curr_y = 0
    for r_idx, cnt in enumerate(row_counts):
        shift = (r_idx % 2) * 1.0
        for k in range(cnt):
            x = k * 2.0 + shift
            rel_pts.append([x, curr_y])
        curr_y += np.sqrt(3)
    pts = np.array(rel_pts)
    
    # Center the pattern at (0, 0)
    pts -= pts.mean(axis=0)
    
    # Scale independently in x and y to fit in [0.1, 0.9]
    span_x = pts[:, 0].max() - pts[:, 0].min()
    span_y = pts[:, 1].max() - pts[:, 1].min()
    
    if span_x > 0:
        pts[:, 0] = pts[:, 0] / span_x * 0.8 + 0.5
    if span_y > 0:
        pts[:, 1] = pts[:, 1] / span_y * 0.8 + 0.5
        
    return pts

def run_packing():
    best_t = 0.0
    best_centers = None
    
    # Various row distributions that sum to 26, known to be near-optimal for square packing
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4], [5, 6, 4, 6, 5],
        [5, 4, 6, 5, 6], [6, 4, 5, 6, 5], [5, 5, 5, 5, 6],
        [6, 5, 5, 5, 5], [4, 5, 6, 5, 6], [5, 4, 5, 6, 6],
        [5, 6, 5, 5, 5], [6, 5, 5, 6, 4], [5, 5, 4, 6, 6]
    ]
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.08, 0.12)]
    cons_dict = {'type': 'ineq', 'fun': constraint_func}
    
    np.random.seed(42)
    
    # Optimization Phase 1: Find best centers for equal radius packing
    for pat in patterns:
        if sum(pat) != N:
            continue
            
        init_pts = generate_hex_pattern(pat)
        x0 = np.concatenate([init_pts.flatten(), [0.09]])
        
        # Multiple perturbations per pattern to escape local minima
        for _ in range(4):
            x0_pert = x0.copy()
            x0_pert[:2 * N] += np.random.uniform(-0.02, 0.02, 2 * N)
            x0_pert[:2 * N] = np.clip(x0_pert[:2 * N], 0.05, 0.95)
            
            try:
                res = minimize(
                    objective, x0_pert, method='SLSQP', bounds=bounds,
                    constraints=cons_dict, options={'maxiter': 3000, 'ftol': 1e-14}
                )
                if res.x[-1] > best_t:
                    c_vals = constraint_func(res.x)
                    if np.min(c_vals) >= -1e-6:
                        best_t = res.x[-1]
                        best_centers = res.x[:2 * N].reshape(N, 2)
            except Exception:
                pass
                
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = generate_hex_pattern([5, 6, 5, 6, 4])
        best_t = 0.09
        
    centers = best_centers
    
    # Optimization Phase 2: Linear Programming to maximize sum of variable radii
    # Given fixed centers, find r_i that maximize sum(r_i)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
            
    num_pairs = len(pairs)
    A_ub = np.zeros((num_pairs + 4 * N, N))
    b_ub = np.zeros(num_pairs + 4 * N)
    
    idx = 0
    for i, j in pairs:
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = dists[i, j]
        idx += 1
        
    for i in range(N):
        x, y = centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    radii = np.full(N, best_t)
    try:
        lp_res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success:
            # Shrink slightly to guarantee strict validity against 1e-12 tolerance
            radii = lp_res.x * 0.9999999
    except Exception:
        pass
        
    return centers, radii, float(np.sum(radii))
