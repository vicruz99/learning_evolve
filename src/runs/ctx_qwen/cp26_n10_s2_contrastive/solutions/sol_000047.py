# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000024 (state 7d29769f) state=e4fe2d3c sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute pairwise indices and LP constraint structure for speed
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)
A_ub_pairs = np.zeros((NUM_PAIRS, N))
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_ub_pairs[idx, i] = 1.0
        A_ub_pairs[idx, j] = 1.0
        idx += 1

def get_optimal_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    # Compute pairwise distances efficiently
    dx = centers[:, 0][:, None] - centers[None, :, 0]
    dy = centers[:, 1][:, None] - centers[None, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    b_ub = dists[I_IDX, J_IDX]
    
    # Variable bounds: 0 <= r_i <= min(x, 1-x, y, 1-y)
    bounds = []
    for i in range(N):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, max_r)))
        
    # Solve LP: minimize -sum(r) subject to A_ub @ r <= b_ub
    for method in ['highs', 'interior-point', 'revised simplex']:
        try:
            res = linprog(-np.ones(N), A_ub=A_ub_pairs, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-7):
                return res.x
        except Exception:
            continue
    return np.full(N, 0.01)

def eval_sum_radii(centers):
    """Evaluate sum of optimal radii for given centers."""
    return np.sum(get_optimal_radii_lp(centers))

def objective_full(x):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_full(x):
    """Constraints for SLSQP: boundaries and non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + NUM_PAIRS)
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    # Pairwise non-overlap: dist(i,j) >= ri + rj
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - r[I_IDX] - r[J_IDX]
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_full}
    
    # Generate diverse initial configurations
    inits = []
    # Hexagonal patterns with varying spacing to explore different densities
    for sp in np.linspace(0.15, 0.22, 8):
        pts = []
        y = sp
        row = 0
        while len(pts) < N:
            x_start = sp if row % 2 == 0 else sp + sp / 2
            x = x_start
            while x < 1.0 - sp + 0.01 and len(pts) < N:
                pts.append([x, y])
                x += sp
            y += sp * np.sqrt(3) / 2
            row += 1
        inits.append(np.array(pts[:N]))
        
    # Random initializations to escape lattice biases
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Phase 1: SLSQP optimization on full variables (x, y, r)
    for base in inits:
        c_init = base.copy()
        c_init += rng.normal(0, 0.015, c_init.shape)
        c_init = np.clip(c_init, 0.05, 0.95)
        r_init = np.full(N, 0.06)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective_full, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 20000, 'ftol': 1e-12, 'disp': False})
            
            if res.success or -res.fun > best_sum:
                curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                # LP polishing guarantees exact optimal radii for these centers
                curr_r = get_optimal_radii_lp(curr_c)
                curr_s = np.sum(curr_r)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            continue
            
    # Phase 2: Refine centers using derivative-free optimization guided by LP
    # This decouples center arrangement from radius sizing, often finding better local optima
    if best_centers is not None:
        def obj_centers(x_flat):
            return -eval_sum_radii(x_flat.reshape(N, 2))
            
        # Run Powell method from the best SLSQP result
        x0_centers = best_centers.flatten()
        try:
            res_centers = minimize(obj_centers, x0_centers, method='Powell',
                                   options={'maxiter': 3000, 'xtol': 1e-6, 'ftol': 1e-6})
            curr_c = res_centers.x.reshape(N, 2)
            curr_r = get_optimal_radii_lp(curr_c)
            curr_s = np.sum(curr_r)
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
        except Exception:
            pass
            
        # Controlled stochastic local search to escape shallow minima
        step = 0.02
        for _ in range(1500):
            idx = rng.integers(N)
            pert = best_centers.copy()
            pert[idx] += rng.normal(0, step, 2)
            pert[idx] = np.clip(pert[idx], 0.01, 0.99)
            s_new = eval_sum_radii(pert)
            if s_new > best_sum:
                best_sum = s_new
                best_centers = pert
                best_radii = get_optimal_radii_lp(pert)
            step = max(0.0005, step * 0.998)

    # Final validation and numerical cleanup
    if best_centers is None:
        best_centers = np.random.rand(N, 2) * 0.6 + 0.2
        best_radii = np.full(N, 0.02)
        
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Strict boundary enforcement
    for i in range(N):
        x, y = c_final[i]
        r_final[i] = min(r_final[i], x, 1.0 - x, y, 1.0 - y, 0.5)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve microscopic overlaps from numerical drift
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
                if d < r_final[i] + r_final[j] - 1e-12:
                    exc = r_final[i] + r_final[j] - d
                    r_final[i] -= exc * 0.5
                    r_final[j] -= exc * 0.5
                    r_final[i] = max(0.0, r_final[i])
                    r_final[j] = max(0.0, r_final[j])
                    changed = True
        if not changed:
            break
            
    return c_final, r_final, float(np.sum(r_final))
