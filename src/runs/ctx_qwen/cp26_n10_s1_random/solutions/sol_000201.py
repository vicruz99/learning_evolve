# sol_000201 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000197 (state 20ef424a) state=b98daa6f sum of radii=2.482174 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
M_CONSTRAINTS = N_CIRCLES * (N_CIRCLES - 1) // 2

# Precompute the constant structure of the LP constraint matrix
# Each row corresponds to a pair (i, j) with 1.0 at columns i and j
A_LP = np.zeros((M_CONSTRAINTS, N_CIRCLES))
_lp_idx = 0
for _i in range(N_CIRCLES):
    for _j in range(_i + 1, N_CIRCLES):
        A_LP[_lp_idx, _i] = 1.0
        A_LP[_lp_idx, _j] = 1.0
        _lp_idx += 1

def solve_lp_for_centers(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = N_CIRCLES
    c_obj = -np.ones(n)
    bounds = []
    b_ub = np.empty(M_CONSTRAINTS)
    
    # Boundary constraints -> upper bounds for each radius
    for i in range(n):
        x, y = centers[i]
        w = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(w, 1e-9)))
        
    # Pairwise distance constraints
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            b_ub[idx] = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def negative_min_clearance(x_flat):
    """Objective for Nelder-Mead: maximize minimum clearance."""
    c = x_flat.reshape(N_CIRCLES, 2)
    
    # Distance to walls
    d_bound = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                         np.minimum(c[:, 1], 1.0 - c[:, 1]))
    min_b = np.min(d_bound)
    
    # Distance between circles
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return -min(min_b, min_pair)

def generate_hex_init(r0):
    """Generates a hexagonal lattice configuration."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N_CIRCLES:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0 and len(pts) < N_CIRCLES:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    while len(pts) < N_CIRCLES:
        pts.append([0.5, 0.5])
    return np.array(pts[:N_CIRCLES])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse high-quality initial configurations
    inits = []
    for r0 in [0.09, 0.095, 0.10, 0.105, 0.11]:
        for shift_y in [-0.02, 0.0, 0.02]:
            pts = generate_hex_init(r0)
            pts[:, 1] += shift_y
            pts = np.clip(pts, 0.05, 0.95)
            inits.append(pts)
            
    for _ in range(8):
        inits.append(rng.uniform(0.15, 0.85, (N_CIRCLES, 2)))
        
    # 2. Phase 1: Maximize clearance to find structurally sound centers
    for cfg in inits:
        # Nelder-Mead handles the non-smooth min function robustly
        res = minimize(negative_min_clearance, cfg.flatten(), method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-10})
        c_opt = np.clip(res.x.reshape(N_CIRCLES, 2), 0.01, 0.99)
        
        # Exact LP refinement for radii
        r_lp, s_lp = solve_lp_for_centers(c_opt)
        if r_lp is not None and s_lp > best_sum:
            best_sum = s_lp
            best_centers = c_opt.copy()
            best_radii = r_lp.copy()
            
    # Fallback if optimization somehow yields no valid result
    if best_centers is None:
        best_centers = generate_hex_init(0.095)
        best_radii = np.full(N_CIRCLES, 0.09)
        best_sum = np.sum(best_radii)
        
    # 3. Phase 2: Iterative local search (Hill Climbing)
    current_centers = best_centers.copy()
    current_radii = best_radii.copy()
    current_sum = best_sum
    step = 0.025
    
    for it in range(1000):
        i = rng.integers(N_CIRCLES)
        old_c = current_centers[i].copy()
        
        # Perturb one circle
        move = rng.uniform(-step, step, 2)
        current_centers[i] += move
        current_centers[i] = np.clip(current_centers[i], 0.01, 0.99)
        
        r_try, s_try = solve_lp_for_centers(current_centers)
        
        if r_try is not None and s_try > current_sum:
            current_sum = s_try
            current_radii = r_try.copy()
            best_sum = current_sum
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()
        else:
            current_centers[i] = old_c
            
        step *= 0.996  # Decay step size for fine-tuning
        
    # 4. Phase 3: Multi-circle perturbation to break symmetry traps
    for _ in range(60):
        c_pert = best_centers.copy()
        num_perturb = rng.integers(3, 6)
        idx = rng.choice(N_CIRCLES, size=num_perturb, replace=False)
        c_pert[idx] += rng.uniform(-0.015, 0.015, (num_perturb, 2))
        c_pert = np.clip(c_pert, 0.05, 0.95)
        
        # Quick clearance optimization from perturbed state
        res_nm = minimize(negative_min_clearance, c_pert.flatten(), method='Nelder-Mead',
                         options={'maxiter': 600})
        c_ref = np.clip(res_nm.x.reshape(N_CIRCLES, 2), 0.01, 0.99)
        
        r_ref, s_ref = solve_lp_for_centers(c_ref)
        if r_ref is not None and s_ref > best_sum:
            best_sum = s_ref
            best_centers = c_ref.copy()
            best_radii = r_ref.copy()

    # 5. Final safety scaling to guarantee strict numerical validity
    scale = 1.0
    for i in range(N_CIRCLES):
        x, y = best_centers[i]
        r = best_radii[i]
        if r < 1e-12: 
            continue
        scale = min(scale, x/r, (1.0 - x)/r, y/r, (1.0 - y)/r)
        
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs < 1e-12: 
                continue
            scale = min(scale, d / rs)
            
    best_radii *= scale * 0.9999995
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(final_sum)
