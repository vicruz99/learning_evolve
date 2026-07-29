# sol_000101 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000090 (state 81009fa6) state=228be3df sum of radii=2.424594 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIRS_I)

def get_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), PAIRS_I] = 1.0
    A_ub[np.arange(NUM_PAIRS), PAIRS_J] = 1.0
    
    dx = centers[PAIRS_I, 0] - centers[PAIRS_J, 0]
    dy = centers[PAIRS_I, 1] - centers[PAIRS_J, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0-x, y, 1.0-y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def objective_full(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_full(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[PAIRS_I] - cx[PAIRS_J]
    dy = cy[PAIRS_I] - cy[PAIRS_J]
    c[4*N:] = np.hypot(dx, dy) - (r[PAIRS_I] + r[PAIRS_J])
    return c

def make_hex_init(spacing, row_off, col_off, rng):
    """Generate a perturbed hexagonal lattice initialization."""
    centers = np.zeros((N, 2))
    idx = 0
    y = 0.05 + row_off
    row = 0
    while idx < N and y < 0.95:
        x_start = 0.05 + col_off + (row % 2) * spacing / 2.0
        col = 0
        while x_start + col * spacing < 0.95 and idx < N:
            centers[idx] = [x_start + col * spacing, y]
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2.0
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.008, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_full}
    
    rng = np.random.default_rng(42)
    candidates = []
    
    # Phase 1: Diverse structured & random initializations
    for sp in np.linspace(0.19, 0.26, 10):
        for roff in [0.0, 0.04, 0.08]:
            for coff in [0.0, 0.05]:
                c = make_hex_init(sp, roff, coff, rng)
                r = get_lp_radii(c)
                candidates.append((np.sum(r), c, r))
                
    for _ in range(30):
        c = rng.uniform(0.1, 0.9, (N, 2))
        r = get_lp_radii(c)
        candidates.append((np.sum(r), c, r))
        
    # Sort and keep top candidates for intensive search
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:5]
    
    best_sum = top_candidates[0][0]
    best_centers = top_candidates[0][1].copy()
    best_radii = top_candidates[0][2].copy()
    
    # Phase 2: LP-based Basin Hopping on centers
    for s, c_init, r_init in top_candidates:
        current_c = c_init.copy()
        current_r = r_init.copy()
        current_s = s
        local_best_c = current_c.copy()
        local_best_r = current_r.copy()
        local_best_s = current_s
        
        for step in range(500):
            # Adaptive noise schedule
            noise = 0.01 * np.exp(-step / 200.0)
            
            # Global perturbation
            pert = current_c + rng.normal(0, noise, current_c.shape)
            pert = np.clip(pert, 0.01, 0.99)
            r_pert = get_lp_radii(pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > current_s:
                current_c, current_r, current_s = pert, r_pert, s_pert
                if current_s > local_best_s:
                    local_best_c, local_best_r, local_best_s = current_c.copy(), current_r.copy(), current_s
                    
            # Local subset perturbation to break symmetry
            if rng.random() < 0.3:
                idxs = rng.choice(N, size=max(1, N//4), replace=False)
                sub_pert = current_c.copy()
                sub_pert[idxs] += rng.normal(0, noise * 1.5, (len(idxs), 2))
                sub_pert = np.clip(sub_pert, 0.01, 0.99)
                r_sub = get_lp_radii(sub_pert)
                if np.sum(r_sub) > current_s:
                    current_c, current_r, current_s = sub_pert, r_sub, np.sum(r_sub)
                    if current_s > local_best_s:
                        local_best_c, local_best_r, local_best_s = current_c.copy(), current_r.copy(), current_s
                        
        if local_best_s > best_sum:
            best_sum = local_best_s
            best_centers = local_best_c
            best_radii = local_best_r
            
    # Phase 3: SLSQP Joint Polishing
    for _ in range(3):
        x0 = np.zeros(3*N)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
        
        try:
            res = minimize(objective_full, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = get_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt
                    best_radii = r_opt
        except Exception:
            pass
            
        # Small perturbation restart for SLSQP to escape shallow basins
        c_start = best_centers + rng.normal(0, 0.002, best_centers.shape)
        c_start = np.clip(c_start, 0.02, 0.98)
        r_start = get_lp_radii(c_start)
        x0[:] = 0.0
        x0[0::3] = c_start[:, 0]
        x0[1::3] = c_start[:, 1]
        x0[2::3] = np.maximum(r_start * 0.98, 1e-5)

    # Phase 4: Strict Validation & Projection
    c_final = best_centers.copy()
    r_final = get_lp_radii(c_final)  # Re-optimize radii for final centers
    
    # Enforce boundary constraints strictly
    for i in range(N):
        x, y = c_final[i]
        ub = min(x, 1.0-x, y, 1.0-y)
        r_final[i] = min(r_final[i], ub - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = PAIRS_I[k], PAIRS_J[k]
            d = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed:
            break
            
    return c_final, r_final, float(np.sum(r_final))
