# sol_000090 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000050 (state 5c8f47d4) state=81009fa6 sum of radii=2.631730 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIRS_I)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
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
    # Use hypot for stable, non-vanishing gradients at contact
    c[4*N:] = np.hypot(dx, dy) - (r[PAIRS_I] + r[PAIRS_J])
    return c

def get_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    b_ub = np.zeros(NUM_PAIRS)
    
    for k in range(NUM_PAIRS):
        i, j = PAIRS_I[k], PAIRS_J[k]
        d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = d
        
    bounds = []
    for i in range(n):
        mx, my = centers[i]
        ub = min(mx, 1.0-mx, my, 1.0-my)
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def generate_hex_init(spacing, row_offset, col_offset, seed):
    """Generate a hexagonal lattice initialization with controlled noise."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 2))
    idx = 0
    y = 0.05 + row_offset
    row = 0
    while idx < N and y < 0.95:
        x_start = 0.05 + col_offset + (row % 2) * spacing / 2.0
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
    return centers + rng.normal(0, 0.005, centers.shape)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Diverse initial configurations
    inits = []
    # Hexagonal patterns with varying density and alignment
    for sp in np.linspace(0.19, 0.24, 6):
        for roff in [0.0, 0.05]:
            for coff in [0.0, 0.05]:
                inits.append(generate_hex_init(sp, roff, coff, seed=42))
    # Random uniform placements
    for s in range(15):
        rng = np.random.default_rng(s*11+3)
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # Phase 2: SLSQP + LP refinement from multiple starts
    for base in inits:
        c_init = np.clip(base, 0.02, 0.98)
        r_init = get_lp_radii(c_init) * 0.95
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = np.maximum(r_init, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = get_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass

    # Phase 3: Basin hopping / Jiggle refinement to escape local minima
    if best_centers is not None:
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_s = best_sum
        
        for step in range(500):
            rng = np.random.default_rng(step * 17 + 5)
            # Adaptive noise schedule
            noise = 0.008 * max(0.0, 1.0 - step / 450.0)
            c_pert = current_c + rng.normal(0, noise, current_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert = get_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > current_s:
                current_c, current_r, current_s = c_pert, r_pert, s_pert
                
                # Local SLSQP polishing after successful jump
                x0_p = np.zeros(3*N)
                x0_p[0::3] = current_c[:, 0]
                x0_p[1::3] = current_c[:, 1]
                x0_p[2::3] = np.maximum(current_r * 0.99, 1e-5)
                try:
                    res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    if res_p.success:
                        c_opt = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                        r_opt = get_lp_radii(c_opt)
                        s_opt = np.sum(r_opt)
                        if s_opt > current_s:
                            current_c, current_r, current_s = c_opt, r_opt, s_opt
                            best_sum = current_s
                            best_centers = current_c.copy()
                            best_radii = current_r.copy()
                except Exception:
                    pass
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii = get_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx, my = c_final[i]
        ub = min(mx, 1.0-mx, my, 1.0-my)
        r_final[i] = min(r_final[i], ub - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(50):
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
