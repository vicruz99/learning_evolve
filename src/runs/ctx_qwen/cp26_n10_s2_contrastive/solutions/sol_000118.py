# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000091 (state 4dfa0868) state=9beda399 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    b_ub = np.zeros(NUM_PAIRS)
    
    # Compute pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    b_ub[:] = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def objective(params):
    """Minimize negative sum of radii."""
    return -np.sum(params[2::3])

def constraints(params):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx = params[0::3]
    cy = params[1::3]
    r = params[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def make_hex_centers(spacing, margin, noise, rng):
    """Generate a hexagonal lattice initialization."""
    centers = np.zeros((N, 2))
    idx = 0
    y = margin
    row = 0
    while idx < N and y < 1.0 - margin:
        x = margin + (row % 2) * spacing / 2.0
        while x < 1.0 - margin and idx < N:
            centers[idx] = [x, y]
            x += spacing
            idx += 1
        y += spacing * np.sqrt(3) / 2.0
        row += 1
    while idx < N:
        centers[idx] = [0.5, 0.5]
        idx += 1
    if noise > 0:
        centers += rng.normal(0, noise, centers.shape)
    return np.clip(centers, 0.01, 0.99)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng_main = np.random.default_rng(42)
    
    # Phase 1: Diverse Initial Configurations
    inits = []
    # Hexagonal patterns with varying densities
    for sp in np.linspace(0.16, 0.23, 9):
        for noise in [0.0, 0.008, 0.02]:
            inits.append(make_hex_centers(sp, 0.04, noise, rng_main))
    # Random uniform placements
    for _ in range(20):
        inits.append(rng_main.uniform(0.08, 0.92, (N, 2)))
        
    # Phase 2: Multi-start SLSQP with LP initialization
    for base in inits:
        c_init = base.copy()
        r_init = solve_radii_lp(c_init) * 0.92  # Strictly feasible start
        r_init = np.maximum(r_init, 0.005)
        
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt,
                           options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            
            curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
            curr_r = solve_radii_lp(curr_c)
            curr_s = np.sum(curr_r)
            
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
        except Exception:
            pass

    # Phase 3: LP-Driven Basin Hopping
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(120):
            # Adaptive decaying noise
            scale = 0.012 * (0.92 ** (step / 20.0)) + 0.001
            c_pert = curr_c + rng_main.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert = solve_radii_lp(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                
                # Local SLSQP polish after successful jump
                x0_p = np.zeros(3*N)
                x0_p[0::3] = curr_c[:, 0]
                x0_p[1::3] = curr_c[:, 1]
                x0_p[2::3] = np.maximum(curr_r * 0.96, 0.005)
                try:
                    res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt,
                                     options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                    if res_p.success:
                        c_pol = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                        r_pol = solve_radii_lp(c_pol)
                        s_pol = np.sum(r_pol)
                        if s_pol > curr_s:
                            curr_c, curr_r, curr_s = c_pol, r_pol, s_pol
                except Exception:
                    pass
                    
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()

    # Fallback safety net
    if best_centers is None:
        best_centers = make_hex_centers(0.19, 0.05, 0.0, rng_main)
        best_radii = solve_radii_lp(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    final_sum = float(np.sum(r_final))
    return c_final, r_final, final_sum
