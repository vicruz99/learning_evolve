# sol_000116 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000091 (state 4dfa0868) state=8fd1a80d sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute constant structure for LP pairwise constraints: r_i + r_j <= dist_ij
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    b_ub = np.zeros(NUM_PAIRS)
    
    # Compute pairwise distances for RHS
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    b_ub[:] = dists[I_IDX, J_IDX]
    
    # Boundary constraints: 0 <= r_i <= min(x, 1-x, y, 1-y)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    # Solve LP with HiGHS (fast), fallback to interior-point
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return np.maximum(res.x, 0.0)
        except Exception:
            continue
    return np.full(n, 0.01)

def objective(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[2::3])

def constraints(params):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = params[0::3]
    cy = params[1::3]
    r = params[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    # Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    # Pairwise: distance >= r_i + r_j
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    r_sum = r[I_IDX] + r[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - r_sum
    return c

def generate_hex_init(spacing, row_shift, v_scale, margin, noise_scale, rng):
    """Generate a hexagonal lattice initialization with controlled parameters."""
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin + row_shift
    while idx < N and y < 1.0 - margin:
        x_start = margin + (row % 2) * spacing / 2.0
        col = 0
        while x_start + col * spacing < 1.0 - margin and idx < N:
            centers[idx, 0] = x_start + col * spacing
            centers[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2.0 * v_scale
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin, 1.0 - margin, 2)
        idx += 1
        
    if noise_scale > 0:
        centers += rng.normal(0, noise_scale, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    rng_main = np.random.default_rng(42)
    
    # Phase 1: Diverse Structured Initializations
    inits = []
    # Carefully spaced hexagonal grids with vertical compression/expansion
    for s in np.linspace(0.155, 0.215, 7):
        for rs in [0.0, 0.03, 0.06]:
            for vs in [0.95, 1.0, 1.05]:
                inits.append(generate_hex_init(s, rs, vs, margin=0.04, noise_scale=0.0, rng=rng_main))
                
    # Add symmetry-broken perturbations of the grids
    for base in inits[:14]:
        for ns in [0.005, 0.015]:
            pert = base.copy() + rng_main.normal(0, ns, base.shape)
            inits.append(np.clip(pert, 0.02, 0.98))
            
    # Pure random starts for exploration
    for _ in range(15):
        inits.append(rng_main.uniform(0.08, 0.92, (N, 2)))
        
    # Main Optimization Loop
    for base in inits:
        c_init = base.copy()
        # Solve LP for optimal radii at these centers, then shrink slightly for strict SLSQP feasibility
        r_init = solve_lp_radii(c_init) * 0.97
        
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            # First pass: moderate precision
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                # Second pass: high precision refinement
                res = minimize(objective, res.x, method='SLSQP', bounds=bounds_opt,
                               constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                
                if res.success:
                    cx_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_opt = solve_lp_radii(cx_opt)
                    s_opt = np.sum(r_opt)
                    
                    if s_opt > best_sum:
                        best_sum = s_opt
                        best_centers = cx_opt.copy()
                        best_radii = r_opt.copy()
        except Exception:
            pass

    # Phase 2: Adaptive Basin Hopping & Local Refinement
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(60):
            # Decaying noise schedule with random variation
            scale = 0.006 * (1.0 + 0.4 * np.random.rand()) * max(0.05, 1.0 - step / 50.0)
            c_pert = curr_c + np.random.randn(N, 2) * scale
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                
                # Polish successful jumps with SLSQP
                x0_p = np.zeros(3*N)
                x0_p[0::3] = curr_c[:, 0]
                x0_p[1::3] = curr_c[:, 1]
                x0_p[2::3] = curr_r * 0.98
                
                try:
                    res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                    if res_p.success:
                        c_ref = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                        r_ref = solve_lp_radii(c_ref)
                        s_ref = np.sum(r_ref)
                        if s_ref > curr_s:
                            curr_c, curr_r, curr_s = c_ref, r_ref, s_ref
                except Exception:
                    pass
                    
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = rng_main.uniform(0.15, 0.85, (N, 2))
        best_radii = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Strict Post-processing to Guarantee Validator Compliance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps deterministically
    for _ in range(80):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
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
            
    final_sum = float(np.sum(r_final))
    return c_final, r_final, final_sum
