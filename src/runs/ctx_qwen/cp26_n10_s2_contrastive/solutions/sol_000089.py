# sol_000089 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000050 (state 5c8f47d4) state=238138e5 sum of radii=2.594097 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def get_lp_radii(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    b_ub = np.zeros(NUM_PAIRS)
    
    for k in range(NUM_PAIRS):
        i, j = I_IDX[k], J_IDX[k]
        d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = d
        
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def objective_sl(x):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_sl(x):
    """Inequality constraints for SLSQP: boundary and squared non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    r_sum = r[I_IDX] + r[J_IDX]
    c[4*N:] = dx**2 + dy**2 - r_sum**2
    return c

def generate_init_centers(method='hex', rng=None):
    """Generate structured or random initial center configurations."""
    if rng is None:
        rng = np.random.default_rng(0)
    centers = np.zeros((N, 2))
    
    if method == 'hex':
        idx = 0
        row = 0
        y = 0.06
        dy = 0.165
        while idx < N and y < 0.94:
            x = 0.06 + (row % 2) * 0.0825
            while x < 0.94 and idx < N:
                centers[idx] = [x, y]
                idx += 1
                x += 0.165
            y += dy
            row += 1
        if idx < N:
            centers[idx:] = rng.uniform(0.1, 0.9, (N - idx, 2))
    elif method == 'corners':
        corners = [[0.06, 0.06], [0.94, 0.06], [0.06, 0.94], [0.94, 0.94]]
        centers[:4] = corners
        idx = 4
        for val in [0.06, 0.94]:
            for pos in np.linspace(0.15, 0.85, 4):
                if idx < N:
                    centers[idx] = [val, pos]
                    idx += 1
            for pos in np.linspace(0.15, 0.85, 4):
                if idx < N:
                    centers[idx] = [pos, val]
                    idx += 1
        while idx < N:
            centers[idx] = rng.uniform(0.2, 0.8, 2)
            idx += 1
    else:
        centers = rng.uniform(0.1, 0.9, (N, 2))
        
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_sl}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse starts with SLSQP
    starts = []
    for m in ['hex', 'corners', 'random']:
        for _ in range(3):
            starts.append(generate_init_centers(m, rng))
            
    for c0 in starts:
        r0 = get_lp_radii(c0)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0 * 0.95  # Slightly shrink for strict feasibility
        
        try:
            res = minimize(objective_sl, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
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

    # Phase 2: Iterative perturbation + multi-candidate SLSQP refinement
    if best_centers is not None:
        c_curr = best_centers.copy()
        r_curr = best_radii.copy()
        s_curr = best_sum
        
        for step in range(60):
            candidates = []
            for _ in range(10):
                sigma = 0.006 * (0.97 ** (step // 10))
                if rng.random() < 0.12:  # Occasional big jump to escape basins
                    sigma = 0.025
                c_pert = c_curr + rng.normal(0, sigma, c_curr.shape)
                c_pert = np.clip(c_pert, 0.02, 0.98)
                
                r_pert = get_lp_radii(c_pert)
                s_pert = np.sum(r_pert)
                candidates.append((c_pert, r_pert, s_pert))
                
            # Sort candidates by score descending
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Refine top candidates with SLSQP
            for cand in candidates[:3]:
                c_cand, r_cand, s_cand = cand
                if s_cand <= best_sum - 1e-6:
                    continue
                    
                x0_p = np.zeros(3 * N)
                x0_p[0::3] = c_cand[:, 0]
                x0_p[1::3] = c_cand[:, 1]
                x0_p[2::3] = r_cand * 0.99
                
                try:
                    res_p = minimize(objective_sl, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt, options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
                    if res_p.success:
                        c_ref = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                        r_ref = get_lp_radii(c_ref)
                        s_ref = np.sum(r_ref)
                        if s_ref > best_sum:
                            best_sum = s_ref
                            best_centers = c_ref.copy()
                            best_radii = r_ref.copy()
                            c_curr = c_ref.copy()
                            r_curr = r_ref.copy()
                            s_curr = s_ref
                except Exception:
                    pass
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = generate_init_centers('hex')
        best_radii = get_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Strict post-processing to guarantee validity
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    for _ in range(50):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
            if d < r_final[i] + r_final[j] - 1e-10:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed:
            break
            
    return c_final, r_final, float(np.sum(r_final))
