# sol_000274 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000262 (state 4217c70f) state=2a84d47a sum of radii=2.624605 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRIU_I, TRIU_J = np.triu_indices(N_CIRCLES, k=1)
N_PAIRS = len(TRIU_I)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = N_CIRCLES
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds.append((0.0, max(lim, 1e-9)))
        
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), TRIU_I] = 1.0
    A_ub[np.arange(N_PAIRS), TRIU_J] = 1.0
    
    diff = centers[TRIU_I] - centers[TRIU_J]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    b_ub = dists
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success and np.all(res.x >= -1e-7):
        return np.maximum(res.x, 0.0), -res.fun
    return np.full(n, 1e-9), 0.0

def obj_slsqp(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N_CIRCLES:])

def cons_slsqp(v):
    """Inequality constraints >= 0 for SLSQP."""
    n = N_CIRCLES
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    
    # Boundary constraints
    c1 = x - r
    c2 = 1.0 - x - r
    c3 = y - r
    c4 = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[TRIU_I] - x[TRIU_J]
    dy = y[TRIU_I] - y[TRIU_J]
    dr = r[TRIU_I] + r[TRIU_J]
    c5 = dx**2 + dy**2 - dr**2
    
    return np.concatenate([c1, c2, c3, c4, c5])

def generate_hex(row_counts, r0):
    """Generates a hexagonal lattice configuration with specified row counts."""
    n = N_CIRCLES
    pts = []
    y = r0
    for i, cnt in enumerate(row_counts):
        shift = r0 if i % 2 == 1 else 0.0
        width = (cnt - 1) * 2 * r0
        x_start = 0.5 - width / 2.0 + shift
        for k in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x_start + k * 2 * r0, y])
        y += np.sqrt(3) * r0
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Diverse hexagonal patterns that sum to 26
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,6,5,5,4], [6,5,4,6,5], [5,4,6,6,5], [7,5,5,5,4],
        [5,6,6,5,4], [5,5,5,5,6]
    ]
    
    starts = []
    for pat in patterns:
        if sum(pat) != n:
            continue
        pts = generate_hex(pat, 0.10)
        starts.append(pts)
        # Controlled perturbations to break symmetry
        for _ in range(4):
            p = pts + rng.uniform(-0.02, 0.02, pts.shape)
            p = np.clip(p, 0.05, 0.95)
            starts.append(p)
            
    # Add some random dense starts
    for _ in range(8):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    bounds_vars = [(0.0, 1.0)]*(2*n) + [(1e-6, 0.5)]*n
    constraint_dict = {'type': 'ineq', 'fun': cons_slsqp}
    
    # Phase 1: Joint SLSQP from diverse starts
    for cfg in starts:
        # Start with small feasible radii
        r_init = np.full(n, 0.03)
        
        x0 = np.zeros(3*n)
        x0[:n] = cfg[:,0]
        x0[n:2*n] = cfg[:,1]
        x0[2*n:] = r_init
        
        try:
            res = minimize(obj_slsqp, x0, method='SLSQP', bounds=bounds_vars, 
                           constraints=constraint_dict, 
                           options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                c_mat = np.column_stack((cx, cy))
                
                # LP refinement for exact max radii given optimized centers
                r_lp, s_lp = solve_lp(c_mat)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_mat.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # Phase 2: Iterative refinement around best configuration
    if best_centers is not None:
        for _ in range(12):
            pert = best_centers + rng.uniform(-0.015, 0.015, best_centers.shape)
            pert = np.clip(pert, 0.05, 0.95)
            r_init = np.full(n, 0.03)
            x0 = np.zeros(3*n)
            x0[:n] = pert[:,0]
            x0[n:2*n] = pert[:,1]
            x0[2*n:] = r_init
            
            try:
                res = minimize(obj_slsqp, x0, method='SLSQP', bounds=bounds_vars, 
                               constraints=constraint_dict, 
                               options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
                if np.isfinite(res.fun):
                    cx = res.x[:n]
                    cy = res.x[n:2*n]
                    c_mat = np.column_stack((cx, cy))
                    r_lp, s_lp = solve_lp(c_mat)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_mat.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass

    # Fallback
    if best_centers is None:
        best_centers = generate_hex([6,5,6,5,4], 0.09)
        best_radii, best_sum = solve_lp(best_centers)

    # Phase 3: Strict Safety Scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i,0] - best_centers[j,0], 
                         best_centers[i,1] - best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
