# sol_000066 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000025 (state 808bce88) state=30ea2911 sum of radii=1.514490 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
PAIRS = np.triu_indices(N, k=1)

def objective(params):
    """Minimize negative sum of radii."""
    return -np.sum(params[2 * N:])

def constraints(params):
    """Vectorized inequality constraints: boundary and non-overlap."""
    cx = params[:N]
    cy = params[N:2*N]
    r = params[2*N:]
    
    # Preallocate constraint array
    c = np.empty(4 * N + N * (N - 1) // 2)
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    # Pairwise distance constraints: dist(i,j) - r_i - r_j >= 0
    cx_mat = cx[:, None]
    cy_mat = cy[:, None]
    dx = cx_mat - cx_mat.T
    dy = cy_mat - cy_mat.T
    dists = np.hypot(dx, dy)
    r_sum = r[:, None] + r[None, :]
    
    c[4*N:] = dists[PAIRS] - r_sum[PAIRS]
    return c

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    I, J = np.triu_indices(n, k=1)
    num_pairs = len(I)
    A_ub = np.zeros((num_pairs, n))
    A_ub[np.arange(num_pairs), I] = 1.0
    A_ub[np.arange(num_pairs), J] = 1.0
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dists = np.hypot(dx, dy)
    b_ub = dists[I, J]
    
    # Bounds: 0 <= r_i <= distance to nearest boundary
    bounds = []
    for i in range(n):
        mx, my = centers[i]
        ub = min(mx, 1.0 - mx, my, 1.0 - my)
        bounds.append((0.0, max(0.0, ub - 1e-9)))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success:
                return res.x, -res.fun
        except Exception:
            continue
    return np.full(n, 0.01), 0.0

def make_init_hex(seed, spacing=0.18):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    c = np.zeros((N, 2))
    idx = 0
    row = 0
    y = 0.05
    while idx < N and y < 0.95:
        x_start = 0.05 + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 0.95 and idx < N:
            c[idx] = [x_start + col * spacing, y]
            idx += 1
            col += 1
        y += spacing * math.sqrt(3) / 2
        row += 1
    c += rng.randn(N, 2) * 0.005
    return np.clip(c, 0.02, 0.98)

def make_init_random(seed):
    """Generate a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.12, 0.88, (N, 2))

def fix_violations(centers, radii):
    """Post-process to strictly satisfy boundary and overlap constraints."""
    n = centers.shape[0]
    # Enforce boundaries
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-9)
            
    # Iteratively resolve overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                s = radii[i] + radii[j]
                if d < s:
                    diff = s - d + 1e-9
                    radii[i] -= diff * 0.5
                    radii[j] -= diff * 0.5
                    changed = True
        if not changed:
            break
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    best_sum = 0.0
    best_c = None
    best_r = None
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    # Generate diverse initial configurations
    inits = []
    for s in range(15):
        inits.append(make_init_hex(s, spacing=0.17 + s * 0.002))
    for s in range(15):
        inits.append(make_init_random(s))
        
    # Phase 1: Broad search with SLSQP + LP polishing
    for c0 in inits:
        r0 = np.full(N, 0.04)
        x0 = np.concatenate([c0.ravel(), r0])
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_opt = res.x[:2*N].reshape(N, 2)
                # LP polishing guarantees optimal radii for these centers
                r_lp, s_lp = solve_lp_radii(c_opt)
                c_fix, r_fix = fix_violations(c_opt, r_lp)
                s_val = np.sum(r_fix)
                
                if s_val > best_sum:
                    best_sum = s_val
                    best_c = c_fix.copy()
                    best_r = r_fix.copy()
        except Exception:
            continue
            
    # Phase 2: Basin hopping refinement around the best solution
    if best_c is not None:
        for _ in range(35):
            rng = np.random.RandomState()
            # Perturb centers slightly
            c_p = best_c + rng.randn(N, 2) * 0.004
            c_p = np.clip(c_p, 0.01, 0.99)
            
            # Solve LP for new centers to get a feasible start for radii
            r_lp, _ = solve_lp_radii(c_p)
            r_p = r_lp * 0.99  # Scale down slightly for strict feasibility in SLSQP
            x_p = np.concatenate([c_p.ravel(), r_p])
            
            try:
                res_p = minimize(objective, x_p, method='SLSQP', bounds=bounds, constraints=cons_dict,
                                 options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
                if res_p.success:
                    c_op = res_p.x[:2*N].reshape(N, 2)
                    r_lp2, s_lp2 = solve_lp_radii(c_op)
                    c_fix2, r_fix2 = fix_violations(c_op, r_lp2)
                    s_val2 = np.sum(r_fix2)
                    if s_val2 > best_sum:
                        best_sum = s_val2
                        best_c = c_fix2.copy()
                        best_r = r_fix2.copy()
            except Exception:
                continue

    # Fallback safety net
    if best_c is None:
        best_c = make_init_hex(0)
        best_r = np.full(N, 0.04)
        best_c, best_r = fix_violations(best_c, best_r)
        best_sum = np.sum(best_r)
        
    return best_c, best_r, float(best_sum)
