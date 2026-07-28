# sol_000097 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000075 (state 5b5bfa68) state=b64973fa sum of radii=1.591034 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRI_IDX = np.triu_indices(N_CIRCLES, k=1)

def objective_centers(vars):
    """Objective: maximize the minimum distance D by minimizing -D"""
    return -vars[-1]

def constraints_centers(vars):
    """
    Inequality constraints >= 0:
    1. Boundary clearances: x >= D, 1-x >= D, y >= D, 1-y >= D
    2. Pairwise separations: ||c_i - c_j||^2 >= D^2
    """
    n = N_CIRCLES
    cx = vars[:n]
    cy = vars[n:2*n]
    D = vars[2*n]
    
    cons = np.empty(4*n + n*(n-1)//2)
    cons[:n] = cx - D
    cons[n:2*n] = 1.0 - cx - D
    cons[2*n:3*n] = cy - D
    cons[3*n:4*n] = 1.0 - cy - D
    
    cx_diff = cx[:, None] - cx[None, :]
    cy_diff = cy[:, None] - cy[None, :]
    d2 = cx_diff**2 + cy_diff**2
    cons[4*n:] = d2[TRI_IDX] - D**2
    return cons

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = N_CIRCLES
    # Maximum radius allowed by boundaries
    limits = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    limits = np.maximum(limits, 1e-9)
    
    c = -np.ones(n)  # Maximize sum(r) -> minimize -sum(r)
    bounds = [(0.0, lim) for lim in limits]
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = np.linalg.norm(centers[i] - centers[j])
            idx += 1
            
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 0.08), 0.0

def get_hex_init(row_counts, r0=0.1):
    """Generates an initial hexagonal grid based on row distribution."""
    n = N_CIRCLES
    pts = []
    y = r0
    row = 0
    dy = np.sqrt(3) * r0
    for cnt in row_counts:
        if len(pts) >= n: break
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n: break
            if x + r0 <= 1.0:
                pts.append([x, y])
            x += 2 * r0
        y += dy
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Various row distributions that sum to 26, mimicking dense hexagonal packings
    row_configs = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [6, 6, 4, 5, 5], [5, 4, 6, 6, 5]
    ]
    
    starts = []
    rng = np.random.RandomState(42)
    for rc in row_configs:
        pts = get_hex_init(rc, 0.1)
        starts.append(pts)
        # Generate perturbed variants to escape local minima
        for _ in range(3):
            p = pts + rng.uniform(-0.025, 0.025, pts.shape)
            starts.append(np.clip(p, 0.05, 0.95))
            
    # Add fully random valid starts
    for _ in range(5):
        starts.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    bounds_c = [(0.0, 1.0)] * (2 * n) + [(0.05, 0.15)]
    cons_dict = {'type': 'ineq', 'fun': constraints_centers}
    
    for cfg in starts:
        x0 = np.concatenate([cfg.flatten(), [0.09]])
        try:
            res = minimize(
                objective_centers, x0, method='SLSQP', bounds=bounds_c,
                constraints=cons_dict, options={'maxiter': 10000, 'ftol': 1e-13}
            )
            if np.isfinite(res.fun):
                opt_centers = res.x[:2*n].reshape(n, 2)
                radii, s = solve_radii_lp(opt_centers)
                if s > best_sum:
                    best_sum = s
                    best_centers = opt_centers.copy()
                    best_radii = radii.copy()
        except Exception:
            continue

    # Fallback if optimization unexpectedly yields no valid result
    if best_centers is None:
        best_centers = starts[0]
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # Final safety scaling to guarantee strict validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
