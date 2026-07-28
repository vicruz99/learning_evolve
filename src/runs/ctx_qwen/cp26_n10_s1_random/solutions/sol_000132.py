# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000081 (state 6da8454c) state=fb176a7f sum of radii=1.280434 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def obj_equal(vars_arr):
    """Objective: minimize negative radius r (maximize r)."""
    return -vars_arr[-1]

def constr_equal(vars_arr):
    """Constraints for equal-radius packing: boundary and non-overlap."""
    n = N
    r = vars_arr[-1]
    x = vars_arr[:n]
    y = vars_arr[n:2*n]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_b = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap: dist^2 >= (2r)^2
    X = x[:, None] - x[None, :]
    Y = y[:, None] - y[None, :]
    idx = np.triu_indices(n, k=1)
    c_p = (X[idx]**2 + Y[idx]**2) - 4.0 * r**2
    
    return np.concatenate([c_b, c_p])

def generate_hex_init(row_counts, r0=0.08):
    """Generate initial positions on a hexagonal lattice with specified row counts."""
    pts = []
    y = r0
    for idx, cnt in enumerate(row_counts):
        shift = r0 if idx % 2 == 1 else 0.0
        row_width = (cnt - 1) * 2 * r0
        x_start = 0.5 - row_width / 2.0 + shift
        for c in range(cnt):
            x = x_start + c * 2 * r0
            pts.append([x, y])
        y += np.sqrt(3) * r0
    return np.array(pts[:N])

def solve_radii_lp(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    
    c = -np.ones(n)  # Maximize sum -> minimize -sum
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 0.05)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_r_eq = 0.0
    best_centers_eq = None
    
    # Diverse row distributions summing to 26
    row_configs = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [5, 5, 5, 6, 5], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4],
        [4, 6, 6, 6, 4], [5, 6, 6, 5, 4], [5, 5, 5, 5, 5, 1]
    ]
    
    inits = []
    for rc in row_configs:
        pts = generate_hex_init(rc, r0=0.08)
        inits.append(pts)
        
    # Controlled perturbations to escape local minima
    for _ in range(8):
        base = inits[0].copy()
        base += rng.uniform(-0.03, 0.03, base.shape)
        base = np.clip(base, 0.05, 0.95)
        inits.append(base)
        
    # Random starts
    for _ in range(3):
        pts = rng.uniform(0.1, 0.9, (N, 2))
        inits.append(pts)
        
    cons_eq = {'type': 'ineq', 'fun': constr_equal}
    bounds_eq = [(0.0, 1.0)] * (2 * N) + [(0.01, 0.5)]
    
    for cfg in inits:
        # Start with slightly smaller r to ensure strict feasibility
        v0 = np.concatenate([cfg.flatten(), [0.07]])
        
        try:
            res = minimize(obj_equal, v0, method='SLSQP', constraints=cons_eq,
                          bounds=bounds_eq,
                          options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            
            if np.isfinite(res.fun):
                r_val = res.x[-1]
                c_val = constr_equal(res.x)
                # Accept if constraints are satisfied within tolerance
                if np.min(c_val) > -1e-6 and r_val > best_r_eq:
                    best_r_eq = r_val
                    best_centers_eq = res.x[:2 * N].reshape(N, 2).copy()
        except Exception:
            continue
            
    if best_centers_eq is None:
        best_centers_eq = inits[0]
        
    # LP refinement to allow unequal radii and maximize sum
    radii = solve_radii_lp(best_centers_eq)
    
    # Safety scaling to guarantee strict validity within 1e-12 tolerance
    scale = 1.0
    for i in range(N):
        x, y = best_centers_eq[i]
        r = radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_centers_eq[i] - best_centers_eq[j])
            rs = radii[i] + radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    radii *= scale * 0.999999
    best_sum = float(np.sum(radii))
    
    return best_centers_eq, radii, best_sum
