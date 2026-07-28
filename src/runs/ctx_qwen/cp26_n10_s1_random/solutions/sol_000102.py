# sol_000102 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000075 (state 5b5bfa68) state=603fd6ee sum of radii=1.195913 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def get_hex_starts(n):
    """Generates multiple hexagonal lattice configurations as initial guesses."""
    starts = []
    row_patterns = [
        [6, 6, 5, 5, 4], [5, 6, 6, 5, 4], [6, 5, 6, 5, 4],
        [5, 5, 5, 5, 6], [6, 6, 6, 4, 4], [5, 6, 5, 6, 4]
    ]
    
    for pat in row_patterns:
        # Base configuration
        pts = []
        r0 = 0.10
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            total_w = (cnt - 1) * 2 * r0
            x_start = (1.0 - total_w) / 2.0 + shift
            for k in range(cnt):
                x = x_start + k * 2 * r0
                if len(pts) < n:
                    pts.append([x, y])
            y += r0 * np.sqrt(3)
        base_pts = np.array(pts[:n])
        starts.append(base_pts)
        
        # Perturbed variants
        rng = np.random.RandomState(len(pat) * 1000)
        for _ in range(3):
            pert = base_pts + rng.uniform(-0.015, 0.015, base_pts.shape)
            starts.append(np.clip(pert, 0.05, 0.95))
            
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        starts.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    return starts

def objective_equal_radius(vars_array):
    """Objective: minimize negative radius."""
    return -vars_array[-1]

def constraints_equal_radius(vars_array, n):
    """Inequality constraints >= 0 for equal radius packing."""
    cx = vars_array[:n]
    cy = vars_array[n:2*n]
    r = vars_array[-1]
    
    # Boundary constraints
    b_cons = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Pairwise non-overlap constraints
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    p_cons = dist[triu_idx] - 2.0 * r
    
    return np.concatenate([b_cons, p_cons])

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    limits = np.maximum(limits, 0.0)
    
    c_obj = np.ones(n) * -1.0
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
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_eq = [(0.0, 1.0)] * (2 * n) + [(0.08, 0.15)]
    cons_eq = {'type': 'ineq', 'fun': lambda v: constraints_equal_radius(v, n)}
    
    starts = get_hex_starts(n)
    
    # Phase 1: Optimize centers assuming equal radii
    for cfg in starts:
        v0 = np.concatenate([cfg.flatten(), [0.095]])
        try:
            res = minimize(objective_equal_radius, v0, method='SLSQP', 
                          bounds=bounds_eq, constraints=cons_eq,
                          options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            if res.success:
                c_vals = constraints_equal_radius(res.x, n)
                if np.min(c_vals) >= -1e-9:
                    r_val = res.x[-1]
                    current_sum = r_val * n
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = res.x[:2*n].reshape(n, 2).copy()
                        best_radii = np.full(n, r_val)
        except Exception:
            continue
            
    # Phase 2: Use LP to optimally assign variable radii to best centers
    if best_centers is not None:
        radii_lp, sum_lp = solve_radii_lp(best_centers)
        if radii_lp is not None:
            best_radii = radii_lp
            best_sum = sum_lp
            
        # Local search: perturb centers slightly and re-solve LP
        rng = np.random.RandomState(123)
        for _ in range(30):
            pert_centers = best_centers + rng.uniform(-0.005, 0.005, best_centers.shape)
            pert_centers = np.clip(pert_centers, 0.01, 0.99)
            
            radii_p, sum_p = solve_radii_lp(pert_centers)
            if radii_p is not None and sum_p > best_sum:
                best_sum = sum_p
                best_centers = pert_centers
                best_radii = radii_p
                
    # Fallback if optimization completely fails
    if best_centers is None:
        r_fb = 0.095
        pts = []
        y = r_fb
        row = 0
        while len(pts) < n:
            shift = r_fb if row % 2 else 0.0
            x = r_fb + shift
            while x + r_fb <= 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2 * r_fb
            y += np.sqrt(3) * r_fb
            row += 1
        best_centers = np.array(pts[:n])
        best_radii = np.full(n, r_fb)
        best_sum = np.sum(best_radii)

    # Phase 3: Strict safety scaling to guarantee validity within 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= max(scale * 0.99999999, 0.0)
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
