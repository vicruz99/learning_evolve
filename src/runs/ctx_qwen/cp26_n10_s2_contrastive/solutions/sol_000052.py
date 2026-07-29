# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000041 (state 046a36a4) state=461554be sum of radii=0.133154 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to find radii maximizing sum(radii)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((n * (n - 1) // 2, n))
    b_ub = np.zeros(n * (n - 1) // 2)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            b_ub[idx] = d
            idx += 1
            
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 0.01), 0.0

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2 * N:])

def constraints(x):
    """Inequality constraints: boundary and pairwise non-overlap."""
    cx = x[:N]
    cy = x[N:2 * N]
    r = x[2 * N:]
    
    c_bound = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    r_sum = r[:, None] + r[None, :]
    
    c_pair = dists[IDX] - r_sum[IDX]
    
    return np.concatenate([c_bound, c_pair])

def generate_initializations():
    """Create diverse starting points for optimization."""
    configs = []
    for k in range(40):
        rng = np.random.RandomState(k)
        centers = np.zeros((N, 2))
        
        if k < 15:
            # Perturbed 5x5 grid + 1 extra circle in a gap
            pts = []
            for i in range(5):
                for j in range(5):
                    pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
            pts.append([0.2 + rng.uniform(-0.05, 0.05), 0.2 + rng.uniform(-0.05, 0.05)])
            centers = np.array(pts)
            centers += rng.normal(0, 0.015, centers.shape)
        elif k < 25:
            # Hexagonal lattice with varying spacing
            s = 0.18 + 0.02 * rng.uniform(-1, 1)
            pts = []
            row = 0
            while len(pts) < N:
                y = 0.05 + row * s * np.sqrt(3) / 2
                if y > 0.95:
                    break
                x_start = 0.05 + (row % 2) * s / 2
                col = 0
                while x_start + col * s <= 0.95 and len(pts) < N:
                    pts.append([x_start + col * s, y])
                    col += 1
                row += 1
            centers = np.array(pts[:N])
            centers += rng.normal(0, 0.015, centers.shape)
        else:
            # Random placement in safe inner region
            centers = rng.uniform(0.15, 0.85, (N, 2))
            
        centers = np.clip(centers, 0.02, 0.98)
        r, _ = solve_lp_radii(centers)
        x0 = np.concatenate([centers.ravel(), r])
        configs.append(x0)
    return configs

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_x = None
    
    configs = generate_initializations()
    
    # Phase 1: SLSQP optimization from diverse starts
    for x0 in configs:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
        except Exception:
            continue
            
    if best_x is None:
        best_x = configs[0]
        
    cx = best_x[:N].copy()
    cy = best_x[N:2 * N].copy()
    r = best_x[2 * N:].copy()
    
    # Phase 2: Alternating local refinement (Centers -> LP Radii)
    best_local_sum = np.sum(r)
    for step in range(25):
        sigma = 0.012 * max(0.1, 1.0 - step / 25.0)
        pert_cx = cx + np.random.normal(0, sigma, N)
        pert_cy = cy + np.random.normal(0, sigma, N)
        pert_cx = np.clip(pert_cx, 0.01, 0.99)
        pert_cy = np.clip(pert_cy, 0.01, 0.99)
        
        temp_centers = np.column_stack([pert_cx, pert_cy])
        new_r, new_sum = solve_lp_radii(temp_centers)
        
        if new_sum > best_local_sum:
            best_local_sum = new_sum
            cx, cy, r = pert_cx, pert_cy, new_r
            
    # Phase 3: Strict validity enforcement & numerical cleanup
    for i in range(N):
        mx = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        if r[i] > mx:
            r[i] = mx
            
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < r[i] + r[j] - 1e-10:
                    excess = r[i] + r[j] - d
                    r[i] -= excess / 2.0
                    r[j] -= excess / 2.0
                    changed = True
        if not changed:
            break
            
    r = np.maximum(r, 0.0)
    centers = np.column_stack([cx, cy])
    return centers, r, float(np.sum(r))
