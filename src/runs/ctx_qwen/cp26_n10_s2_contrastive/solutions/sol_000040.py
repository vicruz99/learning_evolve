# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 33c0c451) state=dbbb12bc sum of radii=1.340320 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix
import math

N_CIRCLES = 26

def solve_radii_lp(centers):
    """
    Given fixed centers, solve LP to find radii that maximize sum(radii)
    subject to non-overlap and boundary constraints.
    """
    n = len(centers)
    c_obj = -np.ones(n)
    
    # Precompute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Build sparse A_ub for pairwise constraints: r_i + r_j <= d_ij
    rows, cols, vals = [], [], []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(idx)
            cols.append(i)
            vals.append(1.0)
            cols.append(j)
            vals.append(1.0)
            rows.append(idx)
            idx += 1
            
    A_ub = coo_matrix((vals, (rows, cols)), shape=(idx, n)).tocsr()
    b_ub = dists[np.triu_indices(n, k=1)]
    
    # Bounds: 0 <= r_i <= min distance to boundary
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, max_r)))
        
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, -res.fun
    return np.zeros(n), 0.0

def objective_full(x):
    return -np.sum(x[2 * N_CIRCLES:])

def constraints_full(x):
    cx = x[0:N_CIRCLES]
    cy = x[N_CIRCLES:2 * N_CIRCLES]
    r = x[2 * N_CIRCLES:]
    
    cons = []
    # Boundary constraints
    cons.extend(cx - r)
    cons.extend(1.0 - cx - r)
    cons.extend(cy - r)
    cons.extend(1.0 - cy - r)
    
    # Overlap constraints (squared for smoothness)
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            cons.append(d2 - r_sum * r_sum)
            
    return np.array(cons)

def init_hex(r_start=0.06, noise=0.01):
    centers = []
    y = r_start
    row = 0
    while len(centers) < N_CIRCLES:
        x = r_start + (r_start if row % 2 == 1 else 0)
        while x <= 1.0 - r_start and len(centers) < N_CIRCLES:
            centers.append([x, y])
            x += 2 * r_start
        y += r_start * math.sqrt(3)
        row += 1
        
    centers = np.array(centers[:N_CIRCLES])
    centers += np.random.randn(*centers.shape) * noise
    centers = np.clip(centers, r_start, 1.0 - r_start)
    r = np.full(N_CIRCLES, r_start)
    return np.concatenate([centers.ravel(), r])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = 0.0
    best_centers = None
    
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints_full}
    
    # Strategy 1: Hexagonal lattice starts with varying densities
    for trial in range(12):
        np.random.seed(trial)
        r0 = 0.045 + trial * 0.004
        x0 = init_hex(r_start=r0, noise=0.004)
        
        res = minimize(objective_full, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 2500, 'ftol': 1e-12, 'disp': False})
        if res.success and -res.fun > best_sum:
            best_sum = -res.fun
            best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2).copy()
            
    # Strategy 2: Random perturbed starts
    for trial in range(10):
        np.random.seed(trial + 100)
        cx = np.random.uniform(0.15, 0.85, N_CIRCLES)
        cy = np.random.uniform(0.15, 0.85, N_CIRCLES)
        r = np.full(N_CIRCLES, 0.035)
        x0 = np.concatenate([cx, cy, r])
        
        res = minimize(objective_full, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 2500, 'ftol': 1e-12, 'disp': False})
        if res.success and -res.fun > best_sum:
            best_sum = -res.fun
            best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2).copy()
            
    # Strategy 3: Local perturbation refinement of best found so far
    if best_centers is not None:
        for trial in range(8):
            np.random.seed(trial + 200)
            x_curr = np.concatenate([best_centers.ravel(), np.full(N_CIRCLES, 0.08)])
            x_curr[:2 * N_CIRCLES] += np.random.randn(2 * N_CIRCLES) * 0.006
            x_curr[2 * N_CIRCLES:] += np.random.randn(N_CIRCLES) * 0.003
            x_curr = np.clip(x_curr, 1e-5, 0.995)
            
            res = minimize(objective_full, x_curr, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if res.success and -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2).copy()
                
    if best_centers is None:
        # Fallback to safe grid
        best_centers = np.array([(0.1 + 0.18*i, 0.1 + 0.18*j) for i in range(5) for j in range(5)])
        best_centers = np.vstack([best_centers, [0.5, 0.5]])
        
    # Clip centers strictly inside
    best_centers = np.clip(best_centers, 1e-9, 1.0 - 1e-9)
    
    # LP Refinement: Fix centers and exactly optimize radii
    radii, lp_sum = solve_radii_lp(best_centers)
    if lp_sum > best_sum:
        best_sum = lp_sum
        
    # Final strict feasibility correction
    for i in range(N_CIRCLES):
        x, y = best_centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if radii[i] > max_r + 1e-12:
            radii[i] = max(0.0, max_r - 1e-10)
            
    # Iterative overlap resolution
    for _ in range(30):
        overlap_found = False
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < radii[i] + radii[j] - 1e-12:
                    excess = radii[i] + radii[j] - d
                    radii[i] -= excess / 2.0
                    radii[j] -= excess / 2.0
                    radii[i] = max(0.0, radii[i])
                    radii[j] = max(0.0, radii[j])
                    overlap_found = True
        if not overlap_found:
            break
            
    final_sum = np.sum(radii)
    return best_centers, radii, float(final_sum)
