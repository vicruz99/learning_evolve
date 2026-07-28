# sol_000199 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000193 (state 2527644c) state=124c1a08 sum of radii=2.303972 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

def compute_penalty_and_grad(centers, r):
    """
    Computes the sum of squared violation penalties and its gradient
    for a given configuration of centers and a target radius r.
    Violations occur when circles overlap or exceed boundaries.
    """
    n = centers.shape[0]
    penalty = 0.0
    grad = np.zeros((n, 2))
    
    x, y = centers[:, 0], centers[:, 1]
    
    # Boundary penalties and gradients
    # Left wall: x >= r  => violation = r - x
    v = r - x
    mask = v > 0
    penalty += np.sum(v[mask]**2)
    grad[mask, 0] -= 2.0 * v[mask]
    
    # Right wall: 1-x >= r => violation = r - (1-x)
    v = r - (1.0 - x)
    mask = v > 0
    penalty += np.sum(v[mask]**2)
    grad[mask, 0] += 2.0 * v[mask]
    
    # Bottom wall: y >= r => violation = r - y
    v = r - y
    mask = v > 0
    penalty += np.sum(v[mask]**2)
    grad[mask, 1] -= 2.0 * v[mask]
    
    # Top wall: 1-y >= r => violation = r - (1-y)
    v = r - (1.0 - y)
    mask = v > 0
    penalty += np.sum(v[mask]**2)
    grad[mask, 1] += 2.0 * v[mask]
    
    # Pairwise overlap penalties and gradients
    # dist(i,j) >= 2r => violation = 2r - dist
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    v = 2.0 * r - dist
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask = (v > 0) & triu_mask
    
    viol = v[mask]
    penalty += np.sum(viol**2)
    
    d_dx = dx[mask]
    d_dy = dy[mask]
    d_dist = dist[mask]
    # Gradient of violation^2 wrt centers: -2*viol / dist * diff
    factor = -2.0 * viol / d_dist
    
    i_idx, j_idx = np.where(mask)
    grad[i_idx, 0] += factor * d_dx
    grad[i_idx, 1] += factor * d_dy
    grad[j_idx, 0] -= factor * d_dx
    grad[j_idx, 1] -= factor * d_dy
    
    return penalty, grad

def objective_penalty(x_flat, r, n):
    centers = x_flat.reshape(n, 2)
    pen, _ = compute_penalty_and_grad(centers, r)
    return pen

def gradient_penalty(x_flat, r, n):
    centers = x_flat.reshape(n, 2)
    _, grad = compute_penalty_and_grad(centers, r)
    return grad.flatten()

def try_optimize_r(r, init_centers, n):
    """Attempts to find a valid configuration for radius r starting from init_centers."""
    x0 = init_centers.flatten()
    bounds = [(0.0, 1.0)] * (2 * n)
    res = minimize(objective_penalty, x0, args=(r, n), jac=gradient_penalty,
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 5000, 'ftol': 1e-14})
    return res.fun, res.x.reshape(n, 2)

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub_rows = []
    b_ub_vals = []
    
    # Boundary constraints: r_i <= x, 1-x, y, 1-y
    for i in range(n):
        x, y = centers[i]
        lims = [x, 1.0 - x, y, 1.0 - y]
        for lim in lims:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(lim)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(d)
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-5), 0.0

def generate_hex_centers(n, r0, pattern):
    """Generates a hexagonal lattice configuration based on row counts."""
    pts = []
    y = r0
    row_idx = 0
    for count in pattern:
        shift = r0 if row_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(count):
            if len(pts) >= n:
                break
            if x + r0 <= 1.0:
                pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row_idx += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Diverse row patterns summing to 26
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 4, 5],
        [5, 5, 5, 5, 6], [7, 5, 5, 5, 4]
    ]
    
    inits = []
    for r0 in [0.09, 0.095, 0.10, 0.105]:
        for pat in patterns:
            c = generate_hex_centers(n, r0, pat)
            inits.append(c)
            # Add perturbed versions
            for _ in range(3):
                cp = c + rng.uniform(-0.02, 0.02, (n, 2))
                inits.append(np.clip(cp, 0.05, 0.95))
                
    # Add fully random starts
    for _ in range(8):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Binary search for maximum equal radius r
    low, high = 0.080, 0.115
    feasibility_tol = 1e-7
    
    for _ in range(45):
        if high - low < 1e-7:
            break
        mid = (low + high) / 2.0
        feasible = False
        
        # Try multiple starts for this r
        trial_inits = inits[:10]  # Use first 10 diverse starts
        # Also add perturbation of best known config if available
        if best_centers is not None:
            trial_inits.append(np.clip(best_centers + rng.uniform(-0.015, 0.015, (n, 2)), 0.05, 0.95))
            
        for init_c in trial_inits:
            pen, opt_c = try_optimize_r(mid, init_c, n)
            if pen < feasibility_tol:
                feasible = True
                best_centers = opt_c
                best_radii = np.full(n, mid)
                best_sum = n * mid
                break
        
        if feasible:
            low = mid
        else:
            high = mid
            
    # If binary search failed to find a valid config, fallback
    if best_centers is None:
        best_centers = generate_hex_centers(n, 0.09, [5, 6, 5, 6, 4])
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # Phase 2: LP refinement on fixed centers to allow variable radii
    lp_r, lp_sum = solve_lp_radii(best_centers)
    if lp_sum > best_sum:
        best_sum = lp_sum
        best_radii = lp_r
        
    # Phase 3: Stochastic hill-climbing on centers using LP objective
    curr_centers = best_centers.copy()
    curr_radii, curr_sum = solve_lp_radii(curr_centers)
    best_sum = curr_sum
    best_radii = curr_radii
    best_centers = curr_centers
    
    for step in range(1500):
        i = rng.integers(n)
        old_c = curr_centers[i].copy()
        step_size = 0.020 * (0.997 ** step)
        curr_centers[i] += rng.uniform(-step_size, step_size, 2)
        curr_centers[i] = np.clip(curr_centers[i], 0.02, 0.98)
        
        r_new, s_new = solve_lp_radii(curr_centers)
        if s_new > curr_sum + 1e-9:
            curr_sum = s_new
            curr_radii = r_new
            best_sum = curr_sum
            best_radii = curr_radii.copy()
            best_centers = curr_centers.copy()
        else:
            curr_centers[i] = old_c
            
    # Phase 4: Strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-9:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(best_centers[i, 0] - best_centers[j, 0], 
                           best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
