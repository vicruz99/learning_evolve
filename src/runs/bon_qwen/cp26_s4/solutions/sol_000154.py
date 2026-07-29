# sol_000154 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a1c97a27) state=6c5a464a sum of radii=2.193047 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses numerical optimization with penalty constraints.
    """
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Penalty weights - high values to enforce constraints strictly
    W_bound = 50000.0
    W_overlap = 50000.0

    def objective(params):
        # params layout: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
        cx = params[0::3]
        cy = params[1::3]
        cr = params[2::3]
        
        # Objective: maximize sum of radii -> minimize negative sum
        obj = -np.sum(cr)
        
        # Penalty for boundaries
        # Constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        # Violations: max(0, r - x), etc.
        v1 = np.maximum(0, cr - cx)
        v2 = np.maximum(0, cr - (1 - cx))
        v3 = np.maximum(0, cr - cy)
        v4 = np.maximum(0, cr - (1 - cy))
        pen_bound = W_bound * (np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2))
        
        # Penalty for overlaps
        # Constraint: dist(c_i, c_j) >= r_i + r_j
        # Violation: max(0, r_i + r_j - dist)
        
        # Vectorized distance calculation
        diff_x = cx[:, np.newaxis] - cx[np.newaxis, :]
        diff_y = cy[:, np.newaxis] - cy[np.newaxis, :]
        dists = np.sqrt(diff_x**2 + diff_y**2)
        sum_r = cr[:, np.newaxis] + cr[np.newaxis, :]
        
        # Mask upper triangle to avoid double counting and self-interaction
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        viol = sum_r[mask] - dists[mask]
        pen_overlap = W_overlap * np.sum(np.maximum(0, viol)**2)
        
        return obj + pen_bound + pen_overlap

    def is_valid(params):
        cx = params[0::3]
        cy = params[1::3]
        cr = params[2::3]
        
        if np.any(cr < -1e-9): return False
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9): return False
        if np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9): return False
        
        diff_x = cx[:, np.newaxis] - cx[np.newaxis, :]
        diff_y = cy[:, np.newaxis] - cy[np.newaxis, :]
        dists = np.sqrt(diff_x**2 + diff_y**2)
        sum_r = cr[:, np.newaxis] + cr[np.newaxis, :]
        
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.any(dists[mask] < sum_r[mask] - 1e-9): return False
        
        return True

    def run_opt(x0):
        bounds = []
        for _ in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
        try:
            # L-BFGS-B is efficient for box constraints
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            return res
        except Exception:
            return None

    best_params = None
    
    # Strategy 1: Random starts to explore space
    for seed in range(5):
        np.random.seed(seed)
        cx = np.random.rand(n)
        cy = np.random.rand(n)
        cr = np.full(n, 0.01) # Start small
        x0 = np.empty(3*n)
        x0[0::3] = cx
        x0[1::3] = cy
        x0[2::3] = cr
        
        res = run_opt(x0)
        if res is not None:
            if is_valid(res.x):
                s = np.sum(res.x[2::3])
                if s > best_sum:
                    best_sum = s
                    best_params = res.x
    
    # Strategy 2: Hexagonal-like grid initialization
    # Try to place circles in a staggered pattern (dense packing)
    a = 0.2 # Approx spacing
    pts = []
    y_step = a * math.sqrt(3) / 2
    # y levels
    y_vals = np.arange(0.1, 1.0, y_step)
    for i, y in enumerate(y_vals):
        x_offset = a/2 if i % 2 == 1 else 0.0
        x_vals = np.arange(0.1 + x_offset, 1.0, a)
        for x in x_vals:
            if 0 <= x <= 1:
                pts.append((x, y))
    
    init_cx = [p[0] for p in pts]
    init_cy = [p[1] for p in pts]
    
    # If we have fewer than n points, pad with random
    if len(init_cx) < n:
        np.random.seed(42)
        for _ in range(n - len(init_cx)):
            init_cx.append(np.random.rand())
            init_cy.append(np.random.rand())
    else:
        # Take first n points
        init_cx = init_cx[:n]
        init_cy = init_cy[:n]
    
    x0 = np.empty(3*n)
    x0[0::3] = init_cx
    x0[1::3] = init_cy
    x0[2::3] = np.full(n, 0.05)
    
    res = run_opt(x0)
    if res is not None:
        if is_valid(res.x):
            s = np.sum(res.x[2::3])
            if s > best_sum:
                best_sum = s
                best_params = res.x
                
    # Strategy 3: 5x5 grid + 1 circle
    cx = []
    cy = []
    for i in range(5):
        for j in range(5):
            cx.append(0.1 + 0.2*i)
            cy.append(0.1 + 0.2*j)
    # Add 26th circle at center (occupied, but optimizer will move)
    cx.append(0.5)
    cy.append(0.5)
    
    x0 = np.empty(3*n)
    x0[0::3] = cx
    x0[1::3] = cy
    x0[2::3] = np.full(n, 0.05)
    
    res = run_opt(x0)
    if res is not None:
        if is_valid(res.x):
            s = np.sum(res.x[2::3])
            if s > best_sum:
                best_sum = s
                best_params = res.x

    if best_params is not None:
        cx = best_params[0::3]
        cy = best_params[1::3]
        cr = best_params[2::3]
        
        # Ensure non-negative radii
        cr = np.maximum(cr, 0)
        
        # Clip radii to strictly satisfy boundary constraints
        # This ensures x-r >= 0, etc.
        cr = np.minimum(cr, cx)
        cr = np.minimum(cr, 1 - cx)
        cr = np.minimum(cr, cy)
        cr = np.minimum(cr, 1 - cy)
        
        best_centers = np.column_stack((cx, cy))
        best_radii = cr
        best_sum = np.sum(best_radii)
    else:
        # Fallback: Random valid packing
        np.random.seed(123)
        # Place centers in [0.1, 0.9] to ensure small radii fit
        best_centers = 0.1 + 0.8 * np.random.rand(n, 2)
        best_radii = np.full(n, 0.01)
        best_sum = n * 0.01

    return best_centers, best_radii, best_sum
