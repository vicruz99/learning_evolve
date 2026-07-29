# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2a4ed9f3) state=dfcbd588 sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(centers, radii, weight=10000.0):
    """
    Computes penalty for boundary violations and overlaps.
    Uses squared hinge loss for smoothness.
    
    Args:
        centers: np.array of shape (n, 2)
        radii: np.array of shape (n)
        weight: Penalty weight
        
    Returns:
        float: Total penalty
    """
    n = centers.shape[0]
    penalty = 0.0
    
    # Boundary violations
    # Constraints: r <= x <= 1-r and r <= y <= 1-r
    # Violations occur if x < r, x > 1-r, y < r, or y > 1-r
    
    # x - r >= 0 => violation if r - x > 0
    v1 = np.maximum(0, radii - centers[:, 0])
    # 1 - r - x >= 0 => violation if x + r - 1 > 0
    v2 = np.maximum(0, centers[:, 0] + radii - 1.0)
    # y - r >= 0 => violation if r - y > 0
    v3 = np.maximum(0, radii - centers[:, 1])
    # 1 - r - y >= 0 => violation if y + r - 1 > 0
    v4 = np.maximum(0, centers[:, 1] + radii - 1.0)
    
    pen_bound = np.sum(v1**2 + v2**2 + v3**2 + v4**2)
    
    # Overlap violations
    # Constraint: ||c_i - c_j|| >= r_i + r_j
    # Violation if r_i + r_j - ||c_i - c_j|| > 0
    
    # Compute pairwise distances efficiently
    # c_i - c_j shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Sum of radii matrix
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Violation matrix
    overlap_viol = np.maximum(0, sum_r - dist)
    
    # Sum of squared violations (upper triangle only)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pen_overlap = np.sum(overlap_viol[mask]**2)
    
    return weight * (pen_bound + pen_overlap)

def objective(vars, n):
    """
    Objective function to minimize: -sum(radii) + penalty
    vars contains [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    centers = np.column_stack((vars[0::3], vars[1::3]))
    radii = vars[2::3]
    
    # Ensure non-negative radii for penalty calculation
    radii_safe = np.maximum(radii, 0.0)
    
    penalty = compute_penalty(centers, radii_safe, weight=10000.0)
    
    # We want to maximize sum(radii), so minimize -sum(radii)
    return -np.sum(radii) + penalty

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for optimization: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    def generate_grid_start():
        """Generate a 5x5 grid plus one extra circle."""
        pts = []
        # 5x5 grid centers
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
        # 26th circle placed in a void, e.g., (0.6, 0.6)
        # (0.6, 0.6) is roughly centered between grid points (0.5, 0.5), (0.7, 0.5), etc.
        pts.append([0.6, 0.6])
        return np.array(pts)

    def generate_hex_start():
        """Generate a hexagonal-like arrangement of 26 circles."""
        pts = []
        # Attempt to fit 26 circles in 5 rows with varying counts
        # Row 0: 6 circles
        for i in range(6):
            pts.append([0.08 + 0.15*i, 0.1]) 
        # Row 1: 5 circles (offset)
        for i in range(5):
            pts.append([0.13 + 0.15*i, 0.25])
        # Row 2: 6 circles
        for i in range(6):
            pts.append([0.08 + 0.15*i, 0.4])
        # Row 3: 5 circles (offset)
        for i in range(5):
            pts.append([0.13 + 0.15*i, 0.55])
        # Row 4: 4 circles
        for i in range(4):
            pts.append([0.13 + 0.15*i, 0.7])
        return np.array(pts)

    starts = []
    
    # 1. Grid start
    pts = generate_grid_start()
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pts[i, 0]
        x0[3*i+1] = pts[i, 1]
        x0[3*i+2] = 0.06 # Start with a feasible radius
    starts.append(x0)
    
    # 2. Hexagonal start
    pts = generate_hex_start()
    if len(pts) == 26:
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = pts[i, 0]
            x0[3*i+1] = pts[i, 1]
            x0[3*i+2] = 0.05
        starts.append(x0)
        
    # 3. Random starts
    for _ in range(4):
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = np.random.rand()
            x0[3*i+1] = np.random.rand()
            x0[3*i+2] = 0.05
        starts.append(x0)
        
    # 4. Perturbed Grid start
    pts = generate_grid_start()
    pts = pts + np.random.randn(*pts.shape) * 0.02
    pts = np.clip(pts, 0.05, 0.95)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pts[i, 0]
        x0[3*i+1] = pts[i, 1]
        x0[3*i+2] = 0.06
    starts.append(x0)

    for x0 in starts:
        try:
            # Optimize
            res = minimize(objective, x0, args=(n,), method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 3000, 'ftol': 1e-12})
            opt_vars = res.x
            centers = np.column_stack((opt_vars[0::3], opt_vars[1::3]))
            radii = opt_vars[2::3]
            
            # Verify validity
            # Recompute penalty with weight 1 to check magnitude
            pen = compute_penalty(centers, radii, weight=1.0)
            
            # If penalty is negligible, it's a valid packing
            if pen < 1e-9:
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
        except Exception:
            pass

    # Fallback if no valid packing found (should not happen with good starts)
    if best_centers is None:
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.05)
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
        centers[idx] = [0.6, 0.6]
        best_sum = np.sum(radii)
        best_centers = centers
        best_radii = radii

    return best_centers, best_radii, best_sum
