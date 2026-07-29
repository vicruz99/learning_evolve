# sol_000021 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f223c9a2) state=408e5c92 sum of radii=2.134912 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array (26, 2)
        radii: np.array (26)
        sum_radii: float
    """
    n = 26
    
    # --- 1. Initialization: Staggered Grid Layout ---
    centers = []
    row_counts = [5, 6, 5, 5, 5] # Total 26
    # Approximate dimensions for staggered layout
    dy = 1.0 / 6.0
    for i, count in enumerate(row_counts):
        # Stagger even rows
        shift = 0.5 if (i % 2 == 1) else 0.0
        dx = 1.0 / (count + 1)
        for j in range(count):
            x = dx * (j + 1) + shift * (dx / 2)
            y = dy * (i + 1)
            # Clamp to bounds slightly away from edges for stability
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # --- 2. Optimization Loop ---
    # We will iterate: adjust centers -> solve LP for radii
    
    for iteration in range(15): # Run multiple refinement steps
        
        # Solve LP for optimal radii given current centers
        radii = solve_radii_lp(centers, n)
        
        # Check if we have a valid packing
        if not validate_packing(centers, radii):
            # Fallback to safe equal radii if LP fails or is invalid
            r_max = 0.04
            centers = generate_valid_grid(n, r_max)
            radii = np.full(n, r_max)
            break

        # Objective: Minimize overlap "pressure" to allow radius expansion
        # We define a penalty based on how close circles are to touching
        # Penalty = sum( (r_i + r_j - dist_ij)^2 ) if dist_ij < r_i + r_j
        
        # Use a local search to perturb centers
        # We try to move centers away from each other slightly
        def overlap_penalty(c_flat):
            c = c_flat.reshape(-1, 2)
            penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((c[i] - c[j])**2))
                    # We use current radii as a baseline for what we want to maintain
                    # If dist < r_i + r_j, we have overlap.
                    # We want to push dist to be >= r_i + r_j.
                    r_sum = radii[i] + radii[j]
                    if dist < r_sum:
                        # Quadratic penalty for overlap
                        penalty += (r_sum - dist)**2
            # Also penalize if centers are too close to boundaries relative to radii
            # But LP handles radii expansion, so we just want to move centers 
            # to create room.
            return penalty

        # Bounds for centers (must stay in square, leaving room for small radii)
        bnds = [(0.001, 0.999) for _ in range(2 * n)]
        
        # Optimize centers to reduce overlap penalty
        result = opt.minimize(overlap_penalty, centers.flatten(), 
                              method='L-BFGS-B', bounds=bnds, 
                              options={'ftol': 1e-12, 'maxiter': 200})
        
        centers = result.x.reshape(-1, 2)
        
        # After moving centers, re-solve LP to inflate radii
        radii = solve_radii_lp(centers, n)

    return centers, radii, np.sum(radii)

def solve_radii_lp(centers, n):
    """
    Solves the Linear Programming problem to maximize sum of radii 
    for a fixed set of centers.
    """
    # Variables: r_0, ..., r_{n-1}
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    c_obj = np.ones(n) * -1 
    
    # Constraints:
    # 1. Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    #    r_i <= bound_i  =>  [1, 0, ...] * r <= bound_i
    # 2. Non-overlap constraints: r_i + r_j <= dist_ij
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    bounds_limit = np.full(n, 1.0)
    for i in range(n):
        x, y = centers[i]
        limit = min(x, 1-x, y, 1-y)
        bounds_limit[i] = limit
        # r_i <= limit
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(limit)
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            # r_i + r_j <= dist
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for variables: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback to small valid radii if LP fails
        return np.full(n, 0.001)

def generate_valid_grid(n, r):
    """Generates a simple grid of non-overlapping circles as a fallback."""
    centers = []
    # Simple square grid approximation
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    dx = 1.0 / (cols + 1)
    dy = 1.0 / (rows + 1)
    
    count = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if count < n:
                x = dx * (c_idx + 1)
                y = dy * (r_idx + 1)
                centers.append([x, y])
                count += 1
    return np.array(centers)

def validate_packing(centers, radii):
    """
    Validates that circles don't overlap and are inside the unit square.
    """
    n = centers.shape[0]
    
    # Check for NaN values
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    # Check if radii are nonnegative
    if np.any(radii < 0):
        return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True
