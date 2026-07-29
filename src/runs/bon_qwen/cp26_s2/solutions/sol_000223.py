# sol_000223 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=5620d1bc sum of radii=0.000650 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(v, n, lambda_pen):
    """
    Computes the objective function value.
    Objective: Maximize sum of radii => Minimize -sum(r) + penalty.
    
    Args:
        v: array of shape (3*n,) containing [x0, y0, r0, x1, y1, r1, ...]
        n: number of circles
        lambda_pen: penalty weight for constraint violations
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Primary objective: maximize sum of radii
    obj = -np.sum(r)
    
    # 1. Overlap Penalty
    # Calculate pairwise distances
    # dx matrix of shape (n, n)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # r_sum matrix of shape (n, n)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Overlap amount: positive if overlapping
    overlap = r_sum - dist
    
    # Apply penalty only for overlaps (positive values)
    # We sum the square of the violation
    # Note: dist is symmetric, overlap is symmetric. 
    # Summing all elements counts each pair twice (i,j) and (j,i).
    # Diagonal is 0.
    overlap_penalty = np.sum(np.maximum(0, overlap)**2) / 2.0
    
    # 2. Boundary Penalty
    # Constraints:
    # x - r >= 0  => violation if r - x > 0
    # x + r <= 1  => violation if r + x - 1 > 0
    # y - r >= 0  => violation if r - y > 0
    # y + r <= 1  => violation if r + y - 1 > 0
    
    pen_x_left = np.sum(np.maximum(0, r - x)**2)
    pen_x_right = np.sum(np.maximum(0, r + x - 1)**2)
    pen_y_bottom = np.sum(np.maximum(0, r - y)**2)
    pen_y_top = np.sum(np.maximum(0, r + y - 1)**2)
    
    boundary_penalty = pen_x_left + pen_x_right + pen_y_bottom + pen_y_top
    
    total_penalty = overlap_penalty + boundary_penalty
    return obj + lambda_pen * total_penalty

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    best_v = None
    best_loss = np.inf
    
    # We try multiple random restarts to find a good local optimum
    # 20 restarts should be sufficient for n=26
    num_restarts = 20
    
    # Penalty weight. Higher is stricter but might be harder to optimize.
    # 10000 is usually safe for unit square scale.
    lambda_pen = 10000.0
    
    bounds = []
    for _ in range(n):
        bounds.extend([
            (0.0, 1.0), # x
            (0.0, 1.0), # y
            (0.0, 0.5)  # r (cannot be larger than 0.5)
        ])

    for seed in range(num_restarts):
        rng = np.random.RandomState(seed)
        
        # Initialize centers randomly in the middle region to avoid boundary issues initially
        # [0.2, 0.8] is a safe starting box
        x0 = rng.uniform(0.2, 0.8, n)
        y0 = rng.uniform(0.2, 0.8, n)
        # Small initial radii
        r0 = 0.05 * np.ones(n)
        
        v0 = np.empty(3 * n)
        v0[0::3] = x0
        v0[1::3] = y0
        v0[2::3] = r0
        
        # Run optimization
        try:
            res = minimize(compute_loss, v0, args=(n, lambda_pen), method='L-BFGS-B', 
                           bounds=bounds, options={'ftol': 1e-12, 'maxiter': 1000})
            
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_v = res.x
        except Exception:
            continue

    if best_v is None:
        # Fallback if optimization failed completely
        # Return a trivial valid packing
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i] = [0.5, 0.5]
            radii[i] = 0.0
        return centers, radii, 0.0

    # Extract results
    x = best_v[0::3]
    y = best_v[1::3]
    r = best_v[2::3]
    
    # Ensure non-negative radii (numerical noise might cause tiny negatives)
    r = np.maximum(r, 0.0)
    
    centers = np.column_stack((x, y))
    
    # Validate and clean up tiny negative values if any
    # (Though optimizer bounds should handle this)
    
    # Check for validity (debugging info not required but good practice)
    # The problem statement asks to return centers, radii, sum_radii
    
    sum_radii = np.sum(r)
    
    return centers, r, sum_radii

# To run and print result (not part of the required function, but for testing)
if __name__ == "__main__":
    centers, radii, total_r = run_packing()
    print(f"Sum of radii: {total_r}")
    
    # Optional: Validate with the provided function logic
    # (Assuming validate_packing is available in context, but here we just trust our logic)
    # We can implement a quick check
    valid = True
    for i in range(len(radii)):
        xi, yi = centers[i]
        ri = radii[i]
        if ri < 0 or xi - ri < 0 or xi + ri > 1 or yi - ri < 0 or yi + ri > 1:
            valid = False
            break
        for j in range(i + 1, len(radii)):
            dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if dist < ri + radii[j] - 1e-12:
                valid = False
                break
    print(f"Valid: {valid}")
