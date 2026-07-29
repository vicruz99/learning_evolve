# sol_000363 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b037cf31) state=08593bf1 sum of radii=1.880881 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constant for number of circles
N_CIRCLES = 26

def get_penalty(x):
    """
    Calculates the penalty for constraint violations.
    x: array of shape (2*N_CIRCLES + 1,) containing [x1, y1, ..., xN, yN, r]
    """
    # Safety check
    if x.size != 2 * N_CIRCLES + 1:
        return 1e12
    
    # Extract centers and radius
    # x[0:52] are centers, x[52] is radius
    centers = x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = x[2 * N_CIRCLES]
    
    penalty = 0.0
    
    # 1. Boundary Constraints
    # Circles must be inside [0, 1] x [0, 1]
    # r <= x <= 1 - r
    # r <= y <= 1 - r
    
    # Vectorized calculation
    # Left boundary: x_i >= r  => r - x_i <= 0. Violation if r - x_i > 0.
    term_left = r - centers[:, 0]
    penalty += np.sum(np.maximum(0, term_left)**2)
    
    # Right boundary: x_i <= 1 - r => x_i + r - 1 <= 0. Violation if x_i + r - 1 > 0.
    term_right = centers[:, 0] + r - 1.0
    penalty += np.sum(np.maximum(0, term_right)**2)
    
    # Bottom boundary: y_i >= r => r - y_i <= 0. Violation if r - y_i > 0.
    term_bottom = r - centers[:, 1]
    penalty += np.sum(np.maximum(0, term_bottom)**2)
    
    # Top boundary: y_i <= 1 - r => y_i + r - 1 <= 0. Violation if y_i + r - 1 > 0.
    term_top = centers[:, 1] + r - 1.0
    penalty += np.sum(np.maximum(0, term_top)**2)
    
    # 2. Overlap Constraints
    # Distance between centers >= 2r
    # dist(i, j) - 2r >= 0. Violation if dist < 2r.
    
    # Compute pairwise Euclidean distances
    # c[i] - c[j] for all i, j
    # Shape (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    
    # We only care about pairs (i, j) with i < j to avoid double counting
    # Create a mask for the upper triangle
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    
    # Violation amount: 2r - dist
    # If dist < 2r, violation is positive
    violation = 2.0 * r - dist
    
    # Apply mask and sum squared positive violations
    # Only count violations where dist < 2r
    positive_violation = np.maximum(0, violation)
    penalty += np.sum(positive_violation[mask]**2)
    
    return penalty

def objective_function(x):
    """
    Objective function to minimize.
    We want to maximize sum of radii = N * r.
    So we minimize -N * r + penalty.
    """
    r = x[2 * N_CIRCLES]
    # Weight for penalty term
    # Higher weight enforces constraints more strictly
    weight = 5000.0
    return -N_CIRCLES * r + weight * get_penalty(x)

def run_packing():
    """
    Main function to run the packing optimization.
    Returns centers, radii, sum_radii.
    """
    n = N_CIRCLES
    
    # Initialization
    # We start with a valid configuration of small circles to ensure the penalty is 0 initially.
    # A grid layout is a good starting point.
    # We can fit 6x6 grid of points with small radius.
    # Let's use a 6x6 grid of points and pick 26.
    # Points spaced by ~0.18, radius 0.05.
    
    # Generate grid points
    # We want points in [0.05, 0.95] to fit radius 0.05 comfortably
    x_coords = np.linspace(0.05, 0.95, 6)
    y_coords = np.linspace(0.05, 0.95, 6)
    
    points = []
    for y in y_coords:
        for x in x_coords:
            points.append([x, y])
    
    # We have 36 points. Pick 26.
    centers_init = np.array(points[:26])
    r_init = 0.05
    
    # Flatten centers and append r
    # Vector x: [x0, y0, x1, y1, ..., x25, y25, r]
    x0 = np.hstack([centers_init.flatten(), r_init])
    
    # Bounds for optimization
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)]
    
    # Run optimization
    # Using L-BFGS-B which supports bounds. 
    # The penalty method converts constrained problem to unconstrained (w.r.t constraints).
    try:
        res = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
        final_x = res.x
    except Exception:
        # Fallback to initial guess
        final_x = x0
    
    # Extract results
    final_r = final_x[2 * n]
    final_centers = final_x[:2 * n].reshape(n, 2)
    
    # Ensure validity (clamp)
    # Although optimizer should handle it, numerical precision might drift.
    # Clamping centers to [r, 1-r]
    final_centers = np.clip(final_centers, final_r, 1.0 - final_r)
    
    # Radii array
    radii = np.full(n, final_r)
    
    # Sum of radii
    sum_radii = n * final_r
    
    return final_centers, radii, sum_radii
