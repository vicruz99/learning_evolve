# sol_000344 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2c580e0d) state=70190bb9 sum of radii=2.541601 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def calculate_penalty(centers, radii, penalty_weight=1000.0):
    n = centers.shape[0]
    boundary_penalty = 0.0
    overlap_penalty = 0.0
    
    # Boundary constraints
    # x - r >= 0  => violation if x < r
    # x + r <= 1  => violation if x > 1 - r
    boundary_violations_x = np.maximum(0, radii - centers[:, 0]) + np.maximum(0, centers[:, 0] + radii - 1.0)
    boundary_violations_y = np.maximum(0, radii - centers[:, 1]) + np.maximum(0, centers[:, 1] + radii - 1.0)
    boundary_penalty = np.sum(boundary_violations_x**2 + boundary_violations_y**2)
    
    # Overlap constraints
    # dist >= r_i + r_j
    # Using broadcasting to calculate distances and sums of radii
    dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Mask out self-interaction (i == j)
    mask = np.ones_like(dist_sq, dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Vectorized penalty calculation for overlaps
    # overlap amount = (r_i + r_j) - dist
    # we only care if overlap amount > 0
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlaps = radii_sum - dist
    overlaps = np.maximum(0, overlaps)
    overlap_penalty = np.sum(overlaps[mask]**2)
    
    # Objective: maximize sum of radii (minimize negative sum)
    # Cost = Penalty - Sum_Radii
    cost = penalty_weight * (boundary_penalty + overlap_penalty) - np.sum(radii)
    return cost

def get_constraints_for_slsqp(centers, radii):
    # This is not used in the penalty method but kept for reference if switching methods
    pass

def run_packing():
    n = 26
    # Initial positions: 5x5 grid plus one in the center gap
    # Grid spacing 0.2, centers at 0.1, 0.3, 0.5, 0.7, 0.9
    x_coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_centers = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Add one circle at a gap, e.g., (0.2, 0.2) is distance ~0.14 from (0.1,0.1)
    extra_center = np.array([[0.2, 0.2]])
    initial_centers = np.vstack([grid_centers, extra_center])
    
    # Initial radii
    initial_radii = np.full(n, 0.05)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[i*3] = initial_centers[i, 0]
        x0[i*3+1] = initial_centers[i, 1]
        x0[i*3+2] = initial_radii[i]
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    
    # Optimization objective
    def objective(vars):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = vars[i*3]
            centers[i, 1] = vars[i*3+1]
            radii[i] = vars[i*3+2]
        return calculate_penalty(centers, radii)

    # Run optimization
    # Using L-BFGS-B for box constraints and gradient-based search
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 2000, 'disp': False})
    
    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i, 0] = res.x[i*3]
        final_centers[i, 1] = res.x[i*3+1]
        final_radii[i] = res.x[i*3+2]
    
    # Final cleanup: ensure strict validity by slightly reducing radii if overlaps are detected numerically
    # This is a safety measure to pass the validation function robustly
    # Re-calculate overlaps and shrink radii if necessary
    n = len(final_radii)
    for _ in range(10): # Iterate a few times to propagate shrinking
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                required_dist = final_radii[i] + final_radii[j]
                if dist < required_dist - 1e-12:
                    # Shrink both radii slightly to resolve overlap
                    shrink = (required_dist - dist) * 0.5
                    final_radii[i] -= shrink
                    final_radii[j] -= shrink
        
        # Check boundaries
        for i in range(n):
            x, y = final_centers[i]
            r = final_radii[i]
            # Ensure inside [0,1]
            if x - r < 0: final_radii[i] = x
            if x + r > 1: final_radii[i] = 1 - x
            if y - r < 0: final_radii[i] = y
            if y + r > 1: final_radii[i] = 1 - y
            final_radii[i] = max(0, final_radii[i])

    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii

# To run and verify locally
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Validation check
    n = centers.shape[0]
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i]-centers[j])**2))
            if d < radii[i] + radii[j] - 1e-9:
                valid = False
    print(f"Valid: {valid}")
