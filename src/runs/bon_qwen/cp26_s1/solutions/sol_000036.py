# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8e46300b) state=82a3b316 sum of radii=0.538306 correctness=1.0
# stdout(first 200): Circle 7 at (0.9133379619023004, 0.7884285122587618) with radius 0.08666203816352559 is outside the unit square Circles 7 and 19 overlap: dist=6.829798719651631e-09, r1+r2=6.93393556113918e-09
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import time

def compute_objective(vars_array, n):
    """
    Computes the negative sum of radii.
    vars_array: flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = vars_array[2::3]
    return -np.sum(radii)

def compute_constraints(vars_array, n):
    """
    Computes all inequality constraints.
    Returns an array of values that must be >= 0.
    """
    centers = vars_array[:2*n].reshape(n, 2)
    radii = vars_array[2*n:]

    constraints = []

    # 1. Boundary constraints
    # x - r >= 0
    constraints.append(centers[:, 0] - radii)
    # 1 - x - r >= 0
    constraints.append(1.0 - centers[:, 0] - radii)
    # y - r >= 0
    constraints.append(centers[:, 1] - radii)
    # 1 - y - r >= 0
    constraints.append(1.0 - centers[:, 1] - radii)
    
    # 2. Non-overlap constraints
    # dist(i, j) >= r_i + r_j  => dist(i, j) - (r_i + r_j) >= 0
    # Vectorized distance calculation
    # centers shape (n, 2)
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # r_i + r_j shape (n, n)
    sum_radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # We only need upper triangular part (i < j)
    # But computing full matrix and flattening is easier, 
    # though redundant. To save memory/time, let's extract upper triangle.
    # However, simple vectorization is fast enough for N=26.
    # Let's just take the upper triangle values to reduce constraint count.
    indices = np.triu_indices(n, k=1)
    overlap_constraints = dists[indices] - sum_radii_matrix[indices]
    
    constraints.append(overlap_constraints)
    
    return np.concatenate(constraints)

def run_packing():
    n_circles = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * n_circles + [(0, 1)] * n_circles + [(0, 0.5)] * n_circles
    
    # Generate multiple starting configurations
    num_starts = 5
    rng = np.random.RandomState(42) # Fixed seed for reproducibility
    
    start_configs = []
    
    # Config 1: Grid based
    x_grid = np.linspace(0.15, 0.85, 6)
    y_grid = np.linspace(0.15, 0.85, 5)
    coords = []
    count = 0
    for y in y_grid:
        for x in x_grid:
            if count < n_circles:
                coords.append([x, y])
                count += 1
            else:
                break
        if count >= n_circles:
            break
    # Fill remaining if needed (though 6*5=30 > 26)
    while len(coords) < n_circles:
        coords.append([0.5, 0.5])
    start_configs.append(coords)
    
    # Configs 2-5: Random with some spread
    for _ in range(4):
        # Random points in [0.1, 0.9]
        pts = rng.uniform(0.1, 0.9, size=(n_circles, 2))
        start_configs.append(pts)

    for i, initial_centers in enumerate(start_configs):
        # Initialize radii small
        initial_radii = np.full(n_circles, 0.02)
        
        # Flatten to optimization vector
        x0 = np.concatenate([np.array(initial_centers).flatten(), initial_radii])
        
        # Objective function
        def objective(x):
            return compute_objective(x, n_circles)
        
        # Constraint function
        def constraint_fun(x):
            return compute_constraints(x, n_circles)
        
        # Run optimization
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_fun},
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
            
            if res.success or (res.nit > 100): # Allow some failures if it ran long
                centers_opt = res.x[:2*n_circles].reshape(n_circles, 2)
                radii_opt = res.x[2*n_circles:]
                
                # Check validity manually to be safe
                if validate_packing(centers_opt, radii_opt):
                    current_sum = np.sum(radii_opt)
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = centers_opt.copy()
                        best_radii = radii_opt.copy()
        except Exception as e:
            print(f"Optimization start {i} failed: {e}")
            continue

    if best_centers is None:
        # Fallback to a valid packing (e.g. tiny circles)
        best_centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
        best_radii = np.full(n_circles, 0.01)
        best_sum_radii = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum_radii

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

# To run and print results if executed directly
if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    print(f"Valid: {validate_packing(centers, radii)}")
    print("Centers:")
    print(centers)
    print("Radii:")
    print(radii)
