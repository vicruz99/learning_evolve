import numpy as np
import scipy.optimize
import math

# Global constant for number of circles
N_CIRCLES = 26

def objective_function(params):
    """
    Objective function to minimize (negative sum of radii).
    params structure: [x1, y1, x2, y2, ..., x26, y26, r1, r2, ..., r26]
    """
    # Radii are the last N_CIRCLES elements
    radii = params[2 * N_CIRCLES:]
    return -np.sum(radii)

def constraint_inside(params):
    """
    Constraints ensuring all circles are inside the unit square [0,1]x[0,1].
    Returns an array of constraint values that must be >= 0.
    Constraints:
    x_i - r_i >= 0
    1 - (x_i + r_i) >= 0
    y_i - r_i >= 0
    1 - (y_i + r_i) >= 0
    """
    n = N_CIRCLES
    # Extract centers and radii
    centers = params[:2 * n].reshape((n, 2))
    radii = params[2 * n:]
    
    # Vectorized constraint calculation
    c1 = centers[:, 0] - radii          # x - r >= 0
    c2 = 1.0 - centers[:, 0] - radii    # 1 - (x + r) >= 0
    c3 = centers[:, 1] - radii          # y - r >= 0
    c4 = 1.0 - centers[:, 1] - radii    # 1 - (y + r) >= 0
    
    return np.concatenate([c1, c2, c3, c4])

def constraint_no_overlap(params):
    """
    Constraints ensuring no two circles overlap.
    Returns an array of constraint values that must be >= 0.
    Constraint: dist(i, j)^2 - (r_i + r_j)^2 >= 0 for all i < j
    """
    n = N_CIRCLES
    centers = params[:2 * n].reshape((n, 2))
    radii = params[2 * n:]
    
    # Compute pairwise squared distances
    # diff shape: (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dist_sq shape: (n, n)
    dist_sq = np.sum(diff**2, axis=2)
    
    # Compute sum of radii matrix
    # sum_r shape: (n, n)
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Violation: we need dist_sq >= sum_r^2
    violation = dist_sq - sum_r**2
    
    # We only need to enforce constraints for i < j (upper triangle)
    # Extract upper triangle indices
    idx = np.triu_indices(n, k=1)
    return violation[idx]

def get_initial_guess():
    """
    Generate an initial valid configuration of circles.
    Uses a subset of a 6x5 grid to provide 26 well-separated points.
    """
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.09
    
    # Grid dimensions
    cols = 6
    rows = 5
    
    # Generate grid points
    grid_x = np.linspace(0.05, 0.95, cols)
    grid_y = np.linspace(0.05, 0.95, rows)
    
    count = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if count < n:
                centers[count] = [grid_x[c_idx], grid_y[r_idx]]
                count += 1
            else:
                break
        if count >= n:
            break
            
    return centers, radii

def run_packing():
    """
    Main function to pack 26 circles and maximize sum of radii.
    """
    n = N_CIRCLES
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Run optimization multiple times with different random seeds to find global optimum
    for seed in range(10):
        np.random.seed(seed)
        
        # Get initial guess and perturb it
        centers, radii = get_initial_guess()
        
        # Add small random noise to break symmetry and explore space
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        # Ensure centers stay strictly inside bounds for safety
        centers = np.clip(centers, 0.01, 0.99)
        radii[:] = 0.09  # Reset radii to uniform small value
        
        # Flatten parameters for optimizer: [x1, y1, ..., x26, y26, r1, ..., r26]
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Define bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': constraint_inside},
            {'type': 'ineq', 'fun': constraint_no_overlap}
        ]
        
        try:
            # Minimize negative sum of radii
            res = scipy.optimize.minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 5000, 'ftol': 1e-10}
            )
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[:2 * n].reshape((n, 2))
                    best_radii = res.x[2 * n:]
        except Exception:
            continue
            
    # Fallback to a known valid packing if optimization fails or yields invalid result
    if best_centers is None:
        centers_fb = np.zeros((n, 2))
        radii_fb = np.ones(n) * 0.08
        cols = 6
        rows = 5
        # Create a safe grid
        grid_x = np.linspace(0.08, 0.92, cols)
        grid_y = np.linspace(0.08, 0.92, rows)
        count = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if count < n:
                    centers_fb[count] = [grid_x[c_idx], grid_y[r_idx]]
                    count += 1
        return centers_fb, radii_fb, float(np.sum(radii_fb))
    
    return best_centers, best_radii, float(best_sum)