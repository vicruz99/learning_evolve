# sol_000360 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b037cf31) state=1abbc592 sum of radii=2.220133 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def solve_radii_lp(centers):
    """
    Solves the Linear Program to maximize sum of radii for fixed centers.
    
    Args:
        centers: np.array of shape (n, 2)
        
    Returns:
        radii: np.array of shape (n)
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints matrix A_ub * x <= b_ub
    # 1. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # 2. Overlap constraints: r_i + r_j <= dist_ij
    
    # Prepare constraints lists
    rows = []
    bounds_rhs = []
    
    # 1. Boundary constraints (4 per circle)
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        bounds_rhs.append(x)
        
        # r_i <= 1 - x
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        bounds_rhs.append(1.0 - x)
        
        # r_i <= y
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        bounds_rhs.append(y)
        
        # r_i <= 1 - y
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        bounds_rhs.append(1.0 - y)
        
    # 2. Overlap constraints (r_i + r_j <= dist)
    # Calculate distances between all pairs
    dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis] - centers[np.newaxis, :]) ** 2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            bounds_rhs.append(dist_matrix[i, j])
            
    A_ub = np.array(rows)
    b_ub = np.array(bounds_rhs)
    
    # Bounds for variables (radii >= 0)
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # method='highs' is usually available and efficient in recent scipy versions
    try:
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except ValueError:
        # Fallback to simplex if highs is not available or fails
        try:
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')
        except:
            # Return zeros if all fails
            return np.zeros(n)
            
    if res.success:
        return res.x
    else:
        # Fallback to small radii if LP fails
        return np.full(n, 0.01)

def get_total_radii(centers):
    """Helper to get objective value for SA"""
    radii = solve_radii_lp(centers)
    return np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square.
    """
    n = 26
    np.random.seed(42)
    
    # --- Step 1: Initialize Centers ---
    # Start with a hexagonal-like grid pattern for better convergence
    centers = np.zeros((n, 2))
    
    # Approximate radius for 26 circles ~ 0.101
    # We can fit roughly 5 circles per row in a grid. 
    # Let's create a perturbed grid.
    # 5 rows of 5 circles = 25. We need 1 more.
    # Let's place them in a 6x5 grid pattern and pick the best 26, or just random perturbation of 5x5 + 1
    
    # Base grid coordinates for 5x5
    step = 1.0 / 5.0
    grid_x = np.linspace(step/2, 1-step/2, 5)
    grid_y = np.linspace(step/2, 1-step/2, 5)
    
    # Create 5x5 grid
    grid_centers = np.array([[x, y] for y in grid_y for x in grid_x])
    
    # Add 1 extra circle. Where? Maybe center of a gap or random.
    # Let's try placing the 26th circle at the center of the square initially
    extra_center = np.array([0.5, 0.5])
    
    centers = np.vstack([grid_centers, extra_center])
    
    # Add some random noise to break symmetry
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    
    # Clip to ensure inside [0, 1]
    centers = np.clip(centers, 0.01, 0.99)

    # --- Step 2: Simulated Annealing to Optimize Centers ---
    
    current_centers = centers.copy()
    current_val = get_total_radii(current_centers)
    
    best_centers = current_centers.copy()
    best_val = current_val
    
    # SA Parameters
    T = 0.5 # Initial temperature
    alpha = 0.99 # Cooling rate
    min_T = 1e-6
    
    # Number of iterations per temperature step
    n_iter_per_T = 50 
    
    # Total steps roughly: 50 * log(0.5/min_T)/log(0.99) ~ 50 * 270 ~ 13500 iterations.
    # This is fast enough.
    
    import time
    start_time = time.time()
    
    # Limit runtime to avoid timeout, e.g., 2 seconds
    max_time = 2.0 
    
    while T > min_T and (time.time() - start_time) < max_time:
        for _ in range(n_iter_per_T):
            # Generate neighbor solution by moving one random circle
            trial_centers = current_centers.copy()
            idx = np.random.randint(0, n)
            
            # Random move
            move_size = T * 0.1
            dx = np.random.uniform(-move_size, move_size)
            dy = np.random.uniform(-move_size, move_size)
            
            trial_centers[idx, 0] = np.clip(trial_centers[idx, 0] + dx, 0.001, 0.999)
            trial_centers[idx, 1] = np.clip(trial_centers[idx, 1] + dy, 0.001, 0.999)
            
            trial_val = get_total_radii(trial_centers)
            
            # Acceptance criterion
            if trial_val > current_val:
                current_centers = trial_centers
                current_val = trial_val
                if current_val > best_val:
                    best_centers = current_centers.copy()
                    best_val = current_val
            else:
                # Probability of acceptance
                prob = math.exp((trial_val - current_val) / T)
                if np.random.rand() < prob:
                    current_centers = trial_centers
                    current_val = trial_val
        
        T *= alpha
        
        # Occasionally perturb all centers to escape deep local minima if stuck
        if T < 0.01 and np.random.rand() < 0.1:
             for i in range(n):
                current_centers[i] += np.random.uniform(-T, T, 2)
                current_centers[i] = np.clip(current_centers[i], 0.001, 0.999)
             current_val = get_total_radii(current_centers)

    # --- Step 3: Final Radius Calculation ---
    # The best_centers found might have slightly suboptimal radii if we just took the last LP result.
    # Recompute radii for the best centers found.
    final_radii = solve_radii_lp(best_centers)
    final_sum = np.sum(final_radii)
    
    return best_centers, final_radii, final_sum

if __name__ == "__main__":
    # Just for local testing if needed, but run_packing is the required entry point
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
